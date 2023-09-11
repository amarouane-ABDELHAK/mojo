from patchify import patchify, unpatchify
import numpy as np
import boto3
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import rasterio
import os
import uuid
import cv2
import urllib.parse
from argparse import ArgumentParser
import datetime
import requests
import geopandas as gpd
from rasterio.features import shapes
from geometry_transformation import project_geojson_file
import re

tf.keras.utils.disable_interactive_logging()


def download_s3_file(s3_client, s3_uri, destination_folder="/tmp"):
    parsed_url = urllib.parse.urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    s3_key = parsed_url.path.lstrip("/")
    file_name = os.path.basename(s3_key)
    destination_file_path = f'{destination_folder.rstrip("/")}/{file_name}'
    s3_client.download_file(bucket_name, s3_key, destination_file_path)
    return destination_file_path


def generate_gtif(bitmap, profile, outfile):
    """
    Generates GeoTiff for a bitmap
    :param bitmap: (numpy arr) Bitmap array
    :param profile: Profile data from the original scene
    :param outfile: Name of the output GeoTiff file
    :return: None
    """
    ras_meta = profile
    ras_meta["count"] = 1
    ras_meta["nodata"] = 0
    ras_meta["width"] = bitmap.shape[1]
    ras_meta["height"] = bitmap.shape[0]
    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(bitmap, 1)


def push_bulk_items(req_session, items_list):
    bearer_token = os.getenv("BEARER_TOKEN")
    headers = {"Authorization": "Bearer " + bearer_token}
    endpoint = os.getenv("ENDPOINT")
    response = req_session.post(url=endpoint, headers=headers, json=items_list)
    print(response.json())
    return response.status_code


def push_to_stac(planet_item_id, path_geojson, batch_to_ingest=500):
    req_session = requests.Session()
    today = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%dT%H:%M:%SZ")
    data = project_geojson_file(in_file=path_geojson)
    batch = {"items": {}}
    counter = 0
    request_states = {}
    for tile in data:
        counter += 1
        id = str(uuid.uuid4())
        args = {
            "collection": "sredd",
            "properties": {"datetime": today, "event": "lake"},
            "id": id,
            "stac_version": "1.0.0",
            "links": [
                {
                    "href": f"/collections/planet/{planet_item_id}",
                    "rel": "derived_from",
                    "type": "item",
                }
            ],
        }
        tile.update(**args)
        batch["items"].update({id: tile})
        if counter == batch_to_ingest:
            status = push_bulk_items(req_session=req_session, items_list=batch)
            request_states[status] = request_states.get(status, 0) + 1
            batch = {"items": {}}
            counter = 0
    if batch:
        status = push_bulk_items(req_session=req_session, items_list=batch)
        request_states[status] = request_states.get(status, 0) + 1
    return request_states


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="sredd",
        description="Generate SREDD mimage",
        epilog="Contact Sujit/Ashish/Marouane for extra help",
    )
    parser.add_argument(
        "--list-planet-tiffs",
        dest="list_input_planet_tiffs",
        help="List of planet tiffs",
        required=True,
    )
    parser.add_argument(
        "--model-s3uri",
        dest="model_s3uri",
        help="Model S3URI",
        required=True,
    )
    parser.add_argument(
        "--lakenet-model-s3uri",
        dest="lakenet_model_s3uri",
        help="Model S3URI",
        required=True,
    )
    parser.add_argument(
        "--destination-bucket",
        dest="destination_bct",
        help="Bucket name destination for geojson",
        required=True,
    )
    args = parser.parse_args()

    dest_bucket, list_input_planet_tiffs, model_s3uri, lakenet_model_s3uri = (
        args.destination_bct,
        args.list_input_planet_tiffs.split(","),
        args.model_s3uri,
        args.lakenet_model_s3uri,
    )
    reconstructed_image = None
    s3_client = boto3.client("s3")
    count = 0
    model_fs_path = download_s3_file(s3_client, model_s3uri)
    lakenet_model_fs_path = download_s3_file(s3_client, lakenet_model_s3uri)
    with open(model_fs_path, "r") as json_file:
        loaded_model_json = json_file.read()
    test_unet_model = model_from_json(loaded_model_json)
    task_results = list()
    test_unet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=False),
    )
    with open(lakenet_model_fs_path, "r") as lakenet_model_file:
        test_unet_model.load_weights(lakenet_model_fs_path)
    for path_to_tiffs in list_input_planet_tiffs:
        s3_object_regex = r"s3://[^/]+/.+/planet/(?P<planet_item_id>.*)/assets/.*\.tif"
        match = re.fullmatch(s3_object_regex, path_to_tiffs)
        if not match:
            raise Exception(f"Can't get the planet item id from {path_to_tiffs}")
        planet_item_id = match.group("planet_item_id")

        input_planet_file_name = os.path.basename(path_to_tiffs)
        input_planet_file_name_wo_extension = input_planet_file_name[
            : input_planet_file_name.index(".")
        ]
        with rasterio.open(path_to_tiffs, dtype="uint8", nodata=1) as src:
            large_image = src.read()
            fixed_scene = np.moveaxis(large_image, 0, -1)
        fixed_scene1 = (fixed_scene.astype(np.uint8)) / 255
        patches = patchify(fixed_scene1, (128, 128, 4), step=128)
        # print("Large image shape is: ", large_image.shape)
        # print("Patches array shape is: ", patches.shape)

        predicted_patches = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                # print("Now predicting on patch", i,j)

                single_patch = patches[i, j, :, :, :]
                # print('single_path_max', single_patch.max())
                # single_patch = single_patch

                # single_patch = np.expand_dims(np.array(single_patch), axis=2)
                single_patch_input = np.expand_dims(single_patch, 0)
                single_patch_prediction = test_unet_model.predict(single_patch)
                single_patch_prediction[single_patch_prediction <= 0.6] = 0
                single_patch_prediction[single_patch_prediction > 0.6] = 1
                single_patch_predicted_img = single_patch_prediction
                predicted_patches.append(single_patch_predicted_img)

        predicted_patches = np.array(predicted_patches)
        predicted_patches_reshaped = np.reshape(
            predicted_patches, (patches.shape[0], patches.shape[1], 128, 128)
        )
        reconstructed_image = unpatchify(
            predicted_patches_reshaped, [patches.shape[0] * 128, patches.shape[1] * 128]
        )
        with rasterio.open(path_to_tiffs) as src:
            generate_gtif(
                reconstructed_image,
                src.profile,
                f"{input_planet_file_name_wo_extension}_v1.tif",
            )
        count += 1
        del predicted_patches
        with rasterio.open(f"{input_planet_file_name_wo_extension}_v1.tif") as src:
            image = src.read(1)

            # Convert the image to a binary format
            binary_image = image.astype(np.uint8)

            # Apply dilation to remove noise
            kernel = np.ones((3, 3), np.uint8)
            dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

            # Find contours in the dilated image
            contours, _ = cv2.findContours(
                dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Sort contours by area in descending order
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Select the top 6 contours with the highest area
            top_contours = contours[:6]

            # Create a blank image to draw the smoothed contours
            contour_image = np.zeros_like(image)

            # Draw and smooth the contours
            for contour in top_contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                # cv2.drawContours(contour_image, [smoothed_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                epsilon_dp = 0.1  # Adjust this value for desired simplification level
                approximated_contour_dp = cv2.approxPolyDP(
                    smoothed_contour, epsilon_dp, True
                )
                cv2.drawContours(
                    contour_image,
                    [smoothed_contour],
                    -1,
                    (255, 255, 255),
                    thickness=cv2.FILLED,
                )
            profile = src.profile
            profile.update(count=1, dtype="uint8")
        with rasterio.open(
            f"{input_planet_file_name_wo_extension}_mini.tif", "w", **profile
        ) as dst:
            dst.write(contour_image, 1)
        with rasterio.open(
            f"{input_planet_file_name_wo_extension}_mini.tif"
        ) as dataset:
            # Read the raster data
            image = dataset.read(1)

            # Extract metadata
            transform = dataset.transform
            crs = dataset.crs

            # Convert the raster to vector shapes
            results = (
                {"properties": {"raster_val": v}, "geometry": s}
                for i, (s, v) in enumerate(
                    shapes(image, mask=None, transform=transform)
                )
            )

            # Create a GeoDataFrame from the vector shapes
            gdf = gpd.GeoDataFrame.from_features(list(results))
        # Save the GeoDataFrame as GeoJSON
        gdf.crs = crs
        geojson_path = f"{input_planet_file_name_wo_extension}.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        s3_client.upload_file(
            geojson_path, dest_bucket, f"geojson_files/{geojson_path}"
        )

        push_status = push_to_stac(
            planet_item_id=planet_item_id,
            path_geojson=geojson_path,
            batch_to_ingest=200,
        )
        task_results.append(
            {
                "geojson_s3uri": f"s3://{dest_bucket}/geojson_files/{geojson_path}",
                "push_status": push_status,
            }
        )
        os.remove(f"{input_planet_file_name_wo_extension}_v1.tif")
        os.remove(f"{input_planet_file_name_wo_extension}_mini.tif")

    print(task_results)
