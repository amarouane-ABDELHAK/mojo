import json
from pyproj import Transformer
import numbers


def get_bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [
        min(x_coordinates),
        min(y_coordinates),
        max(x_coordinates),
        max(y_coordinates),
    ]


def project_coords(coords, from_proj, to_proj):
    if len(coords) < 1:
        return []

    if isinstance(coords[0], numbers.Number):
        from_x, from_y = coords
        transformer = Transformer.from_crs(from_proj, to_proj)
        to_x, to_y = transformer.transform(from_x, from_y)
        return [to_x, to_y]

    new_coords = []
    for coord in coords:
        new_coords.append(project_coords(coord, from_proj, to_proj))
    return new_coords


def project_feature(feature, from_proj, to_proj):
    if "geometry" not in feature or "coordinates" not in feature["geometry"]:
        print("Failed project feature", feature)
        return None
    new_coordinates = project_coords(
        feature["geometry"]["coordinates"], from_proj, to_proj
    )
    feature["geometry"]["coordinates"] = new_coordinates
    feature["bbox"] = get_bounding_box(new_coordinates[0])
    return feature


def read_data(geom_file):
    with open(geom_file, encoding="utf-8") as data:
        data = json.load(data)
    return data


def project_geojson_file(in_file, from_proj="EPSG:32611", to_proj="EPSG:4326"):
    data = read_data(in_file)
    for feature in data["features"]:
        yield project_feature(feature, from_proj, to_proj)


if __name__ == "__main__":
    # Or any other projection you want
    geo_path = "/tmp/new_geojson.geojson"
    foo = project_geojson_file(geo_path)
