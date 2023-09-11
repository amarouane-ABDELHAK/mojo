
from python.python import (
    Python,
    _destroy_python,
    _init_python,
)


def read_file_with_mojo(file_path: StringRef) -> PythonObject:
    print("Hello Mojo 🔥!")

    try:
        Python.add_to_path(".")
        Python.add_to_path("./sredd-inference")

        let test_module = Python.import_module("helpers")
        let req = test_module.read_file(file_path)
        return req
    except e:
        print(e.value)
        print("could not find module simple_interop")
    return
fn main() raises:

    let model_from_json = Python.import_module("tensorflow.keras.models")
    let rasterio = Python.import_module("rasterio")
    let loaded_model_json = read_file_with_mojo("./requirements.txt")
    let test_unet_model = model_from_json(loaded_model_json)
