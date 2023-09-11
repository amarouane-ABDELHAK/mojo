def read_file(file_path):
    """
    """
    with open(file_path, "r") as json_file:
        loaded_model_json = json_file.read()
    return loaded_model_json



if __name__=="__main__":
    c = read_file("./requirements.txt")
    print(c)