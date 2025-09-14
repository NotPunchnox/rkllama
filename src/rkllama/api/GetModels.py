import os

import config

MODEL_PATH = os.path.join(config.get_path("models"))


def GetModels():
    print("Retrieving models...")

    if not os.path.exists(MODEL_PATH):
        print("Models directory did not exist.\nCreating it now...")
        os.mkdir(MODEL_PATH)

    models_list = []

    for dest, flooders, files in os.walk(MODEL_PATH):
        for file in files:
            if file.endswith(".rkllm"):
                models_list.append(file)
    
    print("Number of valid models:", len(models_list), "\n")

    return models_list