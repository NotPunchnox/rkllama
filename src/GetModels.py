import os

MODEL_PATH = "~/RKLLAMA/models"


def GetModels():
    print("Récupération des modèles...")

    if not os.path.exists(MODEL_PATH):
        print("Le dossier models n'existait pas.\nCréation en cours...")
        os.mkdir(MODEL_PATH)

    models_list = []

    for dest, flooders, files in os.walk(MODEL_PATH):
        for file in files:
            if file.endswith(".rkllm"):
                models_list.append(file)
    
    print("Nombre de modèles valides:", len(models_list), "\n")

    return models_list