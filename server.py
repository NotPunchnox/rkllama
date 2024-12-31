import sys
import os
import subprocess
import resource
import argparse
from flask import Flask, request, jsonify

from src.classes import *
from src.rkllm import *
from src.variables import * 
from src.process import Request

def print_color(message, color):
    # Fonction pour afficher des messages en couleur
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['reset'])}{message}{colors['reset']}")

current_model = None  # Variable globale pour stocker le modèle chargé
modele_rkllm = None  # Instance du modèle

def load_model(model_name):
    global modele_rkllm
    model_path = f"./models/{model_name}"
    if not os.path.exists(model_path):
        return None, f"Modèle {model_name} introuvable dans le dossier ./models."
    
    # Initialisation du modèle
    modele_rkllm = RKLLM(model_path)
    return modele_rkllm, None

def unload_model():
    global modele_rkllm
    if modele_rkllm:
        modele_rkllm.release()
        modele_rkllm = None

app = Flask(__name__)

# Routes:
# GET  /models
# POST /load_model
# POST /unload_model
# POST /generate

# Route pour voir les modèles
@app.route('/models', methods=['GET'])
def list_models():
    # Retourner la liste des modèles disponibles dans ./models
    models_dir = "./models/"
    if not os.path.exists(models_dir):
        return jsonify({"error": "Le dossier ./models est introuvable."}), 500

    print(os.listdir(models_dir))
    models = [f for f in os.listdir(models_dir) if str(f).endswith(".rkllm")]
    print(models)
    return jsonify({"models": models}), 200

# Route pour charger un modèle dans le NPU
@app.route('/load_model', methods=['POST'])
def load_model_route():
    global current_model, modele_rkllm

    # Vérifier si un modèle est actuellement chargé
    if modele_rkllm:
        return jsonify({"error": "Un modèle est déjà chargé. Veuillez d'abord le décharger."}), 400

    data = request.json
    if "model_name" not in data:
        return jsonify({"error": "Veuillez fournir le nom du modèle à charger."}), 400

    model_name = data["model_name"]
    modele_rkllm, error = load_model(model_name)
    if error:
        return jsonify({"error": error}), 400

    current_model = model_name
    return jsonify({"message": f"Modèle {model_name} chargé avec succès."}), 200

# Route pour décharger un modèle du NPU
@app.route('/unload_model', methods=['POST'])
def unload_model_route():
    global current_model, modele_rkllm

    if not modele_rkllm:
        return jsonify({"error": "Aucun modèle n'est actuellement chargé."}), 400

    unload_model()
    current_model = None
    return jsonify({"message": "Modèle déchargé avec succès."}), 200

# Route pour récupérer le modèle en cours
@app.route('/current_model', methods=['GET'])
def get_current_model():
    global current_model

    if current_model:
        return jsonify({"model_name": current_model}), 200
    else:
        return jsonify({"error": "Aucun modèle n'est actuellement chargé."}), 404

# Route pour faire une requête au modèle
@app.route('/generate', methods=['POST'])
def recevoir_message():
    global modele_rkllm

    if not modele_rkllm:
        return jsonify({"error": "Aucun modèle n'est actuellement chargé."}), 400

    verrou.acquire()
    return Request(modele_rkllm)

@app.route('/', methods=['GET'])
def default_route():
    return jsonify({"Welcome to RK-LLama !"}), 200

# Fonction de lancement
def main():
    # Définir les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Initialisation du serveur RKLLM avec des options configurables.")
    parser.add_argument('--target_platform', type=str, help="Plateforme cible : par exemple, rk3588/rk3576.")
    args = parser.parse_args()

    if not args.target_platform:
        print_color("Erreur argument manquant: --target_platform")
    else:
        if args.target_platform not in ["rk3588", "rk3576"]:
            print_color("Erreur : Plateforme cible invalide. Veuillez entrer rk3588 ou rk3576.", "red")
            sys.exit(1)
        print_color(f"Fixation de la fréquence pour la plateforme {args.target_platform}...", "cyan")
        commande = f"sudo bash ./lib/fix_freq_{args.target_platform}.sh"
        subprocess.run(commande, shell=True)

    # Définir une limite de ressources
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    print_color("Démarrage de l'application Flask sur http://0.0.0.0:8080", "blue")
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)


# Lancer le programme
if __name__ == "__main__":
    main()