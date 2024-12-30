import sys
import os
import subprocess
import resource
import time
import argparse
import json
from flask import Flask, request, jsonify, Response

from src.classes import *
from src.rkllm import *
from src.variables import * 
from src.process import Request

if __name__ == "__main__":

    # Créer une application Flask
    app = Flask(__name__)

    # Définir les arguments de ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, help="Chemin absolu du modèle RKLLM converti sur la carte Linux ;")
    parser.add_argument('--target_platform', type=str, required=True, help="Plateforme cible : par exemple, rk3588/rk3576 ;")
    parser.add_argument('--lora_path', type=str, help="Chemin absolu du modèle LORA sur la carte Linux ;")
    parser.add_argument('--path_prompt_cache', type=str, help="Chemin absolu du fichier cache des prompts sur la carte Linux ;")
    args = parser.parse_args()

    # Vérifier si le chemin du modèle RKLLM existe
    if not os.path.exists(args.rkllm_model_path):
        print("Erreur : Veuillez fournir le chemin correct pour le modèle RKLLM, et vous assurer qu'il s'agit d'un chemin absolu sur la carte.")
        sys.stdout.flush()
        exit()

    # Vérifier si la plateforme cible est correcte
    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("Erreur : Veuillez spécifier la bonne plateforme cible : rk3588/rk3576.")
        sys.stdout.flush()
        exit()

    # Vérifier si le chemin du modèle LORA est valide
    if args.lora_path:
        if not os.path.exists(args.lora_path):
            print("Erreur : Veuillez fournir le chemin correct pour le modèle LORA, et vous assurer qu'il s'agit d'un chemin absolu sur la carte.")
            sys.stdout.flush()
            exit()

    # Vérifier si le chemin du cache de prompts est valide
    if args.path_prompt_cache:
        if not os.path.exists(args.path_prompt_cache):
            print("Erreur : Veuillez fournir le chemin correct pour le fichier cache des prompts, et vous assurer qu'il s'agit d'un chemin absolu sur la carte.")
            sys.stdout.flush()
            exit()

    # Fixer la fréquence
    commande = f"sudo bash fix_freq_{args.target_platform}.sh"
    subprocess.run(commande, shell=True)

    # Définir une limite de ressources
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Initialiser le modèle RKLLM
    print("========= Initialisation... =========")
    sys.stdout.flush()
    model_path = args.rkllm_model_path
    modele_rkllm = RKLLM(model_path, args.lora_path, args.path_prompt_cache)
    print("Modèle RKLLM initialisé avec succès !")
    print("====================================")
    sys.stdout.flush()

    # Créer une fonction pour recevoir les données envoyées par l'utilisateur
    @app.route('/rkllm_chat', methods=['POST'])
    def recevoir_message():

        # Si le serveur est bloqué, retourner une réponse spécifique
        if isLocked or global_status == 0:
            return jsonify({'status': 'error', 'message': 'Le serveur RKLLM est occupé ! Vous pouvez réessayer plus tard.'}), 503

        verrou.acquire()

        return Request(modele_rkllm)

    # Démarrer l'application Flask
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)

    print("====================")
    print("Inférence du modèle RKLLM terminée, libération des ressources du modèle...")
    modele_rkllm.release()
    print("====================")
