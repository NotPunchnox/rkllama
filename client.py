import requests
import json
import sys
import os

# Constantes pour l'URL de l'API et autres paramètres
API_URL = "http://127.0.0.1:8080/"  # Remplacer par l'URL de votre API si vous l'avez changé
STREAM_MODE = True  # Passer à False si vous voulez désactiver le streaming
HISTORY = []  # Historique des messages pour maintenir la conversation

# Codes ANSI pour la couleur
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


# Affiche le menu d'aide avec toutes les commandes disponibles.
def print_help():
    print(f"{CYAN}{BOLD}Commandes disponibles:{RESET}")
    print(f"{YELLOW}help{RESET}             : Affiche ce menu d'aide.")
    print(f"{YELLOW}list{RESET}             : Liste tous les modèles disponibles sur le serveur.")
    print(f"{YELLOW}load model.rkllm{RESET} : Charge un modèle spécifique.")
    print(f"{YELLOW}unload_model{RESET}     : Décharge le modèle actuellement chargé.")
    print(f"{YELLOW}chat{RESET}             : Entrez en mode conversation avec le modèle.")
    print(f"{YELLOW}set_stream{RESET}       : Active le mode stream.")
    print(f"{YELLOW}unset_stream{RESET}     : Désactive le mode stream.")
    print(f"{YELLOW}exit{RESET}             : Quitte le programme.")


# Récupère la liste des modèles disponibles depuis le serveur.
def list_models():
    try:
        response = requests.get(API_URL + "models")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"{GREEN}{BOLD}Modèles disponibles:{RESET}")
            for model in models:
                print(f"- {model}")
        else:
            print(f"{RED}Erreur lors de la récupération des modèles: {response.status_code} - {response.text}{RESET}")
    except requests.RequestException as e:
        print(f"{RED}Erreur de requête: {e}{RESET}")


# Charge un modèle spécifique sur le serveur.
def load_model(model_name):
    payload = {"model_name": model_name}
    try:
        response = requests.post(API_URL + "load_model", json=payload)
        if response.status_code == 200:
            print(f"{GREEN}{BOLD}Modèle {model_name} chargé avec succès.{RESET}")
            return True
        else:
            print(f"{RED}Erreur lors du chargement du modèle: {response.status_code} - {response.json().get('error', response.text)}{RESET}")
        return False
    except requests.RequestException as e:
        print(f"{RED}Erreur de requête: {e}{RESET}")
        return False


# Décharge le modèle actuellement chargé.
def unload_model():
    try:
        response = requests.post(API_URL + "unload_model")
        if response.status_code == 200:
            print(f"{GREEN}{BOLD}Modèle déchargé avec succès.{RESET}")
        else:
            print(f"{RED}Erreur lors du déchargement du modèle: {response.status_code} - {response.json().get('error', response.text)}{RESET}")
    except requests.RequestException as e:
        print(f"{RED}Erreur de requête: {e}{RESET}")

# Envoie un message au modèle chargé et affiche la réponse.
def send_message(message):
    global HISTORY

    HISTORY.append({"role": "user", "content": message})
    payload = {
        "messages": message,
        "stream": STREAM_MODE
    }

    try:
        if STREAM_MODE:
            with requests.post(API_URL + "generate", json=payload, stream=True) as response:
                
                if response.status_code == 200:
                    print(f"{CYAN}{BOLD}Assistant:{RESET} ", end="")
                    assistant_message = ""

                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                response_json = json.loads(line)
                                content_chunk = response_json["choices"][0]["content"]
                                sys.stdout.write(content_chunk)
                                sys.stdout.flush()
                                assistant_message += content_chunk  # Sauvegarde complète.
                            except json.JSONDecodeError:
                                print(f"{RED}Erreur lors de la détection de la réponse JSON.{RESET}")


                    print("\n")  # Ajout d'une ligne à la fin du flux.
                else:
                    print(f"{RED}Erreur lors du streaming: {response.status_code} - {response.text}{RESET}")

        else:
            response = requests.post(API_URL + "generate", json=payload)
            if response.status_code == 200:
                response_json = response.json()
                assistant_message = response_json["choices"][0]["content"]

                print(f"{CYAN}{BOLD}Assistant:{RESET} {assistant_message}")
                HISTORY.append({"role": "assistant", "content": assistant_message})
            else:
                print(f"{RED}Erreur lors de la requête: {response.status_code} - {response.text}{RESET}")

    except requests.RequestException as e:
        print(f"{RED}Erreur de requête: {e}{RESET}")

# Fonction pour changer de modèle si l'ancien modèle chargé n'est pas le même à executer
def switch_model(new_model):
    response = requests.get(API_URL + "current_model")
    if response.status_code == 200:
        current_model = response.json().get("model_name")
        if current_model:
            print(f"{YELLOW}Déchargement du modèle actuel: {current_model}{RESET}")
            unload_model()

    if not load_model(new_model):
        print(f"{RED}Impossible de charger le modèle {new_model}.{RESET}")
        return False

    print(f"{GREEN}Modèle {new_model} prêt à l'emploi.{RESET}")
    return True


# Fonction interactive pour discuter avec le modèle.
def chat():
    print(f"{GREEN}{BOLD}Mode chat activé. Tapez 'exit' pour quitter.{RESET}")
    while True:
        user_input = input(f"{YELLOW}Vous:{RESET} ")

        if user_input == "clear" or user_input == "c":
            os.system("clear")
            break
        elif user_input.lower() == "exit":
            print(f"{RED}Fin de la conversation.{RESET}")
            break

        # Si le contenu n'est pas une commande, alors envoyer le contenu au modèle
        send_message(user_input)


def main():
    # Vérification du nombre d'entrée minimale
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]

    # Début de la suite de condition
    match command:
        
        case "help":
            print_help()

        case "list":
            list_models()

        case "load_model:":
            if len(sys.argv) < 3:
                print(f"{RED}Erreur: Vous devez spécifier le nom du modèle.{RESET}")
            else:
                load_model(sys.argv[2])

        case "unload_model":
            unload_model()
        
        case "run":
            if not switch_model(sys.argv[2]):
                return
            chat()

        case "set_stream":
            STREAM_MODE = True
        
        case "unset_stream":
            STREAM_MODE = False
        
        case _:
            print(f"{RED}Commande inconnue: {command}.{RESET}")
            print_help()


# Lancement de la fonction main: début du programme
if __name__ == "__main__":
    main()