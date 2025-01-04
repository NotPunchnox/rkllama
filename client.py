import requests
import json
import sys
import os

# Constantes pour l'URL de l'API et autres paramètres
API_URL     = "http://127.0.0.1:8080/"  # Remplacer par l'URL de votre API si vous l'avez changé
STREAM_MODE = True
VERBOSE     = False
HISTORY     = []  # Historique des messages pour maintenir la conversation

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
    print(f"{YELLOW}serve{RESET}            : Lance le serveur ( doit être lancé avec sudo ).")
    print(f"{YELLOW}list{RESET}             : Liste tous les modèles disponibles sur le serveur.")
    print(f"{YELLOW}load model.rkllm{RESET} : Charge un modèle spécifique.")
    print(f"{YELLOW}unload{RESET}     : Décharge le modèle actuellement chargé.")
    print(f"{YELLOW}run{RESET}              : Entrez en mode conversation avec le modèle.")
    print(f"{YELLOW}exit{RESET}             : Quitte le programme.")

def print_help_chat():
    print(f"{CYAN}{BOLD}Commandes disponibles:{RESET}")
    print(f"{YELLOW}/help{RESET}           : Affiche ce menu d'aide.")
    print(f"{YELLOW}/clear{RESET}          : Supprime l'historique de conversation actuel.")
    print(f"{YELLOW}/cls ou /c{RESET}      : Supprime le contenu de la console.")
    print(f"{YELLOW}/set stream{RESET}     : Active le mode stream.")
    print(f"{YELLOW}/unset stream{RESET}   : Désactive le mode stream.")
    print(f"{YELLOW}/set verbose{RESET}    : Active le mode verbose.")
    print(f"{YELLOW}/unset verbose{RESET}  : Désactive le mode verbose.")
    print(f"{YELLOW}exit{RESET}            : Quitte la conversation.\n")

def check_status():
    try:
        response = requests.get(API_URL)
        return response.status_code
    except:
        return 500

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
                    final_json        = ""

                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                response_json = json.loads(line)
                                final_json = response_json

                                content_chunk = response_json["choices"][0]["content"]
                                sys.stdout.write(content_chunk)
                                sys.stdout.flush()
                                assistant_message += content_chunk
                            except json.JSONDecodeError:
                                print(f"{RED}Erreur lors de la détection de la réponse JSON.{RESET}")

                    if VERBOSE == True:
                        print(f"\n\n{GREEN}Tokens par seconde{RESET}: {final_json["usage"]["tokens_per_second"]}")
                        print(f"{GREEN}Nombre de tokens  {RESET}: {final_json["usage"]["completion_tokens"]}")

                    HISTORY.append({"role": "assistant", "content": assistant_message})

                    # Faire revenir à la ligne après le dernier token
                    print("\n")

                else:
                    print(f"{RED}Erreur lors du streaming: {response.status_code} - {response.text}{RESET}")

        else:
            response = requests.post(API_URL + "generate", json=payload)
            if response.status_code == 200:
                response_json = response.json()
                assistant_message = response_json["choices"][0]["content"]

                print(f"{CYAN}{BOLD}Assistant:{RESET} {assistant_message}")

                if VERBOSE == True:
                        print(f"\n\n{GREEN}Tokens par seconde{RESET}: {response_json["usage"]["tokens_per_second"]}")
                        print(f"{GREEN}Nombre de tokens  {RESET}: {response_json["usage"]["completion_tokens"]}")

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

    return True


# Fonction interactive pour discuter avec le modèle.
def chat():
    global VERBOSE, STREAM_MODE, HISTORY
    os.system("clear")
    print_help_chat()
    
    while True:
        user_input = input(f"{CYAN}Vous:{RESET} ")

        if user_input == "/help":
            print_help_chat()
        elif user_input == "/clear":
            HISTORY = []
            print(f"{GREEN}Historique de conversation réinitialisé avec succès{RESET}")
        elif user_input == "/cls" or user_input == "/c":
            os.system("clear")
        elif user_input.lower() == "exit":
            print(f"{RED}Fin de la conversation.{RESET}")
            break
        elif user_input == "/set stream":
            STREAM_MODE = True
            print(f"{GREEN}Mode stream activé avec succès!{RESET}")
        elif user_input == "/unset stream":
            STREAM_MODE = False
            print(f"{RED}Mode stream désactivé avec succès!{RESET}")
        elif user_input == "/set verbose":
            VERBOSE = True
            print(f"{GREEN}Mode verbose activé avec succès!{RESET}")
        elif user_input == "/unset verbose":
            VERBOSE = False
            print(f"{RED}Mode verbose désactivé avec succès!{RESET}")
        else:
            # Si le contenu n'est pas une commande, alors envoyer le contenu au modèle
            send_message(user_input)


def main():
    # Vérification du nombre d'entrée minimale

    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]

    if check_status() != 200 and command != "serve":
        print(f"{RED}Erreur: Le serveur n'est pas lancé!\n{CYAN}Command pour lancer le serveur: {RESET}rkllama serve")
        sys.exit(0)

    # Début de la suite de condition
    match command:
        
        case "help":
            print_help()

        case "serve":
            os.system(f"bash ~/RKLLAMA/server.sh")

        case "list":
            list_models()

        case "load_model:":
            if len(sys.argv) < 3:
                print(f"{RED}Erreur: Vous devez spécifier le nom du modèle.{RESET}")
            else:
                load_model(sys.argv[2])

        case "unload":
            unload_model()
        
        case "run":
            if not switch_model(sys.argv[2]):
                return
            chat()
        
        case _:
            print(f"{RED}Commande inconnue: {command}.{RESET}")
            print_help()


# Lancement de la fonction main: début du programme
if __name__ == "__main__":
    main()