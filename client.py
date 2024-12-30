import requests
import json

# Constantes pour l"URL de l"API et d"autres paramètres
API_URL = "http://127.0.0.1:8080/"  # Remplacer par l"URL de votre API
STREAM_MODE = True  # Passer à True si vous voulez activer le streaming
HISTORY = []  # Historique des messages pour maintenir la conversation

# Codes ANSI pour la couleur
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

def send_message(message):
    """
    Envoie un message à l"API RKLLM et récupère la réponse.
    """
    global HISTORY

    # Ajouter le message de l"utilisateur à l"historique
    HISTORY.append({"role": "user", "content": message})

    # Préparer la requête
    payload = {
        #"messages": json.dumps(HISTORY),
        "messages": message,
        "stream": STREAM_MODE
    }

    try:
        # Envoyer la requête à l"API
        if STREAM_MODE:
            with requests.post(API_URL + "rkllm_chat", json=payload, stream=True) as response:
                if response.status_code == 200:

                    assistant_message = f"{CYAN}{BOLD}Assistant: {RESET}"

                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                response_json = json.loads(line)
                                assistant_message += response_json["choices"][0]["content"]
                                print(assistant_message, end="\r", flush=True)
                            except json.JSONDecodeError:
                                print(f"{RED}Erreur lors de la détection de la réponse JSON.{RESET}")
                    print("\n")
                else:
                    print(f"{RED}Erreur lors du streaming: {response.status_code} - {response.text}{RESET}")
        else:
            response = requests.post(API_URL + "rkllm_chat", json=payload)
            if response.status_code == 200:
                response_json = response.json()
                assistant_message = response_json["choices"][0]["content"]
                print(f"{CYAN}{BOLD}Assistant:{RESET} {assistant_message}")
                HISTORY.append({"role": "assistant", "content": assistant_message})
            else:
                print(f"{RED}Erreur lors de la requête: {response.status_code} - {response.text}{RESET}")

    except requests.RequestException as e:
        print(f"{RED}Erreur de requête: {e}{RESET}")

def chat():
    """
    Fonction principale pour gérer le chat interactif.
    """
    print(f"{GREEN}{BOLD}Bienvenue dans le chat avec RKLLM! Tapez \"exit\" pour quitter.{RESET}")
    while True:
        # Demander un message à l"utilisateur
        user_input = input(f"{YELLOW}Vous:{RESET} ")

        if user_input.lower() == "exit":
            print(f"{RED}Fin de la conversation.{RESET}")
            break
        
        send_message(user_input)

if __name__ == "__main__":
    chat()
