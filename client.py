import requests
import json
import sys
import os

# Constants for API URL and other parameters
API_URL         = "http://127.0.0.1:8080/" # Replace with your API URL if you've changed it
STREAM_MODE     = True
VERBOSE         = False
HISTORY         = [] # Message history to keep the conversation going
PREFIX_MESSAGE  = "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
SUFIX_MESSAGE   = "<|im_end|><|im_start|>assistant"

# ANSI color codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"

# Displays the help menu with all available commands.
def print_help():
    print(f"{CYAN}{BOLD}Available commands:{RESET}")
    print(f"{YELLOW}help{RESET}                     : Displays this help menu.")
    print(f"{YELLOW}serve{RESET}                    : Starts the server.")
    print(f"{YELLOW}list{RESET}                     : Lists all available models on the server.")
    print(f"{YELLOW}load model.rkllm{RESET}         : Loads a specific model.")
    print(f"{YELLOW}unload{RESET}                   : Unloads the currently loaded model.")
    print(f"{YELLOW}run{RESET}                      : Enters conversation mode with the model.")
    print(f"{YELLOW}pull hf/model/file.rkllm{RESET} : Downloads a model via a file from Hugging Face.")
    print(f"{YELLOW}exit{RESET}                     : Exits the program.")

def print_help_chat():
    print(f"{CYAN}{BOLD}Available commands:{RESET}")
    print(f"{YELLOW}/help{RESET}           : Displays this help menu.")
    print(f"{YELLOW}/clear{RESET}          : Clears the current conversation history.")
    print(f"{YELLOW}/cls or /c{RESET}      : Clears the console content.")
    print(f"{YELLOW}/set stream{RESET}     : Enables stream mode.")
    print(f"{YELLOW}/unset stream{RESET}   : Disables stream mode.")
    print(f"{YELLOW}/set verbose{RESET}    : Enables verbose mode.")
    print(f"{YELLOW}/unset verbose{RESET}  : Disables verbose mode.")
    print(f"{YELLOW}/set system{RESET}     : Modifies the system message.")
    print(f"{YELLOW}exit{RESET}            : Exits the conversation.\n")


# Check status of rkllama API
def check_status():
    try:
        response = requests.get(API_URL)
        return response.status_code
    except:
        return 500

# Retrieves the list of available templates from the server.
def list_models():
    try:
        response = requests.get(API_URL + "models")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"{GREEN}{BOLD}Available models:{RESET}")
            for model in models:
                print(f"- {model}")
        else:
            print(f"{RED}Error retrieving models: {response.status_code} - {response.text}{RESET}")
    except requests.RequestException as e:
        print(f"{RED}Query error: {e}{RESET}")


# Loads a specific template on the server.
def load_model(model_name):
    payload = {"model_name": model_name}
    try:
        response = requests.post(API_URL + "load_model", json=payload)
        if response.status_code == 200:
            print(f"{GREEN}{BOLD}Model {model_name} loaded successfully.{RESET}")
            return True
        else:
            print(f"{RED}Error loading model: {response.status_code} - {response.json().get('error', response.text)}{RESET}")
        return False
    except requests.RequestException as e:
        print(f"{RED}Query error: {e}{RESET}")
        return False


# Unloads the currently loaded model.
def unload_model():
    try:
        response = requests.post(API_URL + "unload_model")
        if response.status_code == 200:
            print(f"{GREEN}{BOLD}Model successfully unloaded.{RESET}")
        else:
            print(f"{RED}Error when unloading model: {response.status_code} - {response.json().get('error', response.text)}{RESET}")
    except requests.RequestException as e:
        print(f"{RED}Query error: {e}{RESET}")


# Sends a message to the loaded model and displays the response.
def send_message(message):
    global HISTORY

    # Temporairement désactivé
    # HISTORY.append({"role": "user", "content": message})
    # historyParsed = "\n".join(
    #     [f"{'User' if entry['role'] == 'user' else 'Assistant'}: {entry['content']}" for entry in HISTORY]
    # )
    # print(historyParsed)

    payload = {
        "messages": PREFIX_MESSAGE + message + SUFIX_MESSAGE,
        "stream": STREAM_MODE
    }

    try:
        if STREAM_MODE:
            with requests.post(API_URL + "generate", json=payload, stream=True) as response:
                
                if response.status_code == 200:
                    print(f"{CYAN}{BOLD}Assistant:{RESET} ", end="")
                    assistant_message = ""
                    final_json        = {
                        "usage": {
                            "tokens_per_second": 0,
                            "completion_tokens": 0
                        }
                    }

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
                        tokens_per_second = final_json["usage"]["tokens_per_second"]
                        completion_tokens = final_json["usage"]["completion_tokens"]
                        print(f"\n\n{GREEN}Tokens par seconde{RESET}: {tokens_per_second}")
                        print(f"{GREEN}Nombre de tokens  {RESET}: {completion_tokens}")

                    HISTORY.append({"role": "assistant", "content": assistant_message})

                    # Return to line after last token
                    print("\n")

                else:
                    print(f"{RED}Streaming error: {response.status_code} - {response.text}{RESET}")

        else:
            response = requests.post(API_URL + "generate", json=payload)
            if response.status_code == 200:
                response_json = response.json()
                assistant_message = response_json["choices"][0]["content"]

                print(f"{CYAN}{BOLD}Assistant:{RESET} {assistant_message}")

                if VERBOSE == True:
                        tokens_per_second = final_json["usage"]["tokens_per_second"]
                        completion_tokens = final_json["usage"]["completion_tokens"]
                        print(f"\n\n{GREEN}Tokens per second{RESET}: {tokens_per_second}")
                        print(f"{GREEN}Number of Tokens  {RESET}: {completion_tokens}")
                        
                HISTORY.append({"role": "assistant", "content": assistant_message})
            else:
                print(f"{RED}Query error: {response.status_code} - {response.text}{RESET}")

    except requests.RequestException as e:
        print(f"{RED}Query error: {e}{RESET}")

# Function to change model if the old model loaded is not the same one to execute
def switch_model(new_model):
    response = requests.get(API_URL + "current_model")
    if response.status_code == 200:
        current_model = response.json().get("model_name")
        if current_model:
            print(f"{YELLOW}Downloading the current model: {current_model}{RESET}")
            unload_model()

    if not load_model(new_model):
        print(f"{RED}Unable to load model {new_model}.{RESET}")
        return False

    return True

# Function for remove model
def remove_model(model):
    response = requests.get(API_URL + "current_model")
    if response.status_code == 200:
        current_model = response.json().get("model_name")
        if current_model == model:
            print(f"{YELLOW}Unloading the current model before deletion: {current_model}{RESET}")
            unload_model()

    response_rm = requests.delete(API_URL + "remove", json={"model": model})

    if response_rm.status_code == 200:
        print(f"{GREEN}The model has been successfully deleted!{RESET}")


# Function for download model
def pull_model(model):

    if model is None or model == "":
        repo = input(f"{CYAN}Repo ID{RESET} ( example: punchnox/Tinnyllama-1.1B-rk3588-rkllm-1.1.4 ): ")
        filename = input(f"{CYAN}File{RESET} ( example: TinyLlama-1.1B-Chat-v1.0-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm ): ")

    model = repo + "/" + filename

    try:
        response = requests.post(API_URL + "pull", json={"model": model}, stream=True)

        if response.status_code != 200:
            print(f"{RED}Error: Received status code {response.status_code}.{RESET}")
            print(response.text)
            return

        def update_progress(progress):
            bar_length = 50  # Length of the progress bar
            block = int(round(bar_length * progress / 100))
            text = f"\r{GREEN}Progress:{RESET} [{CYAN}{'#' * block}{RESET}{'-' * (bar_length - block)}] {progress:.2f}%"
            sys.stdout.write(text)
            sys.stdout.flush()

        # Progress bar
        for line in response.iter_lines(decode_unicode=True):
            if line:
                line = line.strip()
                if line.endswith('%'):  # Checks if the line contains a percentage
                    try:
                        progress = int(line.strip('%'))
                        update_progress(progress)
                    except ValueError:
                        print(f"\n{line}")  # Displays non-numeric messages
                else:
                    print(f"\n{line}")  # Displays other messages

        print(f"\n{GREEN}Download complete.{RESET}")
    except requests.RequestException as e:
        print(f"Error connecting to server: {e}")


# Interactive function for chatting with the model.
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
            print(f"{GREEN}Conversation history successfully reset{RESET}")
        elif user_input == "/cls" or user_input == "/c":
            os.system("clear")
        elif user_input.lower() == "exit":
            print(f"{RED}End of conversation.{RESET}")
            break
        elif user_input == "/set stream":
            STREAM_MODE = True
            print(f"{GREEN}Stream mode successfully activated!{RESET}")
        elif user_input == "/unset stream":
            STREAM_MODE = False
            print(f"{RED}Stream mode successfully deactivated!{RESET}")
        elif user_input == "/set verbose":
            VERBOSE = True
            print(f"{GREEN}Verbose mode successfully activated!{RESET}")
        elif user_input == "/unset verbose":
            VERBOSE = False
            print(f"{RED}Verbose mode successfully deactivated!{RESET}")
        elif user_input == "/set system":
            system_prompt = input(f"{CYAN}System prompt: {RESET}")
            SYSTEM = f"<|im_start|>{system_prompt}<|im_end|> <|im_start|>user"
            print(f"{GREEN}System message successfully modified!")
        else:
            # If content is not a command, then send content to template
            send_message(user_input)


def main():
    # Check minimum number of entries

    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]

    if check_status() != 200 and command != "serve":
        print(f"{RED}Error: Server not started!\n{RESET}rkllama serve{CYAN} command to start the server.{RESET}")
        sys.exit(0)

    # Start of condition sequence
    if command == "help":
        print_help()

    elif command == "serve":
        os.system(f"bash ~/RKLLAMA/server.sh")

    elif command =="list":
        list_models()

    elif command == "load_model":
        if len(sys.argv) < 3:
            print(f"{RED}Error: You must specify the model name.{RESET}")
        else:
            load_model(sys.argv[2])
    
    elif command == "unload":
        unload_model()

    elif command == "run":
        if not switch_model(sys.argv[2]):
            return
        chat()
            
    elif command == "rm":
        if sys.argv[2] is None:
            print(f"{RED}Error: You must specify the model name.{RESET}")
        else:
            remove_model(sys.argv[2])
    
    elif command == "pull":
        pull_model(sys.argv[2] if len(sys.argv) < 2 else "" )
    
    else:
        print(f"{RED}Unknown command: {command}.{RESET}")
        print_help()


# Launching the main function: program start
if __name__ == "__main__":
    main()