import requests
import json
import sys
import os
import configparser
import re
import subprocess

import rkllama.config

STREAM_MODE = True
VERBOSE = False
HISTORY = []
PREFIX_MESSAGE = "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
SUFFIX_MESSAGE = "<|im_end|><|im_start|>assistant"

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

PORT = rkllama.config.get("server", "port")
API_URL = f"http://127.0.0.1:{PORT}/"


# Displays the help menu with all available commands.
def print_help():
    print(f"{CYAN}{BOLD}Available commands:{RESET}")
    print(f"{YELLOW}help{RESET}                     : Displays this help menu.")
    print(f"{YELLOW}update{RESET}                   : Checks for available updates and upgrades.")
    print(f"{YELLOW}serve{RESET}                    : Starts the server.")
    print(f"{YELLOW}list{RESET}                     : Lists all available models on the server.")
    print(f"{YELLOW}info{RESET}                     : Show informations for a specific model.")
    print(f"{YELLOW}pull hf/model/file.rkllm{RESET} : Downloads a model via a file from Hugging Face.")
    print(f"{YELLOW}rm model.rkllm{RESET}           : Remove the model.")
    print(f"{YELLOW}load model.rkllm{RESET}         : Loads a specific model.")
    print(f"{YELLOW}unload model.rkllm{RESET}       : Unloads a specific model.")
    print(f"{YELLOW}ps{RESET}                       : List running models in the server.")
    print(f"{YELLOW}run{RESET}                      : Enters conversation mode with the model.")
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

# Retrieves the list of running models in the server.
def list_running_models():
    try:
        response = requests.get(API_URL + "api/ps")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if not models:
                print(f"{YELLOW}No model currently loaded.{RESET}")
                return

            _ansi_re = re.compile(r'\033\[[0-9;]*m')

            def visible_len(s):
                return len(_ansi_re.sub('', s))

            W = 55

            def row(label, value):
                prefix = f"  {label:<22} "
                pad = W - len(prefix) - visible_len(str(value))
                return f"{YELLOW}│{RESET}{prefix}{value}{' ' * max(pad, 0)}{YELLOW}│{RESET}"

            sep_top = f"{YELLOW}┌{'─' * W}┐{RESET}"
            sep_mid = f"{YELLOW}├{'─' * W}┤{RESET}"
            sep_bot = f"{YELLOW}└{'─' * W}┘{RESET}"

            print(f"\n{CYAN}{BOLD}Running models:{RESET}")

            for i, model in enumerate(models):
                details  = model.get("details", {})
                size_mb  = model.get("size", 0) / 1_073_741_824

                print(sep_top)
                print(row("Model",           f"{BOLD}{model.get('name', '?')}{RESET}"))
                print(row("Format",          f"{details.get('format', '?')}"))
                print(row("Family",          f"{details.get('family', '?')}"))
                print(row("Parameter size",  f"{details.get('parameter_size', '?')}"))
                print(row("Quantization",    f"{details.get('quantization_level', '?')}"))
                print(row("Size (GB)",       f"{GREEN}{size_mb:.2f} GB{RESET}"))
                print(sep_mid)
                print(row("Loaded at",       f"{CYAN}{model.get('loaded_at', '?')}{RESET}"))
                print(row("Last call",       f"{CYAN}{model.get('last_call', '?')}{RESET}"))
                print(row("Expiration",      f"{CYAN}{model.get('expires_at', '?')}{RESET}"))
                print(sep_bot)
                if i < len(models) - 1:
                    print()
        else:
            print(f"{RED}Error retrieving running models: {response.status_code} - {response.text}{RESET}")
    except requests.RequestException as e:
        print(f"{RED}Query error: {e}{RESET}")


# Loads a specific template on the server.
def load_model(model_name, From=None, huggingface_path=None):

    if From != None and huggingface_path != None:
        payload = {"model_name": model_name, "huggingface_path": huggingface_path, "from": From}
    else:
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
def unload_model(model_name):
    try:
        response = requests.post(API_URL + "unload_model", json={"model_name": model_name})
        if response.status_code == 200:
            print(f"{GREEN}{BOLD}Model successfully unloaded.{RESET}")
        else:
            print(f"{RED}Error when unloading model: {response.status_code} - {response.json().get('error', response.text)}{RESET}")
    except requests.RequestException as e:
        print(f"{RED}Query error: {e}{RESET}")


def _print_verbose(usage, model_name="?", finish_reason="?"):
    """Display a formatted verbose statistics block after a response."""
    W = 49  # inner visible width

    prompt_tokens     = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens      = usage.get("total_tokens", 0)
    tps               = usage.get("tokens_per_second", 0)
    prompt_dur        = usage.get("prompt_eval_duration", 0)
    eval_dur          = usage.get("eval_duration", 0)
    total_dur         = usage.get("total_duration", 0)
    load_dur          = usage.get("load_duration", 0)

    _ansi_re = re.compile(r'\033\[[0-9;]*m')

    def visible_len(s):
        return len(_ansi_re.sub('', s))

    def row(label, value):
        prefix = f"  {label:<26} "
        pad = W - len(prefix) - visible_len(str(value))
        return f"{YELLOW}│{RESET}{prefix}{value}{' ' * max(pad, 0)}{YELLOW}│{RESET}"

    sep_top = f"{YELLOW}┌{'─' * W}┐{RESET}"
    sep_mid = f"{YELLOW}├{'─' * W}┤{RESET}"
    sep_bot = f"{YELLOW}└{'─' * W}┘{RESET}"

    print(f"\n{sep_top}")
    print(row("Model", f"{BOLD}{model_name}{RESET}"))
    print(row("Finish reason", f"{BOLD}{finish_reason}{RESET}"))
    print(sep_mid)
    print(row("Prompt tokens",  f"{GREEN}{prompt_tokens}{RESET}"))
    print(row("Generated tokens", f"{GREEN}{completion_tokens}{RESET}  (total: {GREEN}{total_tokens}{RESET})"))
    print(row("Tokens/sec",   f"{GREEN}{tps}{RESET}"))
    print(sep_mid)
    print(row("Model load",     f"{CYAN}{load_dur:.3f}s{RESET}"))
    print(row("Prompt tokenization",   f"{CYAN}{prompt_dur:.3f}s{RESET}"))
    print(row("Response generation",    f"{CYAN}{eval_dur:.3f}s{RESET}"))
    print(row("Total duration",          f"{CYAN}{total_dur:.3f}s{RESET}"))
    print(sep_bot)


# Sends a message to the loaded model and displays the response.
def send_message(model, message):
    global HISTORY

    HISTORY.append({"role": "user", "content": message})

    # if VERBOSE == True:
    #     print(HISTORY)

    payload = {
        "model" : model,
        "messages": HISTORY,
        "stream": STREAM_MODE
    }


    try:
        if STREAM_MODE:
            with requests.post(API_URL + "v1/chat/completions", json=payload, stream=True) as response:
                
                if response.status_code == 200:
                    print(f"{CYAN}{BOLD}Assistant:{RESET} ", end="")
                    assistant_message = ""
                    finish_reason     = ""
                    final_json        = {
                        "usage": {}
                    }

                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                json_line_striped = line.removeprefix("data: ").strip()
                                if json_line_striped != "[DONE]":
                                    response_json = json.loads(json_line_striped) 
                                    
                                    final_json = response_json

                                    if len(response_json["choices"]) > 0 and "delta" in response_json["choices"][0].keys():
                                        content_chunk = response_json["choices"][0]["delta"]["content"]
                                        sys.stdout.write(content_chunk)
                                        sys.stdout.flush()
                                        assistant_message += content_chunk
                                        fr = response_json["choices"][0].get("finish_reason")
                                        if fr:
                                            finish_reason = fr
                            except json.JSONDecodeError:
                                print(f"{RED}Error detecting JSON response.{RESET}")

                    if VERBOSE == True:
                        usage = final_json.get("usage", {})
                        model_name = final_json.get("model", "?")
                        _print_verbose(usage, model_name, finish_reason)

                    HISTORY.append({"role": "assistant", "content": assistant_message})

                    # Return to line after last token
                    print("\n")

                else:
                    print(f"{RED}Streaming error: {response.status_code} - {response.text}{RESET}")

        else:
            response = requests.post(API_URL + "v1/chat/completions", json=payload)
            if response.status_code == 200:
                
                response_json = response.json()
                assistant_message = response_json["choices"][0]["message"]["content"]
                print(f"{CYAN}{BOLD}Assistant:{RESET} {assistant_message}")

                if VERBOSE == True:
                        usage = response_json.get("usage", {})
                        model_name = response_json.get("model", "?")
                        finish_reason = ""
                        if response_json.get("choices"):
                            finish_reason = response_json["choices"][0].get("finish_reason", "?")
                        _print_verbose(usage, model_name, finish_reason)
                        
                HISTORY.append({"role": "assistant", "content": assistant_message})
            else:
                print(f"{RED}Query error: {response.status_code} - {response.text}{RESET}")

    except requests.RequestException as e:
        print(f"{RED}Query error: {e}{RESET}")

# Function for remove model
def remove_model(model):
    response_rm = requests.delete(API_URL + "rm", json={"model": model})

    if response_rm.status_code == 200:
        print(f"{GREEN}The model has been successfully deleted!{RESET}")


# Function for download model
def pull_model(model):

    if model is None or model == "":
        repo = input(f"{CYAN}Repo ID{RESET} ( example: punchnox/Tinnyllama-1.1B-rk3588-rkllm-1.1.4 ): ")
        filename = input(f"{CYAN}File{RESET} ( example: TinyLlama-1.1B-Chat-v1.0-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm ): ")
        model_name = input(f"{CYAN}Custom Model Name{RESET} ( example: tinyllama-chat:1.1b ): ")
        # Construct the repo and filename of the model choosed
        model = repo.strip() + "/" + filename.strip()
    else:
        # Received format: punchnox/Tinnyllama-1.1B-rk3588-rkllm-1.1.4/TinyLlama-1.1B-Chat-v1.0-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm/tinyllama-chat:1.1b
        model, model_name = model.rsplit("/", 1)

    try:
        response = requests.post(API_URL + "pull", json={"model": model, "model_name": model_name}, stream=True)

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
def chat(model):
    global VERBOSE, STREAM_MODE, HISTORY, PREFIX_MESSAGE
    os.system("clear")
    print_help_chat()
    
    while True:
        user_input = input(f"{CYAN}You:{RESET} ")

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
            PREFIX_MESSAGE = f"<|im_start|>{system_prompt}<|im_end|> <|im_start|>user"
            print(f"{GREEN}System message successfully modified!")
        else:
            # If content is not a command, then send content to template
            send_message(model, user_input)

def update():
    README_URL   = "https://raw.githubusercontent.com/NotPunchnox/rkllama/refs/heads/main/README.md"
    INSTALL_URL  = "git+https://github.com/NotPunchnox/rkllama.git"

    # current installed version
    try:
        import importlib.metadata
        current_version = importlib.metadata.version("rkllama")
    except Exception:
        current_version = "unknown"

    # fetch README and extract remote version
    try:
        readme_response = requests.get(README_URL, timeout=10)
        readme_response.raise_for_status()
    except requests.RequestException as e:
        print(f"{RED}Unable to reach GitHub: {e}{RESET}")
        return

    match = re.search(r'###\s*\[Version:\s*([\d.]+)\]', readme_response.text)
    if not match:
        print(f"{RED}Could not find version in remote README.{RESET}")
        return

    remote_version = match.group(1)

    # version comparison
    def parse_version(v):
        try:
            return tuple(int(x) for x in str(v).split("."))
        except Exception:
            return (0,)

    status_text = ""
    if parse_version(remote_version) <= parse_version(current_version):
        status_text = f"{GREEN}Up to date{RESET}"
    else:
        status_text = f"{YELLOW}Update available{RESET}"

    # Pretty table display
    _ansi_re = re.compile(r'\033\[[0-9;]*m')

    def visible_len(s):
        return len(_ansi_re.sub('', str(s)))

    W = 76  # inner visible width

    def row(label, value):
        prefix = f"  {label:<18} "
        pad = W - len(prefix) - visible_len(value)
        return f"{YELLOW}│{RESET}{prefix}{value}{' ' * max(pad,0)}{YELLOW}│{RESET}"

    sep_top = f"{YELLOW}┌{'─' * W}┐{RESET}"
    sep_bot = f"{YELLOW}└{'─' * W}┘{RESET}"

    print(f"\n{CYAN}{BOLD}Update check:{RESET}")
    print(sep_top)
    print(row("Installed version", f"{BOLD}{current_version}{RESET}"))
    print(row("Latest version",    f"{BOLD}{remote_version}{RESET}"))
    print(row("Status",            f"{status_text}"))
    print(sep_bot)

    if parse_version(remote_version) <= parse_version(current_version):
        print(f"\n{GREEN}Already up to date.{RESET}")
        return

    print(f"\n{GREEN}A new version is available: {BOLD}{remote_version}{RESET}{GREEN} (current: {current_version}){RESET}")

    # confirmation
    confirm = input(f"{CYAN}Update now? [y/N]:{RESET} ").strip().lower()
    if confirm not in ("y", "yes"):
        print(f"{RED}Update cancelled.{RESET}")
        return

    # install
    print(f"\n{YELLOW}Installing latest version...{RESET}")
    install = subprocess.run([sys.executable, "-m", "pip", "install", INSTALL_URL])
    if install.returncode == 0:
        print(f"\n{GREEN}{BOLD}Update successful!{RESET} Restart rkllama_client to use version {remote_version}.")
    else:
        print(f"\n{RED}Installation failed. Check the output above.{RESET}")


def show_model_info(model_name):
    try:
        # Préparer les données pour la requête POST
        data = {"name": model_name}
        
        # Envoyer la requête POST à l'endpoint /api/show
        response = requests.post(API_URL + "api/show", json=data)
        
        if response.status_code == 200:
            model_info = response.json()
            
            # Display model information as an English table
            _ansi_re = re.compile(r'\033\[[0-9;]*m')

            def visible_len(s):
                return len(_ansi_re.sub('', str(s)))

            W = 80  # inner visible width

            def row(label, value):
                prefix = f"  {label:<20} "
                pad = W - len(prefix) - visible_len(value)
                return f"{YELLOW}│{RESET}{prefix}{value}{' ' * max(pad,0)}{YELLOW}│{RESET}"

            sep_top = f"{YELLOW}┌{'─' * W}┐{RESET}"
            sep_mid = f"{YELLOW}├{'─' * W}┤{RESET}"
            sep_bot = f"{YELLOW}└{'─' * W}┘{RESET}"

            name = model_info.get("name", "?")
            details = model_info.get("details", {})
            parameters = model_info.get("parameters", "?")
            quant = details.get("quantization_level", "?")
            family = details.get("family", "?")
            size_bytes = model_info.get("size", 0)
            size_gb = size_bytes / (1024 ** 3) if isinstance(size_bytes, (int, float)) else "?"
            modified_at = model_info.get("modified_at", "?")
            license_ = model_info.get("license", "?")
            system_prompt = model_info.get("system") or "None"
            template = model_info.get("template", "?")
            model_meta = model_info.get("model_info", {})

            print(f"\n{CYAN}{BOLD}Model Information: {name}{RESET}")
            print(sep_top)
            print(row("Name", f"{BOLD}{name}{RESET}"))
            print(row("Family", f"{family}"))
            print(row("Parameter size", f"{parameters}"))
            print(row("Quantization", f"{quant}"))
            print(row("Size (GB)", f"{GREEN}{size_gb:.2f} GB{RESET}" if isinstance(size_gb, float) else f"{size_gb}"))
            print(sep_mid)
            print(row("Modified at", f"{CYAN}{modified_at}{RESET}"))
            print(row("License", f"{license_}"))
            print(row("System prompt", f"{system_prompt}"))
            print(row("Template", f"{template}"))
            print(sep_mid)

            # Hugging Face info (if present)
            hf = model_info.get("huggingface")
            if hf:
                hf_repo = hf.get("repo_id", "?")
                hf_desc = hf.get("description", "") or ""
                hf_desc_short = (hf_desc[:80] + "...") if len(hf_desc) > 80 else hf_desc
                hf_tags = ", ".join(hf.get("tags", [])[:5]) or "?"
                hf_downloads = hf.get("downloads", "?")
                hf_likes = hf.get("likes", "?")

                print(row("HF repo", f"{hf_repo}"))
                print(row("HF desc", f"{hf_desc_short}"))
                print(row("HF tags", f"{hf_tags}"))
                print(row("HF downloads", f"{hf_downloads}"))
                print(row("HF likes", f"{hf_likes}"))
                print(sep_mid)

            # Advanced model info
            print(row("Advanced info", ""))
            for i, (k, v) in enumerate(model_meta.items()):
                label = f"  {k}"
                # For long values, convert to short string
                val_str = str(v)
                if len(val_str) > W - 30:
                    val_str = val_str[:W - 33] + "..."
                print(row(k, val_str))

            print(sep_bot)

        elif response.status_code == 400:
            print(f"{RED}Error: Missing model name{RESET}")
        elif response.status_code == 404:
            print(f"{RED}Error: Model '{model_name}' not found{RESET}")
        else:
            print(f"{RED}Error retrieving model info: {response.status_code} - {response.text}{RESET}")
    except requests.RequestException as e:
        print(f"{RED}Query error: {e}{RESET}")


def main():
    global PORT, API_URL

    use_no_conda = "--no-conda" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg != "--no-conda"]

    # Parse host and port from command line arguments
    host = "127.0.0.1"  # default host
    filtered_args = []
    
    for arg in sys.argv:
        if arg.startswith("--host="):
            host = arg.split("=")[1]
        elif arg.startswith("--port="):
            PORT = arg.split("=")[1]
        else:
            filtered_args.append(arg)
    
    # Update sys.argv with filtered arguments
    sys.argv = filtered_args

    # Update API_URL with the correct host and port
    API_URL = f"http://{host}:{PORT}/"

    # Check minimum number of entries
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]

    if check_status() != 200 and command not in ['serve', 'update']:
        print(f"{RED}Error: Server not started or not accessible at {API_URL}!\n{RESET}rkllama serve{CYAN} command to start the server.{RESET}")
        sys.exit(0)

    # Start of condition sequence
    if command == "help":
        print_help()

    elif command == "serve":
        if len(sys.argv) > 2:
            PORT = sys.argv[2]

        server_script = os.path.join(rkllama.config.get_path(), 'server.sh')
        os.system(f"bash {server_script} {'--no-conda' if use_no_conda else ''} --port={PORT}")

    elif command == "update":
        update()

    elif command == "list":
        list_models()

    elif command == "load":
        if len(sys.argv) < 3:
            print(f"{RED}Error: You must specify the model name.{RESET}")
        else:
            load_model(sys.argv[2])

    elif command == "unload":
        if len(sys.argv) < 3:
            print(f"{RED}Error: You must specify the model name.{RESET}")
        else:
            unload_model(sys.argv[2])
    elif command == "ps":
        list_running_models()

    elif command == "run":
        if len(sys.argv) < 3:
            print(f"{RED}Error: You must specify the model name to chat.{RESET}")
            return
        elif len(sys.argv) == 3:
            if load_model(sys.argv[2]):
                chat(sys.argv[2])
            
    elif command == "rm":
        if len(sys.argv) < 3:
            print(f"{RED}Error: You must specify the model name.{RESET}")
        else:
            remove_model(sys.argv[2])
    
    elif command == "pull":
        pull_model(sys.argv[2] if len(sys.argv) > 2 else "")
    
    elif command == "info":
        if len(sys.argv) < 3:
            print(f"{RED}Error: You must specify the model name.{RESET}")
        else:
            show_model_info(sys.argv[2])

    else:
        print(f"{RED}Unknown command: {command}.{RESET}")
        print_help()


# Launching the main function: program start
if __name__ == "__main__":
    main()
