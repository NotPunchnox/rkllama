# Import libs
import sys, os, subprocess, resource, argparse, shutil, time, requests, configparser, json, threading, datetime, logging
from dotenv import load_dotenv
from huggingface_hub import hf_hub_url, HfFileSystem
from flask import Flask, request, jsonify, Response, stream_with_context
from transformers import AutoTokenizer

# Local file
from src.classes import *
from src.rkllm import *
from src.process import Request
import src.variables as variables
from src.server_utils import process_ollama_chat_request
from src.debug_utils import StreamDebugger, check_response_format

# Check for debug mode
DEBUG_MODE = os.environ.get("RKLLAMA_DEBUG", "0").lower() in ["1", "true", "yes", "on"]

# Set up logging with appropriate level based on debug mode
logging_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser("~/RKLLAMA/rkllama_server.log"))
    ]
)
logger = logging.getLogger("rkllama.server")

def print_color(message, color):
    # Function for displaying color messages
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

CONFIG_FILE = os.path.expanduser("~/RKLLAMA/rkllama.ini")
current_model = None  # Global variable for storing the loaded model
modele_rkllm = None  # Model instance


def create_modelfile(huggingface_path, From, system="", temperature=1.0):
    struct_modelfile = f"""
FROM="{From}"

HUGGINGFACE_PATH="{huggingface_path}"

SYSTEM="{system}"

TEMPERATURE={temperature}
"""

    # Expand the path to the full directory path
    path = os.path.expanduser(f"~/RKLLAMA/models/{From.replace('.rkllm', '')}")

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Create the Modelfile and write the content
    with open(os.path.join(path, "Modelfile"), "w") as f:
        f.write(struct_modelfile)


def load_model(model_name, huggingface_path=None, system="", temperature=1.0, From=None):

    model_dir = os.path.expanduser(f"~/RKLLAMA/models/{model_name}")
    
    if not os.path.exists(model_dir):
        return None, f"Model directory '{model_name}' not found."
    
    if not os.path.exists(os.path.join(model_dir, "Modelfile")) and (huggingface_path is None and From is None):
        return None, f"Modelfile not found in '{model_name}' directory."
    elif huggingface_path is not None and From is not None:
        create_modelfile(huggingface_path=huggingface_path, From=From, system=system, temperature=temperature)
        time.sleep(0.1)
    
    # Load modelfile
    load_dotenv(os.path.join(model_dir, "Modelfile"), override=True)
    
    from_value = os.getenv("FROM")
    huggingface_path = os.getenv("HUGGINGFACE_PATH")

    # View config Vars
    print_color(f"FROM: {from_value}\nHuggingFace Path: {huggingface_path}", "green")
    
    if not from_value or not huggingface_path:
        return None, "FROM or HUGGINGFACE_PATH not defined in Modelfile."

    # Change value of model_id with huggingface_path
    variables.model_id = huggingface_path

    
    modele_rkllm = RKLLM(os.path.join(model_dir, from_value))
    return modele_rkllm, None

def unload_model():
    global modele_rkllm
    if modele_rkllm:
        modele_rkllm.release()
        modele_rkllm = None

app = Flask(__name__)

# Original RKLLAMA Routes:
# GET    /models
# POST   /load_model
# POST   /unload_model
# POST   /generate
# POST   /pull
# DELETE /rm

# Route to view models
@app.route('/models', methods=['GET'])
def list_models():
    # Return the list of available models in ~/RKLLAMA/models
    models_dir = os.path.expanduser("~/RKLLAMA/models")
    
    if not os.path.exists(models_dir):
        return jsonify({"error": "Le dossier ~/RKLLAMA/models est introuvable."}), 500

    direct_models = [f for f in os.listdir(models_dir) if f.endswith(".rkllm")]

    for model in direct_models:
        model_name = os.path.splitext(model)[0]
        model_dir = os.path.join(models_dir, model_name)
        
        os.makedirs(model_dir, exist_ok=True)
        
        shutil.move(os.path.join(models_dir, model), os.path.join(model_dir, model))
    
    model_dirs = []
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    model_dirs.append(subdir)
                    break

    return jsonify({"models": model_dirs}), 200


# Delete a model
@app.route('/rm', methods=['DELETE'])
def Rm_model():
    data = response.json
    if "model" not in data:
        return jsonify({"error": "Please specify a model."}), 400

    model_path = os.path.expanduser(f"~/RKLLAMA/models/{model}")
    if not os.path.exists(model_path):
        return jsonify({"error": f"The model: {model} cannot be found."}), 404

    os.remove(model_path)

    return jsonify({"message": f"The model has been successfully deleted!"}), 200

# route to pull a model
@app.route('/pull', methods=['POST'])
def pull_model():
    @stream_with_context
    def generate_progress():
        data = request.json
        if "model" not in data:
            yield "Error: Model not specified.\n"
            return

        splitted = data["model"].split('/')
        if len(splitted) < 3:
            yield f"Error: Invalid path '{data['model']}'\n"
            return

        file = splitted[2]
        repo = data["model"].replace(f"/{file}", "")

        try:
            # Use Hugging Face HfFileSystem to get the file metadata
            fs = HfFileSystem()
            file_info = fs.info(repo + "/" + file)

            total_size = file_info["size"]  # File size in bytes
            if total_size == 0:
                yield "Error: Unable to retrieve file size.\n"
                return

            # Créer un dossier pour le model
            os.makedirs(os.path.expanduser(f"~/RKLLAMA/models/{file.replace('.rkllm', '')}"))

            # Définir le fichier à télécharger
            local_filename = os.path.join(os.path.expanduser(f"~/RKLLAMA/models/{file.replace('.rkllm', '')}"), file)
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)

            # Créer le fichier de configuration du model
            create_modelfile(huggingface_path=repo, From=file)

            yield f"Downloading {file} ({total_size / (1024**2):.2f} MB)...\n"

            try:
                # Download the file with progress
                url = hf_hub_url(repo_id=repo, filename=file)
                with requests.get(url, stream=True) as r, open(local_filename, "wb") as f:
                    downloaded_size = 0
                    chunk_size = 8192  # 8KB

                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            progress = int((downloaded_size / total_size) * 100)
                            yield f"{progress}%\n"

            except Exception as download_error:
                # Remove the file if an error occurs during download
                if os.path.exists(local_filename):
                    os.remove(local_filename)
                yield f"Error during download: {str(download_error)}\n"
                return

        except Exception as e:
            yield f"Error: {str(e)}\n"

    # Use the appropriate content type for streaming responses
    is_ollama_request = request.path.startswith('/api/')
    content_type = 'application/x-ndjson' if is_ollama_request else 'text/plain'
    return Response(generate_progress(), content_type=content_type)

# Route for loading a model into the NPU
@app.route('/load_model', methods=['POST'])
def load_model_route():
    global current_model, modele_rkllm

    # Check if a model is currently loaded
    if modele_rkllm:
        return jsonify({"error": "A model is already loaded. Please unload it first."}), 400

    data = request.json
    if "model_name" not in data:
        return jsonify({"error": "Please enter the name of the model to be loaded."}), 400

    model_name = data["model_name"]

    #print(data)

    # Check if other params like "from" or "huggingface_path" for create modelfile
    if "from" in data or "huggingface_path" in data:
        modele_rkllm, error = load_model(model_name, From=data["from"], huggingface_path=data["huggingface_path"])
    else:
        modele_rkllm, error = load_model(model_name)

    if error:
        return jsonify({"error": error}), 400

    current_model = model_name
    return jsonify({"message": f"Model {model_name} loaded successfully."}), 200

# Route to unload a model from the NPU
@app.route('/unload_model', methods=['POST'])
def unload_model_route():
    global current_model, modele_rkllm

    if not modele_rkllm:
        return jsonify({"error": "No models are currently loaded."}), 400

    unload_model()
    current_model = None
    return jsonify({"message": "Model successfully unloaded!"}), 200

# Route to retrieve the current model
@app.route('/current_model', methods=['GET'])
def get_current_model():
    global current_model

    if current_model:
        return jsonify({"model_name": current_model}), 200
    else:
        return jsonify({"error": "No models are currently loaded."}), 404

# Route to make a request to the model
@app.route('/generate', methods=['POST'])
def recevoir_message():
    global modele_rkllm

    if not modele_rkllm:
        return jsonify({"error": "No models are currently loaded."}), 400

    variables.verrou.acquire()
    return Request(modele_rkllm)

# Ollama API compatibility routes

@app.route('/api/tags', methods=['GET'])
def list_ollama_models():
    # Return models in Ollama API format
    models_dir = os.path.expanduser("~/RKLLAMA/models")
    
    if not os.path.exists(models_dir):
        return jsonify({"models": []}), 200

    models = []
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    size = os.path.getsize(os.path.join(subdir_path, file))
                    models.append({
                        "name": subdir,
                        "model": subdir,
                        "modified_at": datetime.datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(subdir_path, file))
                        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "size": size,
                    })
                    break

    return jsonify({"models": models}), 200

@app.route('/api/show', methods=['POST'])
def show_model_info():
    data = request.json
    model_name = data.get('name')
    
    if not model_name:
        return jsonify({"error": "Missing model name"}), 400
        
    model_dir = os.path.expanduser(f"~/RKLLAMA/models/{model_name}")
    
    if not os.path.exists(model_dir):
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    # Read modelfile content if available
    modelfile_path = os.path.join(model_dir, "Modelfile")
    modelfile_content = ""
    if os.path.exists(modelfile_path):
        with open(modelfile_path, "r") as f:
            modelfile_content = f.read()
    
    # Find the .rkllm file
    model_file = None
    for file in os.listdir(model_dir):
        if file.endswith(".rkllm"):
            model_file = file
            break
    
    if not model_file:
        return jsonify({"error": f"Model file not found in '{model_name}' directory"}), 404
    
    file_path = os.path.join(model_dir, model_file)
    size = os.path.getsize(file_path)
    
    return jsonify({
        "license": "Unknown",
        "modelfile": modelfile_content,
        "parameters": "Unknown",
        "template": "{{ .Prompt }}",
        "name": model_name,
        "details": {
            "parent_model": "",
            "format": "rkllm",
            "family": "llama",
            "parameter_size": "Unknown",
            "quantization_level": "Unknown"
        },
        "size": size
    }), 200

@app.route('/api/create', methods=['POST'])
def create_model():
    data = request.json
    model_name = data.get('name')
    modelfile = data.get('modelfile', '')
    
    if not model_name:
        return jsonify({"error": "Missing model name"}), 400
    
    model_dir = os.path.expanduser(f"~/RKLLAMA/models/{model_name}")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "Modelfile"), "w") as f:
        f.write(modelfile)
    
    # Parse the modelfile to extract parameters
    modelfile_lines = modelfile.strip().split('\n')
    from_line = next((line for line in modelfile_lines if line.startswith('FROM=')), None)
    huggingface_path = next((line for line in modelfile_lines if line.startswith('HUGGINGFACE_PATH=')), None)
    
    if not from_line or not huggingface_path:
        return jsonify({"error": "Invalid Modelfile: missing FROM or HUGGINGFACE_PATH"}), 400
    
    # Extract values
    from_value = from_line.split('=')[1].strip('"\'')
    huggingface_path = huggingface_path.split('=')[1].strip('"\'')
    
    # For compatibility with existing implementation
    return jsonify({"status": "success", "model": model_name}), 200

@app.route('/api/pull', methods=['POST'])
def pull_model_ollama():
    data = request.json
    model = data.get('name')
    
    if not model:
        return jsonify({"error": "Missing model name"}), 400

    # Ollama API uses application/x-ndjson for streaming
    response_stream = pull_model()  # Call the existing function directly
    response_stream.content_type = 'application/x-ndjson'
    return response_stream

@app.route('/api/delete', methods=['DELETE'])
def delete_model_ollama():
    data = request.json
    model_name = data.get('name')
    
    if not model_name:
        return jsonify({"error": "Missing model name"}), 400

    model_path = os.path.expanduser(f"~/RKLLAMA/models/{model_name}")
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    # Check if model is currently loaded
    if current_model == model_name:
        unload_model()
    
    try:
        shutil.rmtree(model_path)
        return jsonify({}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete model: {str(e)}"}), 500

@app.route('/api/generate', methods=['POST'])
def generate_ollama():
    global modele_rkllm, current_model

    data = request.json
    model_name = data.get('model')
    prompt = data.get('prompt')
    system = data.get('system', '')
    stream = data.get('stream', True)
    
    if not model_name:
        return jsonify({"error": "Missing model name"}), 400

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    if DEBUG_MODE:
        logger.debug(f"API generate request: model={model_name}, stream={stream}, prompt_length={len(prompt)}")

    # Load model if needed
    if current_model != model_name:
        if current_model:
            unload_model()
        modele_instance, error = load_model(model_name)
        if error:
            return jsonify({"error": f"Failed to load model '{model_name}': {error}"}), 500
        global modele_rkllm
        modele_rkllm = modele_instance
        current_model = model_name

    # Set system prompt if provided
    variables.system = system
    
    # Use the process_ollama_generate_request function for proper generate format
    variables.verrou.acquire()
    
    try:
        from src.server_utils import process_ollama_generate_request
        return process_ollama_generate_request(
            modele_rkllm,
            model_name,
            prompt,
            system,
            stream
        )
    except Exception as e:
        if DEBUG_MODE:
            logger.exception(f"Error in generate_ollama: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        variables.verrou.release()

@app.route('/api/chat', methods=['POST'])
def chat_ollama():
    global modele_rkllm, current_model

    data = request.json
    model_name = data.get('model')
    messages = data.get('messages', [])
    system = data.get('system', '')
    stream = data.get('stream', True)
    
    if DEBUG_MODE:
        logger.debug(f"API chat request: model={model_name}, stream={stream}, messages_count={len(messages)}")
    
    if not model_name:
        if DEBUG_MODE:
            logger.warning("Missing model name in request")
        return jsonify({"error": "Missing model name"}), 400

    if not messages:
        if DEBUG_MODE:
            logger.warning("Missing messages in request")
        return jsonify({"error": "Missing messages"}), 400

    # Load model if needed
    if current_model != model_name:
        if current_model:
            if DEBUG_MODE:
                logger.debug(f"Unloading current model: {current_model}")
            unload_model()
        
        if DEBUG_MODE:
            logger.debug(f"Loading model: {model_name}")
        modele_instance, error = load_model(model_name)
        if error:
            if DEBUG_MODE:
                logger.error(f"Failed to load model {model_name}: {error}")
            return jsonify({"error": f"Failed to load model '{model_name}': {error}"}), 500
        global modele_rkllm
        modele_rkllm = modele_instance
        current_model = model_name
        if DEBUG_MODE:
            logger.debug(f"Model {model_name} loaded successfully")

    # Acquire lock before processing
    if DEBUG_MODE:
        logger.debug("Acquiring lock")
    variables.verrou.acquire()
    
    try:
        # Process Ollama chat request using the utility function
        if DEBUG_MODE:
            logger.debug("Processing request with process_ollama_chat_request")
        return process_ollama_chat_request(
            modele_rkllm,
            model_name,
            messages,
            system,
            stream
        )
    except Exception as e:
        if DEBUG_MODE:
            logger.exception(f"Error in chat_ollama: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if DEBUG_MODE:
            logger.debug("Releasing lock")
        variables.verrou.release()

# Only include debug endpoint if in debug mode
if DEBUG_MODE:
    @app.route('/api/debug', methods=['POST'])
    def debug_streaming():
        """Endpoint to diagnose streaming issues"""
        data = request.json
        stream_data = data.get('stream_data', '')
        
        issues = check_response_format(stream_data)
        
        if issues:
            return jsonify({
                "status": "error",
                "issues": issues,
                "recommendation": "Check server_utils.py implementation of streaming"
            }), 200
        else:
            return jsonify({
                "status": "ok",
                "message": "No issues found in the response format"
            }), 200

@app.route('/api/embeddings', methods=['POST'])
def embeddings_ollama():
    # This is a placeholder as embeddings aren't implemented in RKLLAMA
    return jsonify({
        "error": "Embeddings not supported in RKLLAMA"
    }), 501

# Default route
@app.route('/', methods=['GET'])
def default_route():
    return jsonify({
        "message": "Welcome to RKLLama with Ollama API compatibility!",
        "github": "https://github.com/notpunhnox/rkllama"
    }), 200

# Launch function
def main():
    # Define the arguments for the launch function
    parser = argparse.ArgumentParser(description="RKLLM server initialization with configurable options.")
    parser.add_argument('--target_platform', type=str, help="Target platform: rk3588/rk3576.")
    parser.add_argument('--port', type=str, default="8080", help="Default port: 8080") # Added default value
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    # Set debug mode if specified
    if args.debug:
        os.environ["RKLLAMA_DEBUG"] = "1"
        global DEBUG_MODE
        DEBUG_MODE = True
        logger.setLevel(logging.DEBUG)
        print_color("Debug mode enabled", "yellow")

    # Check if the configuration file exists
    if not os.path.exists(CONFIG_FILE):
        print("Configuration file not found. Creating with default values...")
        config = configparser.ConfigParser()
        config["server"] = {"port": "8080"}
        with open(CONFIG_FILE, "w") as configfile:
            config.write(configfile)

    # Load the configuration
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    # If a port is specified by the user, update it in the config
    if args.port:
        port = args.port
        config["server"]["port"] = port
        with open(CONFIG_FILE, "w") as configfile:
            config.write(configfile)
    else:
        # Otherwise, use the default port from the config
        port = config["server"].get("port", "8080")

    # Check the target platform
    if not args.target_platform:
        print_color("Error argument not found: --target_platform", "red")
    else:
        if args.target_platform not in ["rk3588", "rk3576"]:
            print_color("Error: Invalid target platform. Please enter rk3588 or rk3576.", "red")
            sys.exit(1)
        print_color(f"Setting the frequency for the {args.target_platform} platform...", "cyan")
        library_path = os.path.expanduser(f"~/RKLLAMA/lib/fix_freq_{args.target_platform}.sh")
        command = f"sudo bash {library_path}"
        subprocess.run(command, shell=True)

    # Set the resource limits
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Start the API server with the chosen port
    print_color(f"Start the API at http://localhost:{port}", "blue")
    
    # Set Flask debug mode to match our debug flag
    flask_debug = DEBUG_MODE
    app.run(host='0.0.0.0', port=int(port), threaded=True, debug=flask_debug)

if __name__ == "__main__":
    main()