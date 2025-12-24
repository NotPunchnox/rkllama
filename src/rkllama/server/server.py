# Import libs
import sys, os, subprocess, resource, argparse, shutil, time, requests, json, datetime, logging
import re
from dotenv import load_dotenv
from huggingface_hub import hf_hub_url, HfFileSystem
from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from flask_cors import CORS
import random

# Local file
from rkllama.api.classes import *
from rkllama.api.rkllm import *
from rkllama.api.process import Request
import rkllama.api.variables as variables
from rkllama.api.debug_utils import check_response_format
from rkllama.api.format_utils import strtobool, openai_to_ollama_chat_request, openai_to_ollama_generate_request
from rkllama.api.model_utils import (
    extract_model_details, 
    get_huggingface_model_info,
    get_property_modelfile, get_model_full_options, find_rkllm_model_name
)
from rkllama.api.worker import WorkerManager

# Import the config module
import rkllama.config

# Check for debug mode using the improved method
DEBUG_MODE = rkllama.config.is_debug_mode()

# Ensure logs directory exists before configuring logging
logs_dir = rkllama.config.get_path("logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up logging with appropriate level based on debug mode
logging_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, "rkllama_server.log"))
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


# loaded_models = [] # Global variable to store the models loaded into memory
variables.worker_manager_rkllm = WorkerManager()


def create_modelfile(huggingface_path, From, system="", model_name=None):
    struct_modelfile = f"""
FROM="{From}"

HUGGINGFACE_PATH="{huggingface_path}"

SYSTEM="{system}"

TEMPERATURE={rkllama.config.get("model", "default_temperature")}

ENABLE_THINKING={rkllama.config.get("model", "default_enable_thinking")}

NUM_CTX={rkllama.config.get("model", "default_num_ctx")}

MAX_NEW_TOKENS={rkllama.config.get("model", "default_max_new_tokens")}

TOP_K={rkllama.config.get("model", "default_top_k")}

TOP_P={rkllama.config.get("model", "default_top_p")}

REPEAT_PENALTY={rkllama.config.get("model", "default_repeat_penalty")}

FREQUENCY_PENALTY={rkllama.config.get("model", "default_frequency_penalty")}

PRESENCE_PENALTY={rkllama.config.get("model", "default_presence_penalty")}

MIROSTAT={rkllama.config.get("model", "default_mirostat")}

MIROSTAT_TAU={rkllama.config.get("model", "default_mirostat_tau")}

MIROSTAT_ETA={rkllama.config.get("model", "default_mirostat_eta")}


"""

    # Use config for models path
    path = os.path.join(rkllama.config.get_path("models"), model_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Create the Modelfile and write the content
    with open(os.path.join(path, "Modelfile"), "w") as f:
        f.write(struct_modelfile)


def load_model(model_name, huggingface_path=None, system="", From=None, request_options=None):
    
    # Use config for models path
    model_dir = os.path.join(rkllama.config.get_path("models"), model_name)
    
    if not os.path.exists(model_dir):
        return None, f"Model directory '{model_name}' not found."
    
    if not os.path.exists(os.path.join(model_dir, "Modelfile")) and (huggingface_path is None and From is None):
        return None, f"Modelfile not found in '{model_name}' directory."
    elif huggingface_path is not None and From is not None:
        create_modelfile(huggingface_path=huggingface_path, From=From, system=system, model_name=model_name)
        time.sleep(0.1)
    
    # Load modelfile
    load_dotenv(os.path.join(model_dir, "Modelfile"), override=True)
    
    from_value = os.getenv("FROM")
    huggingface_path = os.getenv("HUGGINGFACE_PATH")
    vision_encoder = os.getenv("VISION_ENCODER")

    # View config Vars
    print_color(f"FROM: {from_value}\nHuggingFace Path: {huggingface_path}", "green")
    
    if not from_value or not huggingface_path:
        return None, "FROM or HUGGINGFACE_PATH not defined in Modelfile."

    # Change value of model_id with huggingface_path
    variables.model_id = huggingface_path

    # Get model parameters if not provided
    if not request_options:
        request_options = get_model_full_options(model_name, rkllama.config.get_path("models"), request_options)

    # Model loaded into memory
    model_loaded = variables.worker_manager_rkllm.add_worker(model_name, os.path.join(model_dir, from_value), model_dir, options=request_options)

    if not model_loaded:
        return None, f"Unexpected Error loading the model {model_name} into memory. Check the file .rkllm is not corrupted, properties in Modelfile (like Context Length allowed by the model) and resources available in the server"
    else:
        return None, None

def unload_model(model_name):
    # Relese the model from memory
    variables.worker_manager_rkllm.stop_worker(model_name)


app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

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
    # Return the list of available models using config path
    models_dir = rkllama.config.get_path("models")
    
    if not os.path.exists(models_dir):
        return jsonify({"error": f"The models directory {models_dir} is not found."}), 500

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
def rm_model():
    data = request.get_json(force=True)
    if "model" not in data:
        return jsonify({"error": "Please specify a model."}), 400

    model_name = data['model']

    model_path = os.path.join(rkllama.config.get_path("models"), model_name)
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model directory for '{model_name}' not found"}), 404

    # Check if model is currently loaded
    if variables.worker_manager_rkllm.exists_model_loaded(model_name):   
        if DEBUG_MODE:
            logger.debug(f"Unloading model '{model_name}' before deletion")
        unload_model(model_name)
    
    try:
        if DEBUG_MODE:
            logger.debug(f"Deleting model directory: {model_path}")
        shutil.rmtree(model_path)
        
        return jsonify({"message": f"The model has been successfully deleted!"}), 200
    except Exception as e:
        logger.error(f"Failed to delete model '{model_name}': {str(e)}")
        return jsonify({"error": f"Failed to delete model: {str(e)}"}), 500

    # os.remove(model_path)

# route to pull a model
@app.route('/pull', methods=['POST'])
def pull_model():
    @stream_with_context
    def generate_progress():
        data = request.get_json(force=True)
        if "model" not in data:
            yield "Error: Model not specified.\n"
            return

        splitted = data["model"].split('/')
        if len(splitted) < 3:
            yield f"Error: Invalid path '{data['model']}'\n"
            return

        # Check if custom name provided:
        model_name = splitted[len(splitted)-1] if "model_name" not in data else data["model_name"]
        file = splitted[2]
        repo = f"{splitted[0]}/{splitted[1]}"

        try:
            # Use Hugging Face HfFileSystem to get the file metadata
            fs = HfFileSystem()
            file_info = fs.info(repo + "/" + file)

            total_size = file_info["size"]  # File size in bytes
            if total_size == 0:
                yield "Error: Unable to retrieve file size.\n"
                return

            # Use config to get models path
            model_dir = os.path.join(rkllama.config.get_path("models"), model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Define a file to download
            local_filename = os.path.join(model_dir, file)

            # Create fonfiguration file for model
            create_modelfile(huggingface_path=repo, From=file, model_name=model_name)

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
    
    data = request.get_json(force=True)
    model_name = data.get('model_name', None)
    if model_name is None:
        return jsonify({"error": "Please enter the name of the model to be loaded."}), 400
    
    # Check if a model is currently loaded
    if variables.worker_manager_rkllm.exists_model_loaded(model_name):    
        return jsonify({"error": "A model is already loaded. Nothing to do."}), 200

    # Check if other params like "from" or "huggingface_path" for create modelfile
    if "from" in data or "huggingface_path" in data:
        _, error = load_model(model_name, From=data["from"], huggingface_path=data["huggingface_path"])
    else:
        _, error = load_model(model_name)

    if error:
        return jsonify({"error": error}), 400

    return jsonify({"message": f"Model {model_name} loaded successfully."}), 200

# Route to unload a model from the NPU
@app.route('/unload_model', methods=['POST'])
def unload_model_route():
    
    data = request.get_json(force=True)
    model_name = data.get('model_name', None)
    if model_name is None:
        return jsonify({"error": "Please enter the name of the model to be unloaded."}), 400
    
    if not variables.worker_manager_rkllm.exists_model_loaded(model_name):   
        return jsonify({"error": f"No model {model_name} are currently loaded."}), 400

    unload_model(model_name)
    return jsonify({"message": f"Model {model_name} successfully unloaded!"}), 200


# Route to unload a model from the NPU
@app.route('/unload_models', methods=['POST'])
def unload_models_route():
    variables.worker_manager_rkllm.stop_all
    return jsonify({"message": "Models successfully unloaded!"}), 200

# Route to retrieve the current models
@app.route('/current_models', methods=['GET'])
@app.route('/api/ps', methods=['GET'])
def get_current_models():
    
    # Get the models info from Modelfile and HF
    models_dir = rkllama.config.get_path("models")
    models_info = {}
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    size = os.path.getsize(os.path.join(subdir_path, file))
                    
                    # Extract parameter size and quantization details if available
                    model_details = extract_model_details(file)
                    
                    models_info[subdir] = {
                        "name": subdir,        # Use simplified name like qwen:3b
                        "model": subdir,       # Match Ollama's format
                        "modified_at": datetime.datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(subdir_path, file))
                        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "size": size,
                        "digest": "",               # Ollama field (not used but included for compatibility)
                        "details": {
                            "format": "rkllm",
                            "family": "llama",      # Default family
                            "parameter_size": model_details.get("parameter_size", "Unknown"),
                            "quantization_level": model_details.get("quantization_level", "Unknown")
                        }
                    }
                    break

    # Loop over the models currently running
    models_running = []
    for model in variables.worker_manager_rkllm.workers.keys():
        worker_model_info = variables.worker_manager_rkllm.workers[model].worker_model_info
        model_info = {
                    "name": model,
                    "model": model,
                    "size": worker_model_info.size,
                    "digest": models_info[model]["digest"],
                    "details": {
                        "parent_model": "",
                        "format": models_info[model]["details"]["format"],
                        "family": models_info[model]["details"]["family"],
                        "families": [
                            models_info[model]["details"]["family"]
                        ],
                        "parameter_size": models_info[model]["details"]["parameter_size"],
                        "quantization_level": models_info[model]["details"]["quantization_level"]
                    },
                    "expires_at": worker_model_info.expires_at.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    "loaded_at": worker_model_info.loaded_at.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    "base_domain_id": worker_model_info.base_domain_id,
                    "last_call": worker_model_info.last_call.strftime('%Y-%m-%d %H:%M:%S.%f')
                    }
        models_running.append(model_info)

    return jsonify({ "models" : models_running}), 200
    
    

# Route to make a request to the model
@app.route('/generate', methods=['POST'])
def recevoir_message():
    global modele_rkllm

    if not modele_rkllm:
        return jsonify({"error": "No models are currently loaded."}), 400

    # define modelfile path
    modelfile = os.path.join(modele_rkllm.model_dir, "Modelfile")

    variables.verrou.acquire()
    return Request(modele_rkllm, modelfile)


# OpenAI API compatibility routes

@app.route('/v1/models', methods=['GET'])
def list_openai_models():
    # Return models in OpenAI API format
    models_dir = rkllama.config.get_path("models")
    
    # Check if models exists
    if not os.path.exists(models_dir):
        return jsonify({"object": "list", "data": []}), 200

    # Loop over the available models 
    models = []
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm") or file.endswith(".rknn") or file == "unet": # Include Stable Diffusion models and Piper Voices
                    models.append({
                        "id": subdir,      
                        "object": "model",      
                        "created": int(datetime.datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(subdir_path, file))
                        ).timestamp()),
                        "owned_by": "rkllama"
                    })
                    break

    return jsonify({"object": "list", "data": models}), 200


# Default route
@app.route('/v1/models/<model_name>', methods=['GET'])
def list_openai_model(model_name):
    
    # Return models in OpenAI API format
    models_dir = rkllama.config.get_path("models")
    
    # Check if models exists
    if not os.path.exists(models_dir):
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    # Loop over the available models to search for the required one
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm") or file == "unet": # Include Stable Diffusion models
                    if subdir == model_name:
                       return jsonify({
                          "id": subdir,      
                          "object": "model",      
                          "created": int(datetime.datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(subdir_path, file))
                           ).timestamp()),
                          "owned_by": "rkllama"
                    }), 200     
                    
    return jsonify({"error": f"Model '{model_name}' not found"}), 404


# Ollama API compatibility routes

@app.route('/api/tags', methods=['GET'])
def list_ollama_models():
    # Return models in Ollama API format
    models_dir = rkllama.config.get_path("models")
    
    if not os.path.exists(models_dir):
        return jsonify({"models": []}), 200

    models = []
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    size = os.path.getsize(os.path.join(subdir_path, file))
                    
                    # Extract parameter size and quantization details if available
                    model_details = extract_model_details(file)
                    
                    models.append({
                        "name": subdir,        # Use simplified name like qwen:3b
                        "model": subdir,       # Match Ollama's format
                        "modified_at": datetime.datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(subdir_path, file))
                        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "size": size,
                        "digest": "",               # Ollama field (not used but included for compatibility)
                        "details": {
                            "format": "rkllm",
                            "family": "llama",      # Default family
                            "parameter_size": model_details.get("parameter_size", "Unknown"),
                            "quantization_level": model_details.get("quantization_level", "Unknown")
                        }
                    })
                    break

    return jsonify({"models": models}), 200

@app.route('/api/show', methods=['POST'])
def show_model_info():

    ##### Github Copilot Start Workaround
    request_data = request.get_data().decode('UTF-8')
    request_data = request_data.replace("\'", "\"")  
    data = json.loads(request_data) if request_data else {}
    model_name = data.get('name') if "name" in data else data.get('model')
    ##### # Github Copilot End

    # Remove possible namespace in model name. Ollama API allows namespace/model
    model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

    if DEBUG_MODE:
        logger.debug(f"API show request data: {data}")
    
    if not model_name:
        return jsonify({"error": "Missing model name"}), 400
    
    model_dir = os.path.join(rkllama.config.get_path("models"), model_name)
    model_rkllm = find_rkllm_model_name(model_dir)

    if not os.path.exists(model_dir):
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    # Read modelfile content if available
    modelfile_path = os.path.join(model_dir, "Modelfile")
    modelfile_content = ""
    system_prompt = ""
    template = "{{ .Prompt }}"
    license_text = ""
    huggingface_path = None
    
    if os.path.exists(modelfile_path):
        with open(modelfile_path, "r") as f:
            modelfile_content = f.read()
            
            # Extract system prompt if available
            system_match = re.search(r'SYSTEM="(.*?)"', modelfile_content, re.DOTALL)
            if system_match:
                system_prompt = system_match.group(1).strip()
            
            # Check for template pattern
            template_match = re.search(r'TEMPLATE="(.*?)"', modelfile_content, re.DOTALL)
            if template_match:
                template = template_match.group(1).strip()
            
            # Check for LICENSE pattern (some modelfiles have this)
            license_match = re.search(r'LICENSE="(.*?)"', modelfile_content, re.DOTALL)
            if license_match:
                license_text = license_match.group(1).strip()
            
            # Extract HuggingFace path for API access
            hf_path_match = re.search(r'HUGGINGFACE_PATH="(.*?)"', modelfile_content, re.DOTALL)
            if hf_path_match:
                huggingface_path = hf_path_match.group(1).strip()
            
            # Extract temperature if available
            temp_match = re.search(r'TEMPERATURE=(\d+\.?\d*)', modelfile_content)
            if temp_match:
                try:
                    temperature = float(temp_match.group(1))
                except ValueError:
                    pass
    
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
    
    # Extract model details
    model_details = extract_model_details(model_rkllm)
    parameter_size = model_details.get("parameter_size", "Unknown")
    quantization_level = model_details.get("quantization_level", "Unknown")
    
    # Determine model family based on name patterns
    family = "llama"  # default family
    families = ["llama"]
    
    # Try to get enhanced information from Hugging Face API
    hf_metadata = get_huggingface_model_info(huggingface_path) if huggingface_path else None
    
    # Use HF metadata to improve model info if available
    if hf_metadata:
        # Extract tags from HF metadata
        tags = hf_metadata.get('tags', [])
        
        # Better determine model family based on HF tags or architecture field
        if hf_metadata.get('architecture') == 'qwen' or 'qwen' in tags or 'qwen2' in tags:
            family = "qwen2"
            families = ["qwen2"]
        elif hf_metadata.get('architecture') == 'mistral' or 'mistral' in tags:
            family = "mistral"
            families = ["mistral"]
        elif hf_metadata.get('architecture') == 'deepseek' or 'deepseek' in tags:
            family = "deepseek"
            families = ["deepseek"]
        elif hf_metadata.get('architecture') == 'phi' or 'phi' in tags:
            family = "phi"
            families = ["phi"]
        elif hf_metadata.get('architecture') == 'gemma' or 'gemma' in tags:
            family = "gemma"
            families = ["gemma"]
        elif 'tinyllama' in tags:
            family = "tinyllama"
            families = ["tinyllama", "llama"]
        elif any('llama-3' in tag for tag in tags) or any('llama3' in tag for tag in tags):
            family = "llama3"
            families = ["llama3", "llama"]
        elif any('llama-2' in tag for tag in tags) or any('llama2' in tag for tag in tags):
            family = "llama2"
            families = ["llama2", "llama"]
        
        # Extract model card metadata
        model_card = hf_metadata.get('cardData', {})
        
        # Better parameter size from HF metadata
        parameter_count = None
        if 'params' in model_card:
            try:
                params = int(model_card['params'])
                if params >= 1_000_000_000:
                    parameter_size = f"{params/1_000_000_000:.1f}B".replace('.0B', 'B')
                    # Also store the raw parameter count for model_info
                    parameter_count = params
            except (ValueError, TypeError):
                parameter_count = None
        else:
            parameter_count = None
        
        # Extract quantization info
        if 'quantization' in hf_metadata:
            quantization_level = hf_metadata['quantization']
        
        # Better license information
        if 'license' in hf_metadata and not license_text:
            license_text = hf_metadata['license']
    else:
        # Fallback to pattern matching if no HF metadata
        if re.search(r'(?i)Qwen', model_name):
            family = "qwen2"
            families = ["qwen2"]
        elif re.search(r'(?i)Mistral', model_name):
            family = "mistral"
            families = ["mistral"]
        elif re.search(r'(?i)DeepSeek', model_name):
            family = "deepseek"
            families = ["deepseek"]
        elif re.search(r'(?i)Phi', model_name):
            family = "phi"
            families = ["phi"]
        elif re.search(r'(?i)Gemma', model_name):
            family = "gemma"
            families = ["gemma"]
        elif re.search(r'(?i)TinyLlama', model_name):
            family = "tinyllama"
            families = ["tinyllama", "llama"]
        elif re.search(r'(?i)Llama[-_]?3', model_name):
            family = "llama3"
            families = ["llama3", "llama"]
        elif re.search(r'(?i)Llama[-_]?2', model_name):
            family = "llama2"
            families = ["llama2", "llama"]
        
        parameter_count = None
    
    # Convert modelfile to Ollama-compatible format
    ollama_modelfile = f"# Modelfile generated by \"ollama show\"\n"
    ollama_modelfile += f"# To build a new Modelfile based on this, replace FROM with:\n"
    ollama_modelfile += f"# FROM {model_name}\n\n"
    
    # Change this section to use a more compatible FROM format
    # Instead of absolute paths, use the model file name which is more compatible with Ollama
    # Original: model_blob_path = f"{model_dir}/{model_file}"
    
    if DEBUG_MODE:
        # In debug mode, use absolute paths to help with troubleshooting
        model_blob_path = f"{model_dir}/{model_file}"
        ollama_modelfile += f"FROM {model_blob_path}\n"
    else:
        # In normal mode, use the simplified name format that Ollama clients expect
        ollama_modelfile += f"FROM {model_name}\n"
    
    if template != "{{ .Prompt }}":
        ollama_modelfile += f'TEMPLATE """{template}"""\n'
    
    if system_prompt:
        ollama_modelfile += f'SYSTEM "{system_prompt}"\n'
    
    if license_text:
        ollama_modelfile += f'LICENSE """{license_text}"""\n'
    
    # Additional model info from HF
    model_description = ""
    repo_url = None
    if hf_metadata:
        model_description = hf_metadata.get('description', '').strip()
        
        # Add description comment to modelfile if available
        if model_description:
            desc_lines = model_description.split('\n')
            desc_comment = '\n'.join([f"# {line}" for line in desc_lines[:5]])  # First 5 lines only
            ollama_modelfile = desc_comment + "\n\n" + ollama_modelfile
        
        # Extract repo URL if available
        if huggingface_path:
            repo_url = f"https://huggingface.co/{huggingface_path}"
    
    # Parse parameter size into numeric format
    numeric_param_size = None
    if parameter_size != "Unknown":
        param_match = re.search(r'(\d+\.?\d*)B', parameter_size)
        if param_match:
            try:
                size_in_billions = float(param_match.group(1))
                numeric_param_size = int(size_in_billions * 1_000_000_000)
            except ValueError:
                pass
    
    # Use parameter_count from HF metadata if available, otherwise use parsed value
    if parameter_count is None and numeric_param_size is not None:
        parameter_count = numeric_param_size
    elif parameter_count is None:
        # Default fallback
        if "7B" in model_name or "7b" in model_name:
            parameter_count = 7000000000
        elif "3B" in model_name or "3b" in model_name:
            parameter_count = 3000000000
        elif "1.5B" in model_name or "1.5b" in model_name:
            parameter_count = 1500000000
        else:
            parameter_count = 0
    
    # Extract base model name (without fine-tuning suffixes)
    base_name = model_name.split('-')[0]
    
    # Determine finetune type if present
    finetune = None
    if "instruct" in model_name.lower():
        finetune = "Instruct"
    elif "chat" in model_name.lower():
        finetune = "Chat"
    
    # Build a more complete model_info dict with architecture details
    model_info = {
        "general.architecture": family,
        "general.base_model.0.name": f"{base_name} {parameter_size}",
        "general.base_model.0.organization": family.capitalize(),
        "general.basename": base_name,
        "general.file_type": 15,  # RKLLM file type
        "general.parameter_count": parameter_count,
        "general.quantization_version": 2,
        "general.size_label": parameter_size,
        "general.tags": ["chat", "text-generation"],
        "general.type": "model",
        "tokenizer.ggml.pre": family
    }
    
    # Add repo URL if available
    if repo_url:
        model_info["general.base_model.0.repo_url"] = repo_url
        model_info["general.base_model.count"] = 1
    
    # Add finetune info if available
    if finetune:
        model_info["general.finetune"] = finetune
    
    # Add license info if available
    if license_text:
        license_name = "other"
        license_link = ""
        
        # Try to detect common licenses
        if "apache" in license_text.lower():
            license_name = "apache-2.0"
        elif "mit" in license_text.lower():
            license_name = "mit"
        elif "qwen research" in license_text.lower():
            license_name = "qwen-research"
        
        if huggingface_path:
            license_link = f"https://huggingface.co/{huggingface_path}/blob/main/LICENSE"
        
        model_info["general.license"] = license_name
        if license_link:
            model_info["general.license.link"] = license_link
        model_info["general.license.name"] = license_name
    
    # Add language info if we can detect it
    if hf_metadata and 'languages' in hf_metadata:
        model_info["general.languages"] = hf_metadata['languages']
    else:
        # Default to English
        model_info["general.languages"] = ["en"]
    
    # Add architecture-specific parameters based on model family
    if family == "qwen2":
        model_info.update({
            "qwen2.attention.head_count": 16,
            "qwen2.attention.head_count_kv": 2,
            "qwen2.attention.layer_norm_rms_epsilon": 0.000001,
            "qwen2.block_count": 36 if "3B" in parameter_size else 24,
            "qwen2.context_length": 32768,
            "qwen2.embedding_length": 2048 if "3B" in parameter_size else 1536,
            "qwen2.feed_forward_length": 11008 if "3B" in parameter_size else 8192,
            "qwen2.rope.freq_base": 1000000
        })
    elif family == "llama" or family == "llama2" or family == "llama3":
        model_info.update({
            f"{family}.attention.head_count": 32,
            f"{family}.attention.head_count_kv": 4,
            f"{family}.attention.layer_norm_rms_epsilon": 0.000001,
            f"{family}.block_count": 32,
            f"{family}.context_length": 4096,
            f"{family}.embedding_length": 4096,
            f"{family}.feed_forward_length": 11008,
            f"{family}.rope.freq_base": 10000
        })
    elif family == "mistral":
        model_info.update({
            "mistral.attention.head_count": 32,
            "mistral.attention.head_count_kv": 8,
            "mistral.attention.layer_norm_rms_epsilon": 0.000001,
            "mistral.block_count": 32,
            "mistral.context_length": 8192,
            "mistral.embedding_length": 4096,
            "mistral.feed_forward_length": 14336
        })
    
    # Calculate modified timestamp
    modified_at = datetime.datetime.fromtimestamp(
        os.path.getmtime(file_path)
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    
    # Format parameters string nicely
    parameters_str = parameter_size
    if parameters_str == "Unknown" and parameter_count:
        if parameter_count >= 1_000_000_000:
            parameters_str = f"{parameter_count/1_000_000_000:.1f}B".replace('.0B', 'B')
        else:
            parameters_str = f"{parameter_count/1_000_000:.1f}M".replace('.0M', 'M')
    
    # Capabilities based on model family. ### Github Copilot requires this
    capabilities = ["completion"]
    if family in ["qwen2", "phi", "llama3", "mistral"]:
        capabilities.append("tools")
	
    # Prepare response with enhanced metadata
    response = {
        "license": license_text or "Unknown",
        "modelfile": ollama_modelfile,
        "parameters": parameters_str,
        "template": template,
        "system": system_prompt,
        "name": model_name,
        "details": {
            "parent_model": huggingface_path or "",
            "format": "rkllm",
            "family": family,
            "families": families,
            "parameter_size": parameter_size,
            "quantization_level": quantization_level
        },
        "model_info": model_info,
        "size": size,
        "capabilities": capabilities,
        "modified_at": modified_at
    }
    
    # Add Hugging Face specific fields if available
    if hf_metadata:
        response["huggingface"] = {
            "repo_id": huggingface_path,
            "description": model_description[:500] if model_description else "",  # Truncate if too long
            "tags": hf_metadata.get('tags', []),
            "downloads": hf_metadata.get('downloads', 0),
            "likes": hf_metadata.get('likes', 0)
        }
    
    return jsonify(response), 200

@app.route('/api/create', methods=['POST'])
def create_model():
    data = request.get_json(force=True)
    model_name = data.get('name')
    modelfile = data.get('modelfile', '')
    
    # Remove possible namespace in model name. Ollama API allows namespace/model
    model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

    if DEBUG_MODE:
        logger.debug(f"API create request data: {data}")

    if not model_name:
        return jsonify({"error": "Missing model name"}), 400
    
    model_dir = os.path.join(rkllama.config.get_path("models"), model_name)
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
    # TODO: Implement the pull model
    data = request.get_json(force=True)
    model = data.get('name',data.get('model'))
    
    if DEBUG_MODE:
        logger.debug(f"API pull request data: {data}")

    if not model:
        return jsonify({"error": "Missing model name"}), 400

    # Ollama API uses application/x-ndjson for streaming
    response_stream = pull_model()  # Call the existing function directly
    response_stream.content_type = 'application/x-ndjson'
    return response_stream

@app.route('/api/delete', methods=['DELETE'])
def delete_model_ollama():
    data = request.get_json(force=True)
    model_name = data.get('name')

    # Remove possible namespace in model name. Ollama API allows namespace/model
    model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

    if DEBUG_MODE:
        logger.debug(f"API delete request data: {data}")

    if not model_name:
        return jsonify({"error": "Missing model name"}), 400

    model_path = os.path.join(rkllama.config.get_path("models"), model_name)
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model directory for '{model_name}' not found"}), 404

    # Check if model is currently loaded
    if variables.worker_manager_rkllm.exists_model_loaded(model_name):   
        if DEBUG_MODE:
            logger.debug(f"Unloading model '{model_name}' before deletion")
        unload_model(model_name)
    
    try:
        if DEBUG_MODE:
            logger.debug(f"Deleting model directory: {model_path}")
        shutil.rmtree(model_path)
        
        return jsonify({"message": f"The model has been successfully deleted!"}), 200
    except Exception as e:
        logger.error(f"Failed to delete model '{model_name}': {str(e)}")
        return jsonify({"error": f"Failed to delete model: {str(e)}"}), 500


@app.route('/api/generate', methods=['POST'])
@app.route('/v1/completions', methods=['POST'])
def generate_ollama():
    
    lock_acquired = False  # Track lock status
    is_openai_request = request.path.startswith('/v1/completions')

    try:
        data = request.get_json(force=True)

        if is_openai_request:
           if DEBUG_MODE:
              logger.debug(f"API OpenAI completions request data: {data}")
           data = openai_to_ollama_generate_request(data)

        model_name = data.get('model')
        prompt = data.get('prompt')
        system = data.get('system', '')
        stream = data.get('stream', True)
        enable_thinking = data.get('enable_thinking', (data.get('think', None))) # Ollama now uses 'think' in some versions
        images = data.get('images', None)  # For multimodal inputs
        
        # Support format options for structured JSON output
        format_spec = data.get('format')
        options = data.get('options', {})
        
        # Remove possible namespace in model name. Ollama API allows namespace/model
        model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

        if DEBUG_MODE:
            logger.debug(f"API generate request data: {data}")

        if not model_name:
            return jsonify({"error": "Missing model name"}), 400

        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        # Get Thinking setting from modelfile if not provided
        if enable_thinking is None:
            model_thinking_enabled = get_property_modelfile(model_name, 'ENABLE_THINKING', rkllama.config.get_path("models"))
            enable_thinking = strtobool(model_thinking_enabled) if bool(model_thinking_enabled) else False # Disabled by default

        # Get all model options
        options = get_model_full_options(model_name, rkllama.config.get_path("models"), options) 

        # Load model if needed
        if not variables.worker_manager_rkllm.exists_model_loaded(model_name):    
            _, error = load_model(model_name, request_options=options)
            if error:
                return jsonify({"error": f"Failed to load model '{model_name}': {error}"}), 500
        
        # Acquire lock before processing
        variables.verrou.acquire()
        lock_acquired = True
        
        # DIRECTLY use the GenerateEndpointHandler instead of the process_ollama_generate_request wrapper
        from rkllama.api.server_utils import GenerateEndpointHandler
        return GenerateEndpointHandler.handle_request(
            model_name=model_name,
            prompt=prompt,
            system=system,
            stream=stream,
            format_spec=format_spec,
            options=options,
            enable_thinking=enable_thinking,
            is_openai_request=is_openai_request,
            images=images
        )
    except Exception as e:
        if DEBUG_MODE:
            logger.exception(f"Error in generate_ollama: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Only release if we acquired it
        if lock_acquired and variables.verrou.locked():
            variables.verrou.release()


# Also update the chat endpoint for consistency
@app.route('/api/chat', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def chat_ollama():
    
    lock_acquired = False  # Track lock status
    is_openai_request = request.path.startswith('/v1/chat/completions')
        
    try:
        data = request.get_json(force=True)
        
        if is_openai_request:
           if DEBUG_MODE:
              logger.debug(f"API OpenAI chat request data: {data}")
           data = openai_to_ollama_chat_request(data)
        
        model_name = data.get('model')
        messages = data.get('messages', [])
        system = data.get('system', '')
        stream = data.get('stream', True)
        tools = data.get('tools', None)
        enable_thinking = data.get('enable_thinking', (data.get('think', None))) # Ollama now uses 'think' in some versions
        
        # Remove possible namespace in model name. Ollama API allows namespace/model
        model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

        # Extract format parameters - can be object or string
        format_spec = data.get('format')
        options = data.get('options', {})

        if DEBUG_MODE:
            logger.debug(f"API Ollama chat request data: {data}")
        
        # Get Thinking setting from modelfile if not provided
        if enable_thinking is None:
            model_thinking_enabled = get_property_modelfile(model_name, 'ENABLE_THINKING', rkllama.config.get_path("models"))
            enable_thinking = strtobool(model_thinking_enabled) if bool(model_thinking_enabled) else False # Disabled by default

        # Get all model options
        options = get_model_full_options(model_name, rkllama.config.get_path("models"), options) 

        # Check if we're starting a new conversation
        # A new conversation is one that doesn't include any assistant messages
        is_new_conversation = not any(msg.get('role') == 'assistant' for msg in messages)
        
        # Always reset system prompt for new conversations
        if is_new_conversation:
            variables.system = ""
            if DEBUG_MODE:
                logger.debug("New conversation detected, resetting system prompt")
        
        # Extract system message from messages array if present
        system_in_messages = False
        filtered_messages = []
        
        for message in messages:
            if message.get('role') == 'system':
                system = message.get('content', '')
                system_in_messages = True
                # Don't add system message to filtered messages
            else:
                filtered_messages.append(message)
                # CHeck for images in user messages for multimodal
                if message.get('role') == 'user' and 'images' in message:
                    if 'images' not in data:
                        data['images'] = []
                    data['images'].extend(message['images'])
        
        # Review the images in messages
        images = data.get('images', None)

        # Only use the extracted system message or explicit system parameter if provided
        if system_in_messages or system:
            variables.system = system
            messages = filtered_messages
            if DEBUG_MODE:
                logger.debug(f"Using system message: {system}")
        
        # Load model if needed
        if not variables.worker_manager_rkllm.exists_model_loaded(model_name):    
            
            if DEBUG_MODE:
                logger.debug(f"Loading model: {model_name}")
            _, error = load_model(model_name, request_options=options)
            if error:
                if DEBUG_MODE:
                    logger.error(f"Failed to load model {model_name}: {error}")
                return jsonify({"error": f"Failed to load model '{model_name}': {error}"}), 500
            if DEBUG_MODE:
                logger.debug(f"Model {model_name} loaded successfully")
        #else:
        #    
        #    rkllm_loaded = get_model_loaded_by_name(loaded_models,model_name)
        #    rkllm_model_request = rkllm_loaded.model_rkllm
        #    # If model is already loaded, check its options are the same for the current request
        #    if rkllm_model_request.rkllm_param.max_context_len != int(float(options.get("num_ctx", 0))) \
        #        or rkllm_model_request.rkllm_param.max_new_tokens != int(options.get("max_new_tokens")) \
        #        or rkllm_model_request.rkllm_param.top_k != int(options.get("top_k")) \
        #        or round(rkllm_model_request.rkllm_param.top_p,2) != round(float(options.get("top_p")),2) \
        #        or round(rkllm_model_request.rkllm_param.temperature,2) != round(float(options.get("temperature")),2) \
        #        or round(rkllm_model_request.rkllm_param.repeat_penalty,2) != round(float(options.get("repeat_penalty")),2) \
        #        or round(rkllm_model_request.rkllm_param.frequency_penalty,2) != round(float(options.get("frequency_penalty")),2) \
        #        or round(rkllm_model_request.rkllm_param.presence_penalty,2) != round(float(options.get("presence_penalty")),2) \
        #        or rkllm_model_request.rkllm_param.mirostat != int(options.get("mirostat")) \
        #        or round(rkllm_model_request.rkllm_param.mirostat_tau,2) != round(float(options.get("mirostat_tau")),2) \
        #        or round(rkllm_model_request.rkllm_param.mirostat_eta,2) != round(float(options.get("mirostat_eta")),2):
        #    
        #        # Update model parameters if they differ
        #        if DEBUG_MODE:
        #            logger.debug(f"Updating model parameters for {model_name} with options: {options}")
        #        
        #        unload_model(model_name)
        #        
        #        if DEBUG_MODE:
        #            logger.debug(f"Reoading model: {model_name}")
        #        modele_instance, error = load_model(model_name, request_options=options)
        #        if error:
        #            if DEBUG_MODE:
        #                logger.error(f"Failed to reload model {model_name}: {error}")
        #            return jsonify({"error": f"Failed to reload model '{model_name}': {error}"}), 500
        #        rkllm_model_request = modele_instance
        #        if DEBUG_MODE:
        #            logger.debug(f"Model {model_name} reloaded successfully")
                
        # Store format settings in model instance
        #if rkllm_model_request:
        #    rkllm_model_request.format_schema = format_spec
        #    rkllm_model_request.format_options = options
        
        # Acquire lock before processing the request
        variables.verrou.acquire()
        lock_acquired = True  # Mark lock as acquired
        
        # Create custom request for processing
        custom_req = type('obj', (object,), {
            'json': {
                "model": model_name,
                "messages": messages,
                "stream": stream,
                "system": system,
                "format": format_spec,
                "options": options,
                "tools": tools,
                "enable_thinking": enable_thinking,
                "images": images
            },
            'path': '/api/chat'
        })
        
        # Set a flag on the custom request to indicate it should not release the lock
        # as we'll handle it here
        custom_req.handle_lock = False
        
        # Process the request - this won't release the lock
        from rkllama.api.server_utils import ChatEndpointHandler
        return ChatEndpointHandler.handle_request(
              model_name=model_name,
              messages=messages,
              system=system,
              stream=stream,
              format_spec=format_spec,
              options=options,
              tools=tools,
              enable_thinking=enable_thinking,
              is_openai_request=is_openai_request,
              images=images)

    except Exception as e:
        logger.exception("Error in chat_ollama")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Only release if we acquired it
        if lock_acquired and variables.verrou.locked():
            if DEBUG_MODE:
                logger.debug("Releasing lock in chat_ollama")
            variables.verrou.release()


# Only include debug endpoint if in debug mode
if DEBUG_MODE:
    @app.route('/api/debug', methods=['POST'])
    def debug_streaming():
        """Endpoint to diagnose streaming issues"""
        data = request.get_json(force=True)
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
@app.route('/api/embed', methods=['POST'])
@app.route('/v1/embeddings', methods=['POST'])
def embeddings_ollama():
    
    lock_acquired = False  # Track lock status
    is_openai_request = request.path.startswith('/v1/embeddings')

    try:
        data = request.get_json(force=True)
        
        if is_openai_request:
           if DEBUG_MODE:
              logger.debug(f"API OpenAI embedding request data: {data}")
              # Parameter from OpenAI "encoding_format" not supported in Ollama
        
        model_name = data.get('model')
        input_text = data.get('input', data.get('prompt', None)) # Include legacy deprecated api/embedding with prompt
        truncate = data.get('truncate', True)
        keep_alive = data.get('keep_alive', False)
        options = data.get('options', {})
        
        # Remove possible namespace in model name. Ollama API allows namespace/model
        model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

        if DEBUG_MODE:
            logger.debug(f"API embedding request data: {data}")

        if not model_name:
            return jsonify({"error": "Missing model name"}), 400

        if not input_text:
            return jsonify({"error": "Missing input"}), 400

        # Get all model options
        options = get_model_full_options(model_name, rkllama.config.get_path("models"), options) 

        # Load model if needed
        if not variables.worker_manager_rkllm.exists_model_loaded(model_name):    
            _, error = load_model(model_name, request_options=options)
            if error:
                return jsonify({"error": f"Failed to load model '{model_name}': {error}"}), 500
        variables.verrou.acquire()
        lock_acquired = True
        
        # DIRECTLY use the EmbedEndpointHandler instead of the process_ollama_generate_request wrapper
        from rkllama.api.server_utils import EmbedEndpointHandler
        return EmbedEndpointHandler.handle_request(
            model_name=model_name,
            input_text=input_text,
            truncate=truncate,
            keep_alive=keep_alive,
            options=options,
            is_openai_request=is_openai_request
        )
    except Exception as e:
        if DEBUG_MODE:
            logger.exception(f"Error in embeddings_ollama: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Only release if we acquired it
        if lock_acquired and variables.verrou.locked():
            variables.verrou.release()


# Version endpoint for Ollama API compatibility
@app.route('/api/version', methods=['GET'])
def ollama_version():
    """Return a dummy version to be compatible with Ollama clients"""
    return jsonify({
        "version": "0.0.44"
    }), 200


@app.route('/v1/images/generations', methods=['POST'])
def generate_image_openai():
    
    lock_acquired = False  # Track lock status

    try:
        data = request.get_json(force=True)
        
        # Supported OpenAI parameters by RKNN
        prompt = data.get('prompt')
        model_name = data.get('model', None)
        size = data.get('size', "512x512")        
        stream = data.get('stream', False)
        output_format = data.get('output_format', 'png')
        num_images = data.get('n', 1)
        response_format = data.get('response_format', 'b64_json')

        # Non supported OpenAI parameters by RKNN
        background = data.get('background', None)
        moderation = data.get('moderation', None)
        output_compression = data.get('output_compression', None)
        partial_images = data.get('partial_images', None)
        quality = data.get('quality', None)
        style = data.get('style', None)
        user = data.get('user', None)

        # Not OpenAI parameters, but used by rkllama
        seed = data.get('seed', random.randint(1, 99))
        num_inference_steps = data.get('num_inference_steps', 4)
        guidance_scale = data.get('guidance_scale', 7.5)
        
        
        # Remove possible namespace in model name. Ollama API allows namespace/model
        model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

        
        if DEBUG_MODE:
            logger.debug(f"API OpenAI Generate Image request data: {data}")

        # Acquire lock before processing the request
        variables.verrou.acquire()
        lock_acquired = True  # Mark lock as acquired
        
        # Process the request - this won't release the lock
        from rkllama.api.server_utils import GenerateImageEndpointHandler
        return GenerateImageEndpointHandler.handle_request(
              model_name=model_name,
              prompt=prompt,
              stream=stream,
              size=size,
              response_format=response_format,
              output_format=output_format,
              num_images=num_images,
              seed=seed,
              num_inference_steps=num_inference_steps,
              guidance_scale=guidance_scale)

    except Exception as e:
        logger.exception("Error in generate_image_openai")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Only release if we acquired it
        if lock_acquired and variables.verrou.locked():
            if DEBUG_MODE:
                logger.debug("Releasing lock in generate_image_openai")
            variables.verrou.release()


# Default route
@app.route('/files/<model_name>/images/<file_name>', methods=['GET'])
def get_generated_image(model_name, file_name):
    
    # Get the model directory
    model_dir = os.path.join(rkllama.config.get_path("models"), model_name)

    # Get the file path
    file_path = f"{model_dir}/images/{file_name}"

    # Return File as attachment
    return send_file(file_path, as_attachment=True)


@app.route('/v1/audio/speech', methods=['POST'])
def generate_speech_openai():
    
    lock_acquired = False  # Track lock status

    try:
        data = request.get_json(force=True)
        
        # Supported OpenAI parameters by RKNN
        input = data.get('input')
        model_name = data.get('model', None)
        voice = data.get('voice', None)        
        response_format = data.get('response_format', 'mp3')
        stream_format = data.get('stream_format', 'audio')
        speed = data.get('speed', None)
        
        # Non supported OpenAI parameters by Piper
        instructions = data.get('instructions', None)

        # Not OpenAI parameters, but used by rkllama/piper
        volume = data.get('volume', None)
        length_scale = data.get('length_scale', None)
        noise_scale = data.get('noise_scale', None)
        noise_w_scale = data.get('noise_w_scale', None)
        normalize_audio = data.get('normalize_audio', None)
        
        # Remove possible namespace in model name. Ollama API allows namespace/model
        model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

        # Calculate the Piper Lenght Scale bases in speed OpenAI if requested
        if speed:
            length_scale = 1 / speed

        if DEBUG_MODE:
            logger.debug(f"API OpenAI Generate Speech request data: {data}")

        # Acquire lock before processing the request
        variables.verrou.acquire()
        lock_acquired = True  # Mark lock as acquired
        
        # Process the request - this won't release the lock
        from rkllama.api.server_utils import GenerateSpeechEndpointHandler
        return GenerateSpeechEndpointHandler.handle_request(
              model_name=model_name,
              input=input,
              voice=voice,
              response_format=response_format,
              stream_format=stream_format,
              volume=volume,
              length_scale=length_scale,
              noise_scale=noise_scale,
              noise_w_scale=noise_w_scale,
              normalize_audio=normalize_audio)

    except Exception as e:
        logger.exception("Error in generate_speech_openai")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Only release if we acquired it
        if lock_acquired and variables.verrou.locked():
            if DEBUG_MODE:
                logger.debug("Releasing lock in generate_speech_openai")
            variables.verrou.release()


@app.route('/v1/audio/transcriptions', methods=['POST'])
def generate_transcriptions_openai():
    
    lock_acquired = False  # Track lock status

    try:
        form = request.form

        # CHeck the file uploaded file
        if "file" not in request.files:
            return jsonify({"error": "file field missing"}), 400
        
        # Supported OpenAI parameters by RKNN
        file = request.files["file"]
        model_name = form.get('model', None)
        language = form.get('language', None)        
        response_format = form.get('response_format', 'text')
        stream = form.get('stream', False)
        
        # Non supported OpenAI parameters by Piper
        chunking_strategy = form.get('chunking_strategy', None)
        known_speaker_names = form.get('known_speaker_names', None)
        known_speaker_references = form.get('known_speaker_references', None)
        prompt = form.get('prompt', None)
        temperature = form.get('temperature', None)
        timestamp_granularities = form.get('timestamp_granularities', None)

        # Not OpenAI parameters, but used by rkllama/omnilingual

        # Remove possible namespace in model name. Ollama API allows namespace/model
        model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

        # Format bool values
        stream = strtobool(stream) if bool(stream) else False # Disabled by default

        if DEBUG_MODE:
            logger.debug(f"API OpenAI Generate Transcription request data: {form}")

        # Acquire lock before processing the request
        variables.verrou.acquire()
        lock_acquired = True  # Mark lock as acquired
        
        # Process the request - this won't release the lock
        from rkllama.api.server_utils import GenerateTranscriptionsEndpointHandler
        return GenerateTranscriptionsEndpointHandler.handle_request(
              model_name=model_name,
              file=file,
              language=language,
              response_format=response_format,
              stream=stream)

    except Exception as e:
        logger.exception("Error in generate_transcriptions_openai")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Only release if we acquired it
        if lock_acquired and variables.verrou.locked():
            if DEBUG_MODE:
                logger.debug("Releasing lock in generate_transcriptions_openai")
            variables.verrou.release()


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
    parser.add_argument('--processor', type=str, help="Processor: rk3588/rk3576.")
    parser.add_argument('--port', type=str, help="Port for the server")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--models', type=str, help="Path whe models will be loaded from")
    args = parser.parse_args()

    # Load arguments into the config
    rkllama.config.load_args(args)

    # Validate directories for running
    rkllama.config.validate()
    
    # Set debug mode if specified in config - using the improved method
    global DEBUG_MODE
    DEBUG_MODE = rkllama.config.is_debug_mode()
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        print_color("Debug mode enabled", "yellow")
        rkllama.config.display()
        os.environ["RKLLAMA_DEBUG"] = "1"  # Explicitly set for subprocess consistency
  
    # Get port from config
    port = rkllama.config.get("server", "port", "8080")

    # Check the processor
    processor = rkllama.config.get("platform", "processor", None)
    if not processor:
        print_color("Error: processor not configured", "red")
        sys.exit(1)
    else:
        if processor not in ["rk3588", "rk3576"]:
            print_color("Error: Invalid processor. Please enter rk3588 or rk3576.", "red")
            sys.exit(1)
        if os.getuid() == 0:
            print_color(f"Setting the frequency for the {processor} platform...", "cyan")
            library_path = importlib.resources.files("rkllama.lib") / f"fix_freq_{processor}.sh"
            #library_path = os.path.join(rkllama.config.get_path("lib"), f"fix_freq_{processor}.sh")

            # Pass debug flag as parameter to the shell script
            debug_param = "1" if DEBUG_MODE else "0"
            command = f"bash {library_path} {debug_param}"
            subprocess.run(command, shell=True)

    # Set the resource limits
    if os.getuid() == 0:
        resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Start the API server with the chosen port
    print_color(f"Start the API at http://localhost:{port}", "blue")

    # Set Flask debug mode to match our debug flag
    flask_debug = rkllama.config.is_debug_mode()
    app.run(host=rkllama.config.get("server", "host", "0.0.0.0"), port=int(port), threaded=True, debug=flask_debug)

if __name__ == "__main__":
    main()
