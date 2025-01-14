import sys
import os
import subprocess
import resource
import argparse
import requests
from huggingface_hub import hf_hub_url, HfFileSystem
from flask import Flask, request, jsonify, Response, stream_with_context

from src.classes import *
from src.rkllm import *
from src.variables import * 
from src.process import Request

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

current_model = None  # Global variable for storing the loaded model
modele_rkllm = None  # Model instance

def load_model(model_name):
    global modele_rkllm
    model_path = os.path.expanduser(f"~/RKLLAMA/models/{model_name}")
    if not os.path.exists(model_path):
        return None, f"Model {model_name} not found in /models folder."
    
    # Model initialization
    modele_rkllm = RKLLM(model_path)
    return modele_rkllm, None

def unload_model():
    global modele_rkllm
    if modele_rkllm:
        modele_rkllm.release()
        modele_rkllm = None

app = Flask(__name__)

# Routes:
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
        return jsonify({"error": "The ~/RKLLAMA/models folder cannot be found."}), 500

    print(os.listdir(models_dir))
    models = [f for f in os.listdir(models_dir) if str(f).endswith(".rkllm")]
    print(models)
    return jsonify({"models": models}), 200


# Delete a model
@app.route('/rm', methods=['DELETE'])
def Rm_model():

    data = response.json
    if "model" not in data:
        return jsonify({"error": "Please specify a model."}), 400

    model_path = os.path.expanduser(f"~/RKLLAMA/models/{model}")
    if not os.path.exists(model_path):
        return jsonify({"error": f"The model: {model} cannot be found."}), 404

    os.system(f"rm {model_path}")

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

            local_filename = os.path.join(os.path.expanduser("~/RKLLAMA/models"), file)
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)

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

                yield "Download complete.\n"

            except Exception as download_error:
                # Remove the file if an error occurs during download
                if os.path.exists(local_filename):
                    os.remove(local_filename)
                yield f"Error during download: {str(download_error)}\n"
                return

        except Exception as e:
            yield f"Error: {str(e)}\n"

    return Response(generate_progress(), content_type='text/plain')

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

    verrou.acquire()
    return Request(modele_rkllm)

# Default route
@app.route('/', methods=['GET'])
def default_route():
    return jsonify({"message": "Welcome to RKLLama !", "github": "https://github.com/notpunhnox/rk-llama"}), 200

# Launch function
def main():
    # define the arguments for launch function
    parser = argparse.ArgumentParser(description="RKLLM server initialization with configurable options.")
    parser.add_argument('--target_platform', type=str, help="Target platform: rk3588/rk3576.")
    args = parser.parse_args()

    if not args.target_platform:
        print_color("Error argument not found: --target_platform")
    else:
        if args.target_platform not in ["rk3588", "rk3576"]:
            print_color("Error : Invalid target platform. Please enter rk3588 or rk3576.", "red")
            sys.exit(1)
        print_color(f"Setting the frequency for the {args.target_platform} platform...", "cyan")
        library_path = os.path.expanduser(f"~/RKLLAMA/lib/fix_freq_{args.target_platform}.sh")
        commande = f"sudo bash {library_path}"
        subprocess.run(commande, shell=True)

    # Define ressources limite
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    print_color("Start the API at http://localhost:8080", "blue")
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)


# Launch program
if __name__ == "__main__":
    main()