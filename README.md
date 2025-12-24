# RKLLama: LLM Server and Client for Rockchip 3588/3576

### [Version: 0.0.54](#New-Version)

Video demo ( version 0.0.1 ):

[![Watch on YouTube](https://img.youtube.com/vi/Kj8U1OGqGPc/0.jpg)](https://www.youtube.com/watch?v=Kj8U1OGqGPc)

##  Branches

-  [Without Miniconda](https://github.com/NotPunchnox/rkllama/tree/Without-miniconda): This version runs without Miniconda.
-  [Rkllama Docker](https://github.com/NotPunchnox/rkllama/tree/Rkllama-Docker): A fully isolated version running in a Docker container.
-  [Support All Models](https://github.com/NotPunchnox/rkllama/tree/Support-All-models): This branch ensures all models are tested before being merged into the main branch.
-  [Docker Package](https://github.com/NotPunchnox/rkllama/pkgs/container/rkllama)


## Overview
A server to run and interact with LLM models optimized for Rockchip RK3588(S) and RK3576 platforms. The difference from other software of this type like [Ollama](https://ollama.com) or [Llama.cpp](https://github.com/ggerganov/llama.cpp) is that RKLLama allows models to run on the NPU.

* Version `Lib rkllm-runtime`: V 1.2.3.
* Version `Lib rknn-runtime`: V 2.3.2.

## File Structure
- **`./models`**: contains your rkllm models (wihh their rknn models if multimodal) .
- **`./lib`**: C++ `rkllm` and `rklnn` library used for inference and `fix_freqence_platform`.
- **`./app.py`**: API Rest server.
- **`./client.py`**: Client to interact with the server.

## Supported Python Versions:
- Python 3.9 to 3.12

## Tested Hardware and Environment
- **Hardware**: Orange Pi 5 Pro: (Rockchip RK3588S, NPU 6 TOPS), 16GB RAM.
- **Hardware**: Orange Pi 5 Plus: (Rockchip RK3588S, NPU 6 TOPS), 16GB RAM.
- **Hardware**: Orange Pi 5 Max: (Rockchip RK3588S, NPU 6 TOPS), 16GB RAM.
- **OS**: [Ubuntu 24.04 arm64.](https://joshua-riek.github.io/ubuntu-rockchip-download/)
- **OS**: Armbian Linux 6.1.99-vendor-rk35xx (Debian stable bookworm), v25.2.2.

## Main Features
- **Running models on NPU.**
- **Ollama API compatibility** - Support for:
   * `/api/chat`
   * `/api/generate`
   * `/api/ps`
   * `/api/tags`
   * `/api/embed` (and legacy `/api/embeddings`)
   * `/api/version` 
   * `/api/pull` 
- **Partial OpenAI API compatibility** - Support for:
   * `/v1/completions`
   * `/v1/chat/completions`
   * `/v1/embeddings`
   * `/v1/images/generations`
   * `/v1/audio/speech`
   * `/v1/audio/transcriptions`
- **Tool/Function Calling** - Complete support for tool calls with multiple LLM formats (Qwen, Llama 3.2+, others).
- **Pull models directly from Huggingface.**
- **Include a API REST with documentation.**
- **Listing available models.**
- **Multiples RKLLM models running in memory simultaniusly (parallels executions between distintct models in stream mode, FIFO if non stream)**
- **Dynamic loading and unloading of models:**
    * Load the model after new request (if not in memory already)
    * Unload when model expires after inactivity (default 30 min)
    * Unload the oldest model in memory if new model is required to be loaded and there is not memory available in the server
    * 
- **Inference requests with streaming and non-streaming modes.**
- **Message history.**
- **Simplified custom model naming** - Use models with familiar names like "qwen2.5:3b".
- **CPU Model Auto-detection** - Automatic detection of RK3588 or RK3576 platform.
- **Optional Debug Mode** - Detailed debugging with `--debug` flag.
- **Multimodal Suport** - Use Qwen2VL/Qwen2.5VL/Qwen3VL/MiniCPMV4/MiniCPMV4.5/InternVL3.5 vision models to ask questions about images (base64, local file or URL image address). More than one image in the same request is allowed.
- **Image Generation** - Generate images with OpenAI Image generation endpoint using model LCM Stable Diffusion 1.5 RKNN models.
- **Text to Speech (TTS)** - Generate speech with OpenAI Audio Speech endpoint using models for Piper TTS running encoder with ONNX and decoder with RKNN.
- **Speech to Text (STT)** - Generate transcriptions with OpenAI Audio Transcriptions endpoint using models for omniASR-CTC running the model with RKNN.


## Documentation

* French version: [click](./documentation/french.md)

- Client   : [Installation guide](#installation).
- API REST : [English documentation](./documentation/api/english.md)
- API REST : [French documentation](./documentation/api/french.md)
- Ollama API: [Compatibility guide](./documentation/api/ollama-compatibility.md)
- Model Naming: [Naming convention](./documentation/api/model_naming.md)
- Tool Calling: [Tool/Function calling guide](./documentation/api/tools.md)

## Installation

###  Standard Installation (recommended create a virtual environment like: conda, uv, venv)

1. **Clone the repository:**

```bash
git clone https://github.com/notpunchnox/rkllama
cd rkllama
```

2.  **Install RKLLama:**

```bash
python -m pip install .
```

**Output:**
![Image](./documentation/ressources/setup.png)


### Docker Installation

Pull the RKLLama Docker image:

```bash
docker pull ghcr.io/notpunchnox/rkllama:main
```
run server
```bash
docker run -it --privileged -p 8080:8080 -v <local_models_dir>:/opt/rkllama/models ghcr.io/notpunchnox/rkllama:main 
```

*Set up by: [ichlaffterlalu](https://github.com/ichlaffterlalu)*

#### Docker Compose

Docker Compose facilities much of the extra flags declaration such as volumes:

```bash
docker compose up --detach --remove-orphans
```

## Usage

### Run Server
*Virtualization with `conda` is started automatically, as well as the NPU frequency setting.*
1. Start the server
```bash
rkllama_server --models <models_dir>
```

To enable debug mode:
```bash
rkllama_server --debug --models <models_dir>
```

**Output:**
![Image](./documentation/ressources/server.png)


### Run Client
1. Command to start the client
```bash
rkllama_client
```
or 
```bash
rkllama_client help
```

**Output:**
![Image](./documentation/ressources/commands.png)

2. See the available models
```bash
rkllama_client list
```
**Output:**
![Image](./documentation/ressources/list.png)


3. Run a model
```bash
rkllama_client run <model_name>
```
**Output:**
![Image](./documentation/ressources/launch_chat.png)

Then start chatting *( **verbose mode**: display formatted history and statistics )*
![Image](./documentation/ressources/chat.gif)

## Tool Calling Quick Start

RKLLama supports advanced tool/function calling for enhanced AI interactions:

```bash
# Example: Weather tool call
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

**Features:**
- 🔧 **Multiple model support** (Qwen, Llama 3.2+, others)
- 🌊 **Streaming & non-streaming** modes
- 🎯 **Robust JSON parsing** with fallback methods
- 🔄 **Auto format normalization**
- 📋 **Multiple tools** in single request

For complete documentation: [Tool Calling Guide](./documentation/api/tools.md)

## Adding a Model (`file.rkllm`)

### **Using the `rkllama pull` Command**
You can download and install a model from the Hugging Face platform with the following command:

```bash
rkllama_client pull username/repo_id/model_file.rkllm/custom_model_name
```

Alternatively, you can run the command interactively:

```bash
rkllama_client pull
Repo ID ( example: punchnox/Tinnyllama-1.1B-rk3588-rkllm-1.1.4): <your response>
File ( example: TinyLlama-1.1B-Chat-v1.0-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm): <your response>
Custom Model Name ( example: tinyllama-chat:1.1b ): <your response>
```

This will automatically download the specified model file and prepare it for use with RKLLAMA.

*Example with Qwen2.5 3b from [c01zaut](https://huggingface.co/c01zaut): https://huggingface.co/c01zaut/Qwen2.5-3B-Instruct-RK3588-1.1.4*
![Image](./documentation/ressources/pull.png)

---

### **Manual Installation**
1. **Download the Model**
   - Download `.rkllm` models directly from [Hugging Face](https://huggingface.co).
   - Alternatively, convert your GGUF models into `.rkllm` format (conversion tool coming soon on [my GitHub](https://github.com/notpunchnox)).

2. **Place the Model**
   - Create the `models` directory on your system.
   - Make a new subdirectory with model name.
   - Place the `.rkllm` files in this directory.
   - Create `Modelfile` and add this :

   ```env
    FROM="file.rkllm"
    HUGGINGFACE_PATH="huggingface_repository"
    SYSTEM="Your system prompt"
    TEMPERATURE=1.0
    TOKENIZER="path-to-tokenizer"
    ```

   Example directory structure:
   ```
   ~/RKLLAMA/models/
       └── TinyLlama-1.1B-Chat-v1.0
           |── Modelfile
           └── TinyLlama-1.1B-Chat-v1.0.rkllm
   ```

   *You must provide a link to a HuggingFace repository to retrieve the tokenizer and chattemplate. An internet connection is required for the tokenizer initialization (only once), and you can use a repository different from that of the model as long as the tokenizer is compatible and the chattemplate meets your needs.*


### **For Multimodal Encoder Model (.rknn) Installation**
1. **Download the encoder model .rknn**
   - Download `.rknn` models directly from [Hugging Face](https://huggingface.co).
   - Alternatively, convert your ONNX models into `.rknn` format.
   - Place the `.rknn` model inside the `models` directory. RKLLama detected the encoder model present in the directory.
   - Include manually the following properties in the `Modelfile` according to the conversion properties used for the conversion of the vision encoder `.rknn`:
   ```env
    IMAGE_WIDTH=448
    IMAGE_HEIGHT=
    N_IMAGE_TOKENS=
    IMG_START=
    IMG_END=
    IMG_CONTENT=

    # For example, for Qwen2VL/Qwen2.5VL:

    IMAGE_WIDTH=392
    IMAGE_HEIGHT=392
    N_IMAGE_TOKENS=196
    IMG_START=<|vision_start|>
    IMG_END=<|vision_end|>
    IMG_CONTENT=<|image_pad|>

    # For example, for MiniCPMV4:

    IMAGE_WIDTH=448
    IMAGE_HEIGHT=448
    N_IMAGE_TOKENS=64
    IMG_START=<image>
    IMG_END=</image>
    IMG_CONTENT=<unk>
   ```

Example directory structure for multimodal:
   ```
   ~/RKLLAMA/models/
       └── qwen2-vision\:2b
           |── Modelfile
           └── Qwen2-VL-2B-Instruct.rkllm
           └── Qwen2-VL-2B-Instruct.rknn
   ```

### **For Image Generation Installation**
1. In a temporary folder, clone the repository https://huggingface.co/danielferr85/lcm-sd-1.5-rknn-2.3.2-rk3588 or https://huggingface.co/danielferr85/lcm-ssd-1b-rknn-2.3.2-rk3588 from Hugging Face for the desired SD model
2. Execute the ONNX to RKNN convertion of the models for your needs **WITH RKNN TOOLKIT LIBRARY VERSION 2.3.2**. For example:
   
   For LCM SD 1.5
   ```
    python convert-onnx-to-rknn.py --model-dir <directory_download_model> --resolutions 512x512 --components "text_encoder,unet,vae_decoder" --target_platform rk3588
   ```
   
   or

   For LCM SSD1B
   ```
    python convert-onnx-to-rknn.py --model-dir <directory_download_model> --resolutions 1024x1024 --components "text_encoder,text_encoder_2,unet,vae_decoder" --target_platform rk3588
   ```
3. Create a folder inside the models directory in RKLLAMA for the Stable Diffusion RKNN models, For example: **lcm-stable-diffusion** or **lcm-segmind-stable-diffusion** 
4. Copy the folders: "scheduler, text_encoder, text_encoder_2 (for SSD1B only), unet, vae_decoder"  from the cloned repo to the new directory model created in RKLLMA. Just copy the *.json and *.rknn files. 
5. The structure of the model **MUST** be like this:

   For LCM SD 1.5
   ```
   ~/RKLLAMA/models/
       └── lcm-stable-diffusion
           |── scheduler
              |── scheduler_config.json
           └── text_encoder
              |── config.json
              |── model.rknn
           └── unet
              |── config.json
              |── model.rknn
           └── vae_decoder
              |── config.json
              |── model.rknn
           
   ```

   or

   For LCM SSD1B
   ```
   ~/RKLLAMA/models/
       └── lcm-segmind-stable-diffusion
           |── scheduler
              |── scheduler_config.json
           └── text_encoder
              |── config.json
              |── model.rknn
           └── text_encoder_2
              |── config.json
              |── model.rknn
           └── unet
              |── config.json
              |── model.rknn
           └── vae_decoder
              |── config.json
              |── model.rknn
           
   ```


6. Done! You are ready to test the OpenAI endpoint /v1/images/generations to generate images. You can add it to OpenWebUI in the Image Generation section.
7. Available converted models for RK3588 and RKNN 2.3.2 at: https://huggingface.co/danielferr85/lcm-sd-1.5-rknn-2.3.2-rk3588 (only 512x512 resolutions) and https://huggingface.co/danielferr85/lcm-ssd-1b-rknn-2.3.2-rk3588 (only 1024x1024 resolutions)



### **For Speech Generation (TTS) Installation**
1. Download a voice from https://huggingface.co/danielferr85/piper-checkpoints-rknn from Hugging Face. (You can convert new ones, see below)
2. Create a folder inside the models directory in RKLLAMA for the piper Audio model, For example: **es_AR-daniela-high** 
3. Copy the encoder (.onnx), decoder (.rknn) and config (.json) file from the choosed voice to the new directory model created in RKLLMA.
4. The structure of the model **MUST** be like this:

   ```
   ~/RKLLAMA/models/
       └── es_AR-daniela-high
           |── encoder.onnx
           └── decoder.rknn
           └── config.json
          
   ```

5. Done! You are ready to test the OpenAI endpoint /v1/audio/speech to generate audio. You can add it to OpenWebUI in the Audio section for TTS.

**IMPORTANT**
- You must convert only your decoder (.rknn) for your specific platform (for example rk3588).
- The encoder can have any name but must ended with extension .onnx
- The decoder can have any name but must ended with extension .rknn
- The config of the model can have any name but must ended with extension .json
- You must use rknn-toolkit 2.3.2 for RKNN conversion because is the one used by RKLLAMA
- Always **CHECK THE LICENSE** of the voice that you are going to use.
- In OpenAI request, the argument **model** is the name of the model folder that you create and the argument **voice** is the speaker of the voice if the voice is multispeaker (For example 'F' (Female) or 'M' (Male). Check the config of the model). If the model is monospeaker, then voice can be skipped.
- You can convert any Piper TTS model (or create one, or finetuning one) creating first the encoder and decoder in ONNX format and then converting the decoder in RKNN:
   1. Clone this repo and branch **streaming**: https://github.com/mush42/piper/tree/streaming
   2. Create a virtual environemnt and install that repo
   3. Clone the repository https://huggingface.co/danielferr85/piper-checkpoints-rknn from Hugging Face
   4. Download the entire folder of the voice to convert from dataset: https://huggingface.co/datasets/rhasspy/piper-checkpoints and put it inside the repo **piper-checkpoints-rknn** (the structure must be for example: es/es_MX/ald/medium/). You can also use the script  **download_models.py** from download automatically the model you want.
   5. Execute the script export_encoder_decoder.py to export the encoder and decoder IN ONNX format.
   6. Execute the script export_rknn.py to export the decoder in RKNN format (you must uhave installed the rknn-toolkit version 2.3.2).


### **For Transcriptions Generation (STT) Installation**
1. Download a model from https://huggingface.co/danielferr85/omniASR-ctc-rknn from Hugging Face.
2. Create a folder inside the models directory in RKLLAMA for the model, For example: **omniasr-ctc:300m** 
3. Copy the model (.rknn) and vocabulary (.txt) file from the choosed model to the new directory model created in RKLLMA.
4. The structure of the model **MUST** be like this:

   ```
   ~/RKLLAMA/models/
       └── omniasr-ctc:300m
           └── model.rknn
           └── vocab.txt
          
   ```

5. Done! You are ready to test the OpenAI endpoint /v1/audio/transcriptions to generate transcriptions. You can add it to OpenWebUI in the Audio section for STT.

**IMPORTANT**
- The model can have any name but must ended with extension .rknn
- The vocabulary of the model can have any name but must ended with extension .txt 
- You must use rknn-toolkit 2.3.2 for RKNN conversion because is the one used by RKLLAMA


## Configuration

RKLLAMA uses a flexible configuration system that loads settings from multiple sources in a priority order:

See the [Configuration Documentation](documentation/configuration.md) for complete details.

## Uninstall

1. Remove the pyhton package rkllama
    ```
    pip uninstall rkllama
    ```
**Output:**
![Image](./documentation/ressources/uninstall.png)


---

# New-Version

**Ollama API Compatibility**: RKLLAMA now implements key Ollama API endpoints, with primary focus on `/api/chat` and `/api/generate`, allowing integration with many Ollama clients. Additional endpoints are in various stages of implementation.

**Enhanced Model Naming**: Simplified model naming convention allows using models with familiar names like "qwen2.5:3b" or "llama3-instruct:8b" while handling the full file paths internally.

**Improved Performance and Reliability**: Enhanced streaming responses with better handling of completion signals and optimized token processing.

**CPU Auto-detection**: Automatic detection of RK3588 or RK3576 platform with fallback to interactive selection.

**Debug Mode**: Optional debugging tools with detailed logs that can be enabled with the `--debug` flag.

**Simplified Model Management**: 
- Delete models with one command using the simplified name
- Pull models directly from Hugging Face with automatic Modelfile creation
- Custom model configurations through Modelfiles
- Smart collision handling for models with similar names

If you have already downloaded models and do not wish to reinstall everything, please follow this guide: [Rebuild Architecture](./documentation/Guide/en/Rebuild-arch.md)

---

## Upcoming Features
- Add RKNN for onnx models (TTS, image classification/segmentation...)
- `GGUF/HF to RKLLM` conversion software

---

System Monitor:

---

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=notpunchnox/rkllama)

---

##  Author

*  [**NotPunchnox**](https://github.com/notpunchnox/rkllama)

##  Contributors

*  [**ichlaffterlalu**](https://github.com/ichlaffterlalu): Contributed with a pull request for [Docker-Rkllama](https://github.com/NotPunchnox/rkllama/tree/Rkllama-Docker) and fixed multiple errors.
*  [**TomJacobsUK**](https://github.com/TomJacobsUK): Contributed with pull requests for Ollama API compatibility and model naming improvements, and fixed CPU detection errors.
*  [**Yoann Vanitou**](https://github.com/yvanitou): Contributed with Docker implementation improvements and fixed merge conflicts.
*  [**Daniel Ferreira**](https://github.com/danielferr85): Contributed with Tools Support, OpenAI API compatibility and multiload RKLLM models in memory. Also improvements and fixes. Multimodal support implementation.
