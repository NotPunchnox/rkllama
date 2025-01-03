# RKLLama: LLM Server and Client for Rockchip 3588/3576

French version: [click](./documentation/french.md)

## Overview
A server to run and interact with LLM models optimized for Rockchip RK3588(S) and RK3576 platforms. The difference from other software of this type like [Ollama](https://ollama.com) or [Llama.cpp](https://github.com/ggerganov/llama.cpp) is that RKLLama allows models to run on the NPU.

* Version `Lib rkllm-runtime`: V1.1.4.
* Tested on an `Orange Pi 5 Pro (16GB RAM)`.

## File Structure
- **`./models`**: Place your .rkllm models here.
- **`./lib`**: C++ `rkllm` library used for inference and `fix_freqence_platform`.
- **`./app.py`**: API Rest server.
- **`./client.py`**: Client to interact with the server.

## Supported Python Versions:
- Python 3.8
- Python 3.9

## Tested Hardware and Environment
- **Hardware**: Orange Pi 5 Pro: (Rockchip RK3588S, NPU 6 TOPS).
- **OS**: Ubuntu 24.04 arm64.

## Main Features
- **Running models on NPU.**
- **Listing available models.**
- **Dynamic loading and unloading of models.**
- **Inference requests.**
- **Streaming and non-streaming modes.**
- **Message history.**

## Documentation

- Client   : [Installation guide](#installation).
- API REST : [English documentation](./documentation/api/english.md)
- API REST : [French documentation](./documentation/api/french.md)

## Installation
1. Download RKLLama:
```bash
git clone https://github.com/notpunchnox/rkllama
cd rkllama
```

2. Install RKLLama
```bash
chmod +x setup.sh
sudo ./setup.sh
```
**Output:**
![Image](./documentation/ressources/setup.png)

## Add Model (file.rkllm)

1. Download .rkllm models from [HuggingFace](https://huggingface.co), or convert your GGUF models to RKLLM (Software soon available on [my GitHub](https://github.com/notpunchnox))

2. Go to the `~/RKLLAMA` directory and place your files there
    ```bash
    cd ~/RKLLAMA/
    ```

## Usage

### Run Server
*Virtualization with `conda` is started automatically, as well as the NPU frequency setting.*
1. Start the server
```bash
rkllama serve
```
**Output:**
![Image](./documentation/ressources/server.png)


### Run Client
1. Command to start the client
```bash
rkllama
```
or 
```bash
rkllama help
```
**Output:**
![Image](./documentation/ressources/commands.png)

2. See the available models
```bash
rkllama list
```
**Output:**
![Image](./documentation/ressources/list.png)

3. Run a model
```bash
rkllama run <model_name>
```
**Output:**
![Image](./documentation/ressources/launch_chat.png)

Then start chatting
![Image](./documentation/ressources/chat.png)

## Uninstall

1. Go to the `~/RKLLAMA/` folder
    ```bash
    cd ~/RKLLAMA/
    cp ./uninstall.sh ../
    cd ../ && chmod +x ./uninstall.sh && ./uninstall.sh
    ```

2. If you don't have the `uninstall.sh` file:
    ```bash
    wget https://raw.githubusercontent.com/NotPunchnox/rkllama/refs/heads/main/uninstall.sh
    chmod +x ./uninstall.sh
    ./uninstall.sh
    ```

**Output:**
![Image](./documentation/ressources/uninstall.png)


---

## Upcoming Features
- Ability to pull models
- Add multimodal models
- Add embedding models
- `GGUF to RKLLM` conversion software

---

System Monitor:


---

## Author:
[notpunchnox](https://github.com/notpunchnox/rkllama)