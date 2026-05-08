# RKLLAMA Changelog


## Version 0.0.41 (Current)

### New Commands
- `rkllama info param`: Displays model details in the RKLLAMA CLI.

### Technical Changes
- Added temperature parameter to the RKLLM class for enhanced control over generation.
- Introduced context-length in the RKLLM class, dynamically adapting to the model's requirements.
- Enabled custom tokenizer import via the Modelfile, supporting offline deployment without an internet connection.
- **Tool/Function Calling**: Complete implementation of tool calling with multi-format support
  - Standard `<tool_call></tool_call>` tag detection (Qwen models)
  - Generic JSON tool call extraction (Llama 3.2+, others)
  - Streaming and non-streaming tool call responses
  - Ollama API compatible tool call format
  - Automatic format normalization (`parameters` → `arguments`)
  - Robust JSON parsing with multiple fallback methods
- **Enhanced Documentation**: Comprehensive tool calling documentation


## Version 0.0.4

### Major Features
- **Ollama API Compatibility**: Added support for the Ollama API interface, allowing RKLLAMA to work with Ollama clients and tools.
- **Enhanced Streaming Responses**: Improved reliability of streaming responses with better handling of completion signals.
- **Optional Debug Mode**: Added detailed debugging tools that can be enabled with `--debug` flag.
- **CPU Model Auto-detection**: Automatic detection of RK3588 or RK3576 platform with fallback to interactive selection.

### New API Endpoints
- `/api/tags` - List all available models (Ollama-compatible)
- `/api/show` - Show model information
- `/api/create` - Create a new model from a Modelfile
- `/api/pull` - Pull a model from Hugging Face
- `/api/delete` - Delete a model
- `/api/generate` - Generate a completion for a prompt
- `/api/chat` - Generate a chat completion
- `/api/embeddings` - (Placeholder) Generate embeddings
- `/api/debug` - Diagnostic endpoint (available only in debug mode)

### Improvements
- More reliable "done" signaling for streaming responses
- Auto-detection of CPU model (RK3588 or RK3576) with fallback to user selection
- Better error handling and error messages
- Fixed threading issues in request processing
- Automatic content formatting for various response types
- Improved stream handling with token tracking
- Optional debugging mode with detailed logs

### Technical Changes
- Added new utility modules for debugging and API handling
- Improved thread management for streaming responses
- Added CPU model detection and selection
- Updated server configuration options
- Made debugging tools optional through environment variable and command line flag

## Version 0.0.3

### Major Features
- **Extended Compatibility**: Added support for DeepSeek, Qwen, Llama, and other model types.
- **Enhanced Performance**: Inputs now tokenized before being sent to model for improved response speed.
- **Modelfile System**: Implemented a Modelfile system similar to Ollama.

### Improvements
- **Simplified Organization**: Models now organized into dedicated folders.
- **Automatic Modelfile Creation**: Modelfile auto-generated when using the pull command.
- **Tokenizer Integration**: Automatic initialization of tokenizers and chat templates from HuggingFace.

## Version 0.0.1 - 0.0.2

### Features
- **Initial Release**: Basic support for running models on Rockchip NPUs
- **REST API**: Simple API for model management and inference
- **Client Tool**: Command-line client for interacting with the server
- **Model Management**: Loading and unloading models from NPU
- **Streaming Support**: Basic streaming and non-streaming modes
