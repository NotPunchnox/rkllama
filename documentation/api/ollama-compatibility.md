# RKLLAMA - Ollama API Compatibility

## Overview

RKLLAMA now implements an Ollama-compatible API, providing a drop-in interface that matches Ollama's endpoint structure. This enables integration with tools and scripts designed for Ollama's API.

## Supported Endpoints

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/api/tags` | GET | List available models | ✅ |
| `/api/show` | POST | Show model information | ✅ |
| `/api/create` | POST | Create model from Modelfile | ✅ |
| `/api/pull` | POST | Pull a model | ✅ |
| `/api/delete` | DELETE | Delete a model | ✅ |
| `/api/generate` | POST | Generate a completion | ✅ |
| `/api/chat` | POST | Generate a chat completion | ✅ |
| `/api/embeddings` | POST | Generate embeddings | ❌ |

## Usage Examples

### Chat Completion (`/api/chat`)

This endpoint uses a chat-style format with message roles and is best for multi-turn conversations:

```bash
curl -X POST http://localhost:8080/api/chat -d '{
  "model": "TinyLlama-1.1B-Chat-v1.0",
  "messages": [
    {"role": "user", "content": "Hello, how are you today?"}
  ]
}'
```

### Generate Completion (`/api/generate`)

This endpoint is used for single-turn completions based on a prompt:

```bash
curl -X POST http://localhost:8080/api/generate -d '{
  "model": "TinyLlama-1.1B-Chat-v1.0",
  "prompt": "Write a poem about AI"
}'
```

### List Models

```bash
curl http://localhost:8080/api/tags
```

## Platform Auto-detection

RKLLAMA automatically detects whether you're using an RK3588 or RK3576 platform. If detection fails, you'll be prompted to select your CPU model:

```
CPU model not detected automatically.
Please select your CPU model:
1) rk3588
2) rk3576
Enter selection (1-2):
```

## Important Differences from Ollama

1. **Model Format**: RKLLAMA uses `.rkllm` files optimized for Rockchip NPUs, not Ollama's format
2. **Modelfile Requirements**: RKLLAMA Modelfiles require a `HUGGINGFACE_PATH` parameter
3. **NPU Acceleration**: RKLLAMA runs on NPU hardware rather than CPU/GPU
4. **External Tokenizers**: RKLLAMA uses HuggingFace tokenizers fetched at runtime

## Optional Debugging Mode

For troubleshooting purposes, RKLLAMA includes an optional debug mode that can be enabled when starting the server:

```bash
# Using the rkllama command
rkllama serve --debug

# Or directly with the server script
bash ~/RKLLAMA/server.sh --debug
```

When debug mode is enabled:
- Detailed logs are written to `~/RKLLAMA/rkllama_debug.log`
- Additional diagnostic information is displayed in the console
- A special `/api/debug` endpoint becomes available (advanced users only)

Debug mode is entirely optional and not needed for normal operation.

## Stream Reliability Improvements

Recent updates have significantly improved streaming reliability:
- Enhanced "done" signaling for proper stream completion
- Fixed token tracking across streaming sessions

## Limitations

- Embeddings API is not currently implemented
- Some advanced Ollama formatting features are not supported
- Client compatibility has not been extensively tested
- Full OpenAI compatibility is still in development

## Troubleshooting Tips

- Check if server is running and accessible
- Verify models are properly loaded before making requests
- Try non-streaming requests (`"stream": false`) if streaming has issues
- Ensure your Modelfile includes required `HUGGINGFACE_PATH` parameter
- For advanced troubleshooting, enable debug mode with `--debug` flag
