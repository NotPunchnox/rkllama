# RKLLAMA - Ollama API Compatibility

## Overview

RKLLAMA now implements an Ollama-compatible API, providing an interface that matches key Ollama endpoints. This enables integration with many tools and scripts designed for Ollama's API, particularly those using the chat and generate functionality.

## Supported Endpoints

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/api/tags` | GET | List available models | ✅ |
| `/api/version` | GET | Get API version (Dummy version to fix some apps) | ✅ |
| `/api/show` | POST | Show model information | ✅ |
| `/api/create` | POST | Create model from Modelfile | ⚠️ Basic implementation |
| `/api/pull` | POST | Pull a model | ⚠️ Basic implementation |
| `/api/delete` | DELETE | Delete a model | ✅ |
| `/api/generate` | POST | Generate a completion | ✅ |
| `/api/chat` | POST | Generate a chat completion | ✅ **With Tool Calling** |
| `/api/embeddings` | POST | Generate embeddings | ❌ Not implemented |
| `/api/embed` | POST | Generate embeddings | ❌ Not implemented |

## Usage Examples

### Chat Completion (`/api/chat`)

This endpoint uses a chat-style format with message roles and is best for multi-turn conversations:

```bash
curl -X POST http://localhost:8080/api/chat -d '{
  "model": "qwen2.5:3b",
  "messages": [
    {"role": "user", "content": "Hello, how are you today?"}
  ]
}'
```

### Generate Completion (`/api/generate`)

This endpoint is used for single-turn completions based on a prompt:

```bash
curl -X POST http://localhost:8080/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "Write a poem about AI"
}'
```

### List Models

```bash
curl http://localhost:8080/api/tags
```

## Tool/Function Calling Support

✅ **Fully Compatible** - RKLLama implements complete tool calling functionality matching Ollama's API specification.

### Features
- **Multiple Detection Methods**: Works with Qwen's `<tool_call>` tags and generic JSON formats
- **Streaming Support**: Tool calls work in both streaming and non-streaming modes  
- **Format Normalization**: Automatically handles different parameter formats (`parameters` vs `arguments`)
- **Model Agnostic**: Works with any LLM that can output proper JSON

### Tool Call Example

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b",
    "messages": [
      {"role": "user", "content": "Get the weather in Tokyo"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "City name"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

### Response Format

Tool calls return the standard Ollama format:

```json
{
  "model": "qwen2.5:3b",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "get_weather",
          "arguments": {
            "location": "Tokyo"
          }
        }
      }
    ]
  },
  "done_reason": "tool_calls",
  "done": true
}
```

### Model Compatibility for Tools

| Model Type | Tool Format | Support Level |
|------------|-------------|---------------|
| **Qwen 2.5+** | `<tool_call></tool_call>` | ✅ Native (Recommended) |
| **Llama 3.2+** | Generic JSON | ✅ Full Support |
| **Other Models** | JSON Fallback | ✅ Compatible |

**Recommendation**: Use Qwen models for best tool calling performance.

For complete tool calling documentation, see the [Tool Calling Guide](./tools.md).

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

- Only core endpoints (`/api/chat` and `/api/generate`) are fully implemented
- Pull and Create endpoints have basic implementations
- Embeddings API is not currently implemented
- Some advanced Ollama formatting features are not yet supported
- Not all Ollama clients have been tested for compatibility
- OpenAI API compatibility is still in development

## Troubleshooting Tips

- Check if server is running and accessible
- Verify models are properly loaded before making requests
- Try non-streaming requests (`"stream": false`) if streaming has issues
- Ensure your Modelfile includes required `HUGGINGFACE_PATH` parameter
- For advanced troubleshooting, enable debug mode with `--debug` flag
