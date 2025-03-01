# Model Naming in RKLLAMA

RKLLAMA supports both the full model names used in the file system and simplified model names in the Ollama API style.

## Simplified Model Names

When using the Ollama-compatible API, you'll see model names in a simplified format like:
- `qwen:3b` (for Qwen models)
- `mistral:7b` (for Mistral models)
- `llama2:7b` (for Llama 2 models)
- `tinyllama:1.1b` (for TinyLlama models)

This is the format you should use when making API requests to endpoints like `/api/chat` and `/api/generate`.

## Full Model Names

Under the hood, RKLLAMA uses the full model names for file paths and model loading:
- `Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-1.0`
- `Llama-2-7B-Chat-rk3588-w4a16-g64-opt-0-hybrid-ratio-0.4`

## Name Conversion

RKLLAMA automatically handles the conversion between simplified and full names. You can use either format when making API requests, and RKLLAMA will always respond using the simplified format for compatibility with Ollama clients.

## Example

### Request with full name:
