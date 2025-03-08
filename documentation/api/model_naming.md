# Model Naming in RKLLAMA

RKLLAMA supports both the full model names used in the file system and simplified model names in the Ollama API style.

## Simplified Model Names

When using the Ollama-compatible API, you'll see model names in a simplified format:

### Basic Pattern
`family[-variant]:size` where:
- `family` is the model architecture (e.g., qwen2.5, mistral, llama3)
- `variant` (optional) is the model variant (e.g., coder, math, instruct)
- `size` is the parameter size (e.g., 3b, 7b)

The family and any variants appear before the colon, while only the parameter size appears after the colon.

### Examples
- `qwen2.5:3b` - Qwen 2.5 with 3 billion parameters
- `mistral:7b` - Base Mistral 7B model
- `deepseek-coder:7b` - DeepSeek Coder variant with 7 billion parameters
- `deepseek-math:7b` - DeepSeek Math variant with 7 billion parameters
- `llama2-chat:7b` - Llama 2 Chat variant with 7 billion parameters
- `phi3:medium` - Phi-3 Medium model (size is part of family name)
- `qwen2.5-coder-instruct:7b` - Qwen 2.5 Coder with Instruct capabilities, 7B parameters

Multiple variants are combined with hyphens before the colon.

## Full Model Names

Under the hood, RKLLAMA uses the full model names for file paths and model loading:
- `Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-1.0`
- `Llama-2-7B-Chat-rk3588-w4a16_g64-opt-0-hybrid-ratio-0.4`
- `deepseek-coder-7b-instruct-v1.5-rk3588-w8a8-opt-1-hybrid-ratio-0.5`
- `deepseek-math-7b-instruct-rk3588-w8a8-opt-1-hybrid-ratio-0.5`

Full names typically include:
1. Model architecture/family
2. Optional variant (coder, math, etc.)
3. Parameter size (e.g., 7B)
4. Fine-tuning type (e.g., Instruct, Chat)
5. Target platform (e.g., rk3588)
6. Quantization details (e.g., w8a8, w4a16_g64)
7. Optimization settings

## Name Conversion Logic

RKLLAMA automatically handles conversion between simplified and full names with the following rules:

1. **Model Family Detection**:
   - Recognizes common families: `llama`, `llama2`, `llama3`, `mistral`, `qwen`, `qwen2.5`, `deepseek`, `phi`, `phi2`, `phi3`, etc.
   
2. **Variant Detection**:
   - Extracts variants like `coder`, `math`, `instruct`, `chat`, `vision`, etc.
   - Multiple variants are joined with hyphens (e.g., `coder-instruct`)

3. **Parameter Size Detection**:
   - Looks for patterns like `7B`, `3b`, `1.5b` to determine model size
   - Parameter size appears after the colon in simplified names

4. **Simplified Format Construction**:
   - `family-variant:size` where variants are optional and multiple variants are hyphenated

## Using Model Names in API Requests

You can use either format when making API requests:

### Example with simplified name:
```json
{
  "model": "deepseek-coder:7b",
  "messages": [{"role": "user", "content": "Write a Python function to calculate Fibonacci numbers"}]
}
```

### Example with full name:
```json
{
  "model": "deepseek-coder-7b-instruct-v1.5-rk3588-w8a8-opt-1-hybrid-ratio-0.5",
  "messages": [{"role": "user", "content": "Write a Python function to calculate Fibonacci numbers"}]
}
```

RKLLAMA will automatically resolve either format to the correct model.

## Collision Handling

When two models would simplify to the same name (e.g., models with the same family, variants, and parameter size), RKLLAMA uses a tiered approach to create unique names:

1. **Differentiate by Quantization**:
   - If one model uses w8a8 and another uses w8a8_g128, they'll be named:
   - `qwen2.5:7b` and `qwen2.5-w8a8-g128:7b`

2. **Differentiate by Optimization Level**:
   - If models have different optimization levels, they'll be named:
   - `qwen2.5:7b-opt0` and `qwen2.5-opt1:7b`

3. **Differentiate by Hybrid Ratio**:
   - If models have different hybrid ratios, they'll be named:
   - `qwen2.5:7b-r0.5` and `qwen2.5-r1.0:7b`

4. **Numeric Suffix (Last Resort)**:
   - If all other attributes are identical, a numeric suffix is added:
   - `qwen2.5:7b` and `qwen2.5-1:7b`

This hierarchical approach maintains meaningful distinctions between similar models.

## Close Matches and Fuzzy Search

RKLLAMA also supports finding models by close matches to their names. For example:

- Requesting "qwen3b" might match "qwen2.5:3b"
- Requesting "mistral7b-instruct" might match "mistral-instruct:7b"

This fuzzy matching makes it easier to find models without knowing their exact naming.

## Listing Available Models

To see all available models with their simplified names, use the `/v1/models` endpoint:

```bash
curl http://localhost:8080/v1/models
```
