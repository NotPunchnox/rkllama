# CLI Reference

Complete reference for all RKLlama Converter commands and options.

## Global Options

These options work with any command:

| Option | Description |
|--------|-------------|
| `--version, -v` | Show version and exit |
| `--help` | Show help message |

## Commands

### convert

Convert a HuggingFace model to RKLLM format.

```bash
rkllama-convert convert <MODEL_ID> [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `MODEL_ID` | Yes | HuggingFace model ID (e.g., `Qwen/Qwen2.5-7B-Instruct`) |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | PATH | `./output` | Output directory for converted model |
| `-q, --quant` | ENUM | `Q4_0` | Quantization format |
| `-c, --context` | INT | `4096` | Maximum context length in tokens |
| `-d, --dtype` | STR | `float16` | Model data type (`float16` or `float32`) |
| `--device` | ENUM | `auto` | Compute device (`cuda`, `rocm`, `cpu`, `auto`) |
| `-t, --token` | STR | None | HuggingFace API token |
| `-s, --system` | STR | None | System prompt for Modelfile |
| `--temp` | FLOAT | `0.7` | Generation temperature |
| `--top-k` | INT | `40` | Top-K sampling parameter |
| `--top-p` | FLOAT | `0.9` | Nucleus sampling parameter |
| `--thinking/--no-thinking` | BOOL | `False` | Enable reasoning mode |

#### Quantization Formats

| Format | Description | Size | Quality |
|--------|-------------|------|---------|
| `Q4_0` | 4-bit weights, 16-bit activations | ~25% of FP16 | Good |
| `Q4_K_M` | 4-bit grouped (g=128) | ~25% of FP16 | Better |
| `Q8_0` | 8-bit weights and activations | ~50% of FP16 | Excellent |
| `Q8_K_M` | 8-bit grouped (g=512) | ~50% of FP16 | Best |

#### Examples

Basic conversion:
```bash
rkllama-convert convert Qwen/Qwen2.5-7B-Instruct
```

With custom output and quantization:
```bash
rkllama-convert convert Qwen/Qwen2.5-7B-Instruct \
    -o ./models/qwen \
    -q Q8_0
```

Full options:
```bash
rkllama-convert convert meta-llama/Llama-3.2-3B-Instruct \
    -o ./models/llama \
    -q Q4_0 \
    -c 8192 \
    --device cuda \
    -t $HF_TOKEN \
    -s "You are a helpful coding assistant." \
    --temp 0.8 \
    --top-k 50 \
    --thinking
```

---

### info

Show information about a HuggingFace model before conversion.

```bash
rkllama-convert info <MODEL_ID> [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `MODEL_ID` | Yes | HuggingFace model ID to inspect |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-t, --token` | STR | None | HuggingFace API token for private models |

#### Example

```bash
rkllama-convert info Qwen/Qwen2.5-7B-Instruct
```

Output:
```
┌─────────────────────────────────────────────┐
│           Model: Qwen/Qwen2.5-7B-Instruct   │
├──────────────┬──────────────────────────────┤
│ Property     │ Value                        │
├──────────────┼──────────────────────────────┤
│ Model ID     │ Qwen/Qwen2.5-7B-Instruct     │
│ Author       │ Qwen                         │
│ Downloads    │ 1,234,567                    │
│ Likes        │ 5,432                        │
│ License      │ Apache-2.0                   │
│ Pipeline Tag │ text-generation              │
│ Parameters   │ 7.62B                        │
└──────────────┴──────────────────────────────┘

Recommended settings:
  Quantization: Q4_0 (smallest) or Q8_0 (highest quality)
  Context: 4096 (default) or check model card for max
```

---

### list-quants

Display available quantization formats and their characteristics.

```bash
rkllama-convert list-quants
```

Output:
```
┌───────────────────────────────────────────────────────────────────┐
│                     Quantization Formats                          │
├─────────┬───────────┬──────┬─────────────┬───────────┬───────────┤
│ Format  │ RKLLM Type│ Bits │ Size vs FP16│ Quality   │ Status    │
├─────────┼───────────┼──────┼─────────────┼───────────┼───────────┤
│ Q4_0    │ w4a16     │ 4    │ ~25%        │ Good      │ Supported │
│ Q4_K_M  │ w4a16_g128│ 4    │ ~25%        │ Better    │ Partial   │
│ Q8_0    │ w8a8      │ 8    │ ~50%        │ Excellent │ Supported │
│ Q8_K_M  │ w8a8_g512 │ 8    │ ~50%        │ Best      │ Partial   │
└─────────┴───────────┴──────┴─────────────┴───────────┴───────────┘

Note: K_M variants use grouped quantization for better accuracy.
w4a16 = 4-bit weights, 16-bit activations
w8a8 = 8-bit weights, 8-bit activations
```

---

### list-devices

List available compute devices for model conversion.

```bash
rkllama-convert list-devices
```

Output (NVIDIA system):
```
Available Devices

Auto-detected: cuda

CPU: 16 cores
CUDA: NVIDIA GeForce RTX 3080
  Memory: 10.0 GB
  Compute Capability: 8.6
```

Output (AMD system):
```
Available Devices

Auto-detected: rocm

CPU: 12 cores
ROCM: AMD Radeon RX 7900 XTX
  Memory: 24.0 GB
  HIP Version: 6.2.0
```

---

## Environment Variables

| Variable | Description | Used By |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | `convert`, `info` |
| `CUDA_VISIBLE_DEVICES` | GPU selection (NVIDIA) | `convert` |
| `HIP_VISIBLE_DEVICES` | GPU selection (AMD) | `convert` |

Example:
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 rkllama-convert convert model/id

# Set token via environment
export HF_TOKEN=hf_xxxxx
rkllama-convert convert private/model
```

---

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Error during conversion |
| 2 | Invalid arguments |

---

## See Also

- [Installation Guide](installation.md)
- [Quantization Guide](quantization.md)
- [Modelfile Reference](modelfile.md)
