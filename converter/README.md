# RKLlama Converter

Convert HuggingFace models to RKLLM format for Rockchip NPU (RK3588/RK3576).

Uses the official [rkllm-toolkit](https://github.com/airockchip/rknn-llm) from Rockchip.

## Features

- Convert any supported HuggingFace model to RKLLM format
- Q4_0, Q4_K_M, Q8_0, Q8_K_M quantization formats
- Target platform selection (RK3588, RK3576, RK3562, RV1126B)
- NPU core configuration
- Automatic Modelfile generation
- Rich CLI with progress display
- CUDA and CPU support (ROCm untested)

## Prerequisites

- Python 3.10, 3.11, or 3.12
- Linux x86_64 (required for rkllm-toolkit)
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
cd converter

# CPU (works on any system)
uv sync --extra cpu --index pytorch-cpu

# NVIDIA GPU (CUDA 12.1)
uv sync --extra cuda --index pytorch-cu121

# AMD GPU (ROCm 6.2) - untested, may work
uv sync --extra rocm --index pytorch-rocm62
```

## Usage

### Basic Conversion

```bash
uv run rkllama-convert convert Qwen/Qwen2.5-0.5B-Instruct
```

### With Options

```bash
uv run rkllama-convert convert Qwen/Qwen2.5-7B-Instruct \
    -o ./models/qwen \
    -q Q8_0 \
    -c 4096 \
    -p rk3588 \
    -n 3
```

### All Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output directory | `./output` |
| `-q, --quant` | Quantization (Q4_0, Q4_K_M, Q8_0, Q8_K_M) | `Q4_0` |
| `-c, --context` | Max context length | `4096` |
| `-p, --platform` | Target platform (rk3588, rk3576, rk3562, rv1126b) | `rk3588` |
| `-n, --npu-cores` | NPU cores (RK3588: 1-3, RK3576: 1-2) | `3` |
| `-d, --dtype` | Model dtype (float16, float32) | `float16` |
| `--device` | Compute device (cuda, rocm, cpu, auto) | `auto` |
| `-t, --token` | HuggingFace token (or set `HF_TOKEN` env var) | None |
| `-s, --system` | System prompt for Modelfile | None |
| `--temp` | Temperature | `0.7` |
| `--top-k` | Top-K sampling | `40` |
| `--top-p` | Top-P sampling | `0.9` |
| `--thinking/--no-thinking` | Enable reasoning mode | `False` |

### Other Commands

```bash
# Show model info before conversion
uv run rkllama-convert info Qwen/Qwen2.5-7B-Instruct

# List quantization formats
uv run rkllama-convert list-quants

# List available devices
uv run rkllama-convert list-devices

# Show version
uv run rkllama-convert --version
```

## Output Files

The converter generates:

```
output/Model-Name/
├── Model-Name.rkllm    # Converted model binary
├── Modelfile           # RKLlama configuration
└── metadata.json       # Conversion metadata
```

Copy the output directory to your RKLlama models folder to use.

## Quantization Formats

| Format | RKLLM Type | Size | Quality |
|--------|------------|------|---------|
| Q4_0 | w4a16 | ~25% of FP16 | Good |
| Q4_K_M | w4a16_g128 | ~25% of FP16 | Better |
| Q8_0 | w8a8 | ~50% of FP16 | Excellent |
| Q8_K_M | w8a8_g512 | ~50% of FP16 | Best |

## Troubleshooting

### HuggingFace Token

For gated models (Llama, etc.), set your token:

```bash
export HF_TOKEN="hf_xxxxx"
uv run rkllama-convert convert meta-llama/Llama-3.2-3B-Instruct
```

### Memory Issues

- Use CPU with `--device cpu` for large models
- Use Q4_0 quantization for smaller output size
- Reduce context length with `-c 2048`

### Import Errors

Make sure you installed with the correct extra for your system:

```bash
# Check which torch is installed
uv run python -c "import torch; print(torch.__version__)"
```

## License

MIT
