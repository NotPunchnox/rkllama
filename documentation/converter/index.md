# RKLlama Converter

Convert HuggingFace transformer models to RKLLM format for deployment on Rockchip NPU devices (RK3588/RK3576).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/notpunchnox/rkllama
cd rkllama/converter

# Install with uv (choose your GPU backend)
uv sync --extra cuda --index pytorch-cu121    # NVIDIA GPU (CUDA 12.1)
uv sync --extra rocm --index pytorch-rocm62   # AMD GPU (ROCm 6.2)
uv sync --extra cpu --index pytorch-cpu       # CPU only

# Convert a model
uv run rkllama-convert Qwen/Qwen2.5-7B-Instruct -q Q4_0 -o ./models/qwen
```

## Features

- **Modern CLI** - Rich terminal output with progress bars and colored status
- **Multiple backends** - Support for CUDA (NVIDIA), ROCm (AMD), and CPU
- **Quantization options** - Q4_0, Q4_K_M, Q8_0, Q8_K_M formats
- **Modelfile generation** - Automatic configuration file creation
- **Auto device detection** - Automatically selects best available GPU

## Commands

### convert

Convert a HuggingFace model to RKLLM format:

```bash
rkllama-convert <model_id> [OPTIONS]
```

**Arguments:**
- `model_id` - HuggingFace model ID (e.g., `Qwen/Qwen2.5-7B-Instruct`)

**Options:**
- `-o, --output` - Output directory (default: `./output`)
- `-q, --quant` - Quantization format: Q4_0, Q4_K_M, Q8_0, Q8_K_M (default: Q4_0)
- `-c, --context` - Max context length in tokens (default: 4096)
- `-d, --dtype` - Model dtype: float16, float32 (default: float16)
- `--device` - Compute device: cuda, rocm, cpu, auto (default: auto)
- `-t, --token` - HuggingFace token for private models (or set `HF_TOKEN` env var)
- `-s, --system` - System prompt for the model
- `--temp` - Generation temperature (default: 0.7)
- `--top-k` - Top-K sampling parameter (default: 40)
- `--top-p` - Nucleus sampling parameter (default: 0.9)
- `--thinking/--no-thinking` - Enable reasoning mode

**Example:**
```bash
rkllama-convert Qwen/Qwen2.5-7B-Instruct \
    -q Q4_0 \
    -o ./models/qwen \
    -c 8192 \
    --system "You are a coding assistant." \
    --thinking
```

### info

Show information about a HuggingFace model before conversion:

```bash
rkllama-convert info <model_id>
```

### list-quants

Display available quantization formats:

```bash
rkllama-convert list-quants
```

### list-devices

Show available compute devices:

```bash
rkllama-convert list-devices
```

## Output Files

After conversion, you'll find these files in the output directory:

```
output/Model-Name/
├── Model-Name.rkllm    # The converted model binary
├── Modelfile           # Configuration for RKLlama server
└── metadata.json       # Conversion metadata
```

Copy the entire directory to your RKLlama models path to use the model.

## Documentation

- [Installation Guide](installation.md) - Detailed installation instructions
- [CLI Reference](cli-reference.md) - Complete command reference
- [Quantization Guide](quantization.md) - Understanding quantization formats
- [Modelfile Reference](modelfile.md) - Modelfile configuration options
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Requirements

- Python 3.10+
- PyTorch 2.2+ (with CUDA, ROCm, or CPU support)
- 16GB+ RAM (varies by model size)
- GPU recommended for faster conversion
