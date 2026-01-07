# Installation Guide

This guide covers installing the RKLlama Converter with different GPU backends.

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Sufficient RAM (16GB+ recommended for larger models)

### System Dependencies

#### Ubuntu/Debian

```bash
# Base dependencies
sudo apt update
sudo apt install -y python3.10 python3-pip git

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Fedora/RHEL

```bash
sudo dnf install -y python3.10 python3-pip git
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Arch Linux

```bash
sudo pacman -S python python-pip git
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Installation from Source

Since the converter is not yet published to PyPI, install from the repository:

```bash
# Clone the repository
git clone https://github.com/notpunchnox/rkllama
cd rkllama/converter
```

### NVIDIA GPU (CUDA)

**System requirements:**
- NVIDIA GPU with compute capability 3.5+
- NVIDIA driver 525+ installed
- CUDA 12.1 compatible

```bash
# Install NVIDIA drivers (Ubuntu/Debian)
sudo apt install -y nvidia-driver-535

# Verify GPU is detected
nvidia-smi

# Install converter with CUDA support
uv sync --extra cuda --index pytorch-cu121

# Run the converter
uv run rkllama-convert --help
```

### AMD GPU (ROCm)

**System requirements:**
- AMD GPU with ROCm support (RX 6000/7000 series, MI series)
- ROCm 6.2 drivers
- Linux only (ROCm not supported on Windows)

**Supported AMD GPUs:**
- Radeon RX 7900 XTX/XT
- Radeon RX 7800 XT/7700 XT
- Radeon RX 6900 XT/6800 XT/6800
- Radeon RX 6700 XT/6600 XT
- Instinct MI300X/MI250X/MI210/MI100

```bash
# Install ROCm (Ubuntu 22.04/24.04)
# See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

# Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install -y rocm-hip-libraries

# Verify GPU is detected
rocm-smi

# Install converter with ROCm support
uv sync --extra rocm --index pytorch-rocm62

# Run the converter
uv run rkllama-convert --help
```

### CPU Only

For systems without a supported GPU:

```bash
uv sync --extra cpu --index pytorch-cpu

uv run rkllama-convert --help
```

**Note:** CPU conversion is significantly slower than GPU conversion. A 7B model may take 30+ minutes on CPU vs 5-10 minutes on GPU.

---

## Development Installation

For contributing or modifying the converter:

```bash
# Clone the repository
git clone https://github.com/notpunchnox/rkllama
cd rkllama

# Install with dev dependencies
uv sync --extra dev --extra cuda  # or --extra rocm / --extra cpu

# Run tests
uv run pytest converter/tests/
```

---

## Verifying Installation

Check that the converter is installed correctly:

```bash
# Check version
uv run rkllama-convert --version

# List available devices
uv run rkllama-convert list-devices

# List quantization formats
uv run rkllama-convert list-quants
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for private/gated models |
| `CUDA_VISIBLE_DEVICES` | Limit which GPUs are visible (NVIDIA) |
| `HIP_VISIBLE_DEVICES` | Limit which GPUs are visible (AMD) |

## Troubleshooting Installation

### CUDA not found

If you see `CUDA not available`:
1. Check NVIDIA drivers: `nvidia-smi`
2. Reinstall with CUDA extra: `uv sync --extra cuda --reinstall`

### ROCm not detected

If AMD GPU not detected:
1. Check ROCm installation: `rocm-smi`
2. Ensure you installed with ROCm extra: `uv sync --extra rocm`
3. Check `HIP_VISIBLE_DEVICES` is not set to empty

### Memory errors

If you run out of memory during conversion:
1. Use a smaller quantization (Q4_0 uses less memory than Q8_0)
2. Reduce context length with `-c 2048`
3. Use CPU conversion (slower but uses system RAM)

## Next Steps

- [Quick Start Guide](index.md) - Convert your first model
- [CLI Reference](cli-reference.md) - All command options
- [Quantization Guide](quantization.md) - Choose the right quantization
