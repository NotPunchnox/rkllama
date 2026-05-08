# Hugging Face to RKLLM Converter

A tool to convert Hugging Face models to RKLLM format for Rockchip RK3588(S) and RK3576 processors.

## Features

- Conversion of Hugging Face models to RKLLM format
- Support for various model architectures (Qwen, OPT, etc.)
- Support for Q4_0, Q4_K_M, Q8_0, Q8_K_M quantization formats
- Automatic configuration file generation
- Model and parameter validation
- Model metadata support
- Command-line interface for easy conversion
- Detailed logging of conversion process
- Support for 1D and 2D tensor conversion

## Prerequisites

- Python 3.8 or higher
- Docker (for execution on Orange Pi 5)
- Hugging Face account (for accessing models)
- Hugging Face token (for private models)
- CUDA-capable GPU (recommended for faster conversion)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/huggingface-to-rkllm.git
cd huggingface-to-rkllm
```

2. Create and activate a virtual environment (recommended):
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'torch'**
   ```bash
   # Install PyTorch separately
   pip install torch torchvision torchaudio
   ```

2. **CUDA not available**
   - Make sure you have CUDA installed on your system
   - For CPU-only usage, use the `--device cpu` option

3. **Hugging Face token issues**
   - Set your token as an environment variable:
     ```bash
     export HF_TOKEN="your_token_here"
     ```
   - Or use the `--token` option in the command line

4. **Memory issues with large models**
   - Try using a smaller model first
   - Use CPU instead of GPU with `--device cpu`
   - Reduce batch size or context length

### Getting Help

If you encounter any issues:
1. Check the error message carefully
2. Make sure all dependencies are installed
3. Try running with verbose logging:
   ```bash
   python3 converter.py --verbose Qwen/Qwen2.5-7B
   ```
4. Check the [GitHub Issues](https://github.com/yourusername/huggingface-to-rkllm/issues) page

## Usage

### Command Line Interface

The simplest way to convert a model is using the command-line interface:

```bash
# Basic conversion
python3 converter.py Qwen/Qwen2.5-7B

# Conversion with custom options
python3 converter.py Qwen/Qwen2.5-7B \
    --output-dir "models/converted" \
    --quantization "Q4_K_M" \
    --max-context-len 8192 \
    --dtype "float16" \
    --device "cuda" \
    --token "your_hf_token"
```

### Command Line Options

- `model_id`: ID of the model on Hugging Face (required)
- `--output-dir`: Output directory for the converted model (default: "data/converted")
- `--quantization`: Quantization format (choices: Q4_0, Q4_K_M, Q8_0, Q8_K_M)
- `--max-context-len`: Maximum context length in tokens (default: 4096)
- `--dtype`: Model data type (float16 or float32, default: float16)
- `--device`: Device to use for conversion (cuda or cpu, default: cuda if available)
- `--token`: Hugging Face token for private models

### Output Files

The converter generates the following files in the output directory:

- `model.rkllm`: The converted model in RKLLM format
- `Modelfile`: Configuration file for the model
- `metadata.json`: Model metadata including conversion parameters

### Python API

You can also use the converter programmatically:

```python
from src.converter import ConversionConfig, HuggingFaceToRKLLMConverter

config = ConversionConfig(
    model_id="Qwen/Qwen2.5-7B",
    output_dir="path/to/output",
    quantization="Q4_0",
    max_context_len=4096,
    dtype="float16",
    device="cuda",
    token="your_hf_token"  # Optional
)

converter = HuggingFaceToRKLLMConverter(config)
converter.convert()
```

### Docker Usage

1. Build the Docker image:
```bash
docker build -t huggingface-to-rkllm .
```

2. Run the conversion:
```bash
docker run -v /path/to/models:/models huggingface-to-rkllm convert \
    --model-id Qwen/Qwen2.5-7B \
    --output-dir /models/output \
    --quantization Q4_0
```

### Example Conversions

Here are some example commands for different use cases:

```bash
# Convert OPT-125M model with default settings
python3 converter.py facebook/opt-125m

# Convert Qwen2.5-7B with custom quantization
python3 converter.py Qwen/Qwen2.5-7B --quantization Q4_K_M

# Convert a private model using Hugging Face token
python3 converter.py your-org/private-model --token "hf_xxxxx"

# Convert model with CPU (when no GPU is available)
python3 converter.py Qwen/Qwen2.5-7B --device cpu
```

## Project Structure

```
converter/
├── src/
│   ├── converter.py      # Main conversion module
│   ├── quantization.py   # Quantization handling
│   └── utils.py         # Utility functions
├── tests/
│   └── test_converter.py # Unit tests
├── requirements.txt     # Python dependencies
└── README.md           # Documentation
```

## Testing

Run the tests:
```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please:
1. Fork the project
2. Create a branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. 