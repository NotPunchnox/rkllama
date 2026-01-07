"""
Hugging Face to RKLLM Converter
This module provides functionality to convert Hugging Face models to RKLLM format
using the official rkllm-toolkit.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import torch
from huggingface_hub import snapshot_download

from .rkllm import RKLLMConfig, RKLLMConverter

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""

    model_id: str
    output_dir: str
    quantization: str = "Q8_0"
    max_context_len: int = 4096
    dtype: str = "float16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    token: str | None = None
    target_platform: str = "rk3588"
    num_npu_core: int = 3

    @property
    def model_name(self) -> str:
        """Get model name from model ID."""
        return self.model_id.split("/")[-1]

    @property
    def output_path(self) -> str:
        """Get the full output path including model name."""
        return os.path.join(self.output_dir, self.model_name)


class HuggingFaceToRKLLMConverter:
    """Converts Hugging Face models to RKLLM format using official toolkit."""

    QUANTIZATION_MAPPING = {
        "Q4_0": "w4a16",
        "Q4_K_M": "w4a16_g128",
        "Q8_0": "w8a8",
        "Q8_K_M": "w8a8_g512",
    }

    def __init__(self, config: ConversionConfig):
        self.config = config
        self._validate_config()
        self._model_path: str | None = None

    def _validate_config(self) -> None:
        """Validate the conversion configuration."""
        if self.config.quantization not in self.QUANTIZATION_MAPPING:
            raise ValueError(f"Unsupported quantization: {self.config.quantization}")

    def convert(self) -> None:
        """Main conversion method using official rkllm-toolkit."""
        logger.info(f"Starting conversion of {self.config.model_name}")

        # Create output directory
        os.makedirs(self.config.output_path, exist_ok=True)

        # Step 1: Download/locate model
        self._prepare_model()

        # Step 2: Convert using official toolkit
        self._generate_rkllm_file()

        # Step 3: Create Modelfile
        self._create_modelfile()

        # Step 4: Save metadata
        self._save_metadata(self.config.output_path)

        logger.info("Conversion completed successfully")

    def _prepare_model(self) -> None:
        """Download or locate the HuggingFace model."""
        logger.info(f"Preparing model {self.config.model_id}...")

        # Check if it's a local path
        if os.path.isdir(self.config.model_id):
            self._model_path = self.config.model_id
            logger.info(f"Using local model at {self._model_path}")
        else:
            # Download from HuggingFace
            logger.info("Downloading model from HuggingFace...")
            self._model_path = snapshot_download(
                self.config.model_id,
                token=self.config.token,
            )
            logger.info(f"Model downloaded to {self._model_path}")

    def _generate_rkllm_file(self) -> None:
        """Generate the RKLLM binary file using official toolkit."""
        logger.info("Generating RKLLM file with official toolkit...")

        model_name = self.config.model_name
        output_file = os.path.join(self.config.output_path, f"{model_name}.rkllm")
        assert self._model_path
        # Create RKLLM config
        rkllm_config = RKLLMConfig(
            model_path=self._model_path,
            output_path=output_file,
            quantized_dtype=self.QUANTIZATION_MAPPING[self.config.quantization],
            target_platform=self.config.target_platform,
            max_context=self.config.max_context_len,
            num_npu_core=self.config.num_npu_core,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        # Run conversion
        converter = RKLLMConverter(rkllm_config)
        converter.convert()

        logger.info(f"RKLLM file generated at {output_file}")

    def _create_modelfile(self) -> None:
        """Create Modelfile for the converted model."""
        logger.info("Creating Modelfile...")

        model_name = self.config.model_name

        modelfile_content = f'''FROM="{model_name}.rkllm"
HUGGINGFACE_PATH="{self.config.model_id}"
SYSTEM="You are a helpful AI assistant."
TEMPERATURE=0.7
'''

        modelfile_path = os.path.join(self.config.output_path, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        logger.info(f"Modelfile created at {modelfile_path}")

    def _save_metadata(self, output_dir: str) -> None:
        """Save metadata about the conversion to a JSON file."""
        metadata = {
            "model_id": self.config.model_id,
            "quantization": self.config.quantization,
            "target_platform": self.config.target_platform,
            "conversion_date": datetime.now().isoformat(),
            "toolkit_version": "1.2.3",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
            },
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Main entry point for the converter."""
    config = ConversionConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="./output",
        quantization="Q8_0",
        token=os.getenv("HF_TOKEN"),
        dtype="float16",
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_platform="rk3588",
    )

    converter = HuggingFaceToRKLLMConverter(config)
    converter.convert()


if __name__ == "__main__":
    main()
