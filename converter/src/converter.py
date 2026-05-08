"""
Hugging Face to RKLLM Converter
This module provides functionality to convert Hugging Face models to RKLLM format.
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from .quantization import QuantizationConverter
from .utils import ModelMetadata, save_model_metadata
from .rkllm import RKLLMConverter
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    model_id: str
    output_dir: str
    quantization: str = 'Q4_0'
    max_context_len: int = 4096
    dtype: str = 'float16'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    token: Optional[str] = None
    
    @property
    def model_name(self) -> str:
        """Get model name from model ID."""
        return self.model_id.split('/')[-1]

    @property
    def output_path(self) -> str:
        """Get the full output path including model name."""
        return os.path.join(self.output_dir, self.model_name)

class HuggingFaceToRKLLMConverter:
    """Converts Hugging Face models to RKLLM format."""
    
    QUANTIZATION_MAPPING = {
        'Q4_0': 'w4a16',
        'Q4_K_M': 'w4a16_g128',
        'Q8_0': 'w8a8',
        'Q8_K_M': 'w8a8_g512'
    }
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self._validate_config()
        self.model = None
        self.metadata = None
        self.tokenizer = None
        self.processor = None
        self.rkllm_converter = None
    
    def _validate_config(self) -> None:
        """Validate the conversion configuration."""
        if not self.config.quantization in self.QUANTIZATION_MAPPING:
            raise ValueError(f"Unsupported quantization: {self.config.quantization}")
    
    def convert(self) -> None:
        """Main conversion method."""
        logger.info(f"Starting conversion of {self.config.model_name}")
        
        # Create output directory with model name
        os.makedirs(self.config.output_path, exist_ok=True)
        
        # Step 1: Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Step 2: Convert weights
        self._convert_weights()
        
        # Step 3: Generate RKLLM file
        self._generate_rkllm_file()
        
        # Step 4: Create Modelfile
        self._create_modelfile()
        
        # Step 5: Save metadata
        self._save_metadata(self.config.output_path)
        
        logger.info("Conversion completed successfully")
    
    def _load_model_and_tokenizer(self) -> None:
        """Load the model and tokenizer from Hugging Face."""
        logger.info("Loading model and tokenizer...")
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                token=self.config.token
            )
            
            # Try to load processor for multimodal models
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.config.model_id,
                    token=self.config.token
                )
                logger.info("Loaded multimodal processor")
            except:
                logger.info("No multimodal processor found, using text-only mode")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                token=self.config.token,
                torch_dtype=torch.float16 if self.config.dtype == 'float16' else torch.float32,
                device_map=self.config.device
            )
            
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {str(e)}")
            raise
    
    def _convert_weights(self) -> None:
        """Convert model weights to RKLLM format."""
        logger.info("Converting weights...")
        try:
            # Get the target quantization format
            target_format = self.QUANTIZATION_MAPPING[self.config.quantization]
            
            # Convert weights using our quantization converter
            self.model = QuantizationConverter.convert_weights(
                self.model,
                self.config.quantization,
                target_format
            )
            logger.info("Weights converted successfully")
        except Exception as e:
            logger.error(f"Error converting weights: {str(e)}")
            raise
    
    def _generate_rkllm_file(self) -> None:
        """Generate the RKLLM binary file."""
        logger.info("Generating RKLLM file...")
        try:
            # Initialize RKLLM converter
            self.rkllm_converter = RKLLMConverter(
                model=self.model,
                config={
                    'quantization': self.config.quantization,
                    'max_context_len': self.config.max_context_len
                }
            )
            
            # Convert and save RKLLM file with model name
            model_name = self.config.model_id.split('/')[-1]
            output_path = os.path.join(self.config.output_path, f'{model_name}.rkllm')
            self.rkllm_converter.convert(output_path)
            
            logger.info(f"RKLLM file generated at {output_path}")
        except Exception as e:
            logger.error(f"Error generating RKLLM file: {str(e)}")
            raise
    
    def _create_modelfile(self) -> None:
        """Create Modelfile for the converted model."""
        logger.info("Creating Modelfile...")
        
        # Extract model name from model_id
        model_name = self.config.model_id.split('/')[-1]
        
        modelfile_content = f"""FROM="{model_name}.rkllm"
HUGGINGFACE_PATH="{self.config.model_id}"
SYSTEM="You are a helpful AI assistant."
TEMPERATURE=0.7
"""
        
        modelfile_path = os.path.join(self.config.output_path, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfile created at {modelfile_path}")
    
    def _save_metadata(self, output_dir: str) -> None:
        """Save metadata about the conversion to a JSON file."""
        metadata = {
            "model_id": self.config.model_id,
            "quantization": self.config.quantization,
            "conversion_date": datetime.now().isoformat(),
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
                "stop_sequences": ["Human:", "Assistant:"]
            }
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main entry point for the converter."""
    # Example usage
    config = ConversionConfig(
        model_id="Qwen/Qwen2.5-Omni-7B",
        output_dir="data/converted",
        quantization="Q4_0",
        token=os.getenv("HF_TOKEN"),  # Get token from environment variable
        dtype='float16',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    converter = HuggingFaceToRKLLMConverter(config)
    converter.convert()

if __name__ == "__main__":
    main() 