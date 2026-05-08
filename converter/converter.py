"""
Main script for converting Hugging Face models to RKLLM format.
"""

import os
import sys
import argparse
import logging
import torch
from src.converter import HuggingFaceToRKLLMConverter, ConversionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert Hugging Face models to RKLLM format')
    parser.add_argument('model_id', help='Hugging Face model ID')
    parser.add_argument('--output-dir', default='data/models', help='Output directory for converted models')
    parser.add_argument('--quantization', default='Q4_0', help='Quantization format (Q4_0, Q4_K_M, Q8_0, Q8_K_M)')
    parser.add_argument('--max-context-len', type=int, default=4096, help='Maximum context length')
    parser.add_argument('--dtype', default='float16', help='Model data type (float16 or float32)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for conversion')
    parser.add_argument('--token', help='Hugging Face token for private models')
    
    return parser.parse_args()

def main():
    """Main function."""
    try:
        args = parse_args()
        
        # Create configuration
        config = ConversionConfig(
            model_id=args.model_id,
            output_dir=args.output_dir,
            quantization=args.quantization,
            max_context_len=args.max_context_len,
            dtype=args.dtype,
            device=args.device,
            token=args.token
        )
        
        # Create converter and run conversion
        converter = HuggingFaceToRKLLMConverter(config)
        converter.convert()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 