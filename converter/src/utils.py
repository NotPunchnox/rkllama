"""
Utility functions for model conversion and metadata handling.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for a converted model."""
    name: str
    architecture: str
    quantization: str
    parameters: int
    context_length: int
    system_prompt: str
    temperature: float

def save_model_metadata(metadata: ModelMetadata, output_path: str) -> None:
    """
    Save model metadata to a JSON file.
    
    Args:
        metadata: The model metadata to save
        output_path: The directory to save the metadata in
    """
    try:
        # Create metadata dictionary
        metadata_dict = {
            "name": metadata.name,
            "architecture": metadata.architecture,
            "quantization": metadata.quantization,
            "parameters": metadata.parameters,
            "context_length": metadata.context_length,
            "system_prompt": metadata.system_prompt,
            "temperature": metadata.temperature
        }
        
        # Save to JSON file
        metadata_path = os.path.join(output_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
        raise

def load_model_metadata(metadata_path: str) -> ModelMetadata:
    """
    Load model metadata from a JSON file.
    
    Args:
        metadata_path: Path to the metadata JSON file
        
    Returns:
        The loaded model metadata
    """
    try:
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
        
        return ModelMetadata(
            name=metadata_dict["name"],
            architecture=metadata_dict["architecture"],
            quantization=metadata_dict["quantization"],
            parameters=metadata_dict["parameters"],
            context_length=metadata_dict["context_length"],
            system_prompt=metadata_dict["system_prompt"],
            temperature=metadata_dict["temperature"]
        )
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        raise

def get_model_size(model_path: str) -> int:
    """Get the size of a model file in bytes."""
    return os.path.getsize(model_path)

def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)

def validate_model_path(path: str) -> bool:
    """Validate if a path points to a valid model file."""
    if not os.path.exists(path):
        return False
    if not os.path.isfile(path):
        return False
    return True

def get_model_architecture(model_path: str) -> Optional[str]:
    """Detect the model architecture from the model file."""
    # TODO: Implement architecture detection
    pass 