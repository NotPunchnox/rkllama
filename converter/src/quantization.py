"""
Quantization module for converting model weights to different formats.
"""

import torch
import numpy as np
from typing import Any, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class QuantizationConverter:
    """Handles conversion of model weights to different quantization formats."""
    
    @staticmethod
    def convert_weights(model: Any, source_format: str, target_format: str) -> Any:
        """
        Convert model weights from source format to target format.
        
        Args:
            model: The model to convert
            source_format: Source quantization format (e.g., 'Q4_0')
            target_format: Target quantization format (e.g., 'w4a16')
            
        Returns:
            The converted model
        """
        logger.info(f"Converting weights from {source_format} to {target_format}")
        
        try:
            # Convert each layer's weights
            for name, param in model.named_parameters():
                if 'weight' in name:
                    logger.debug(f"Converting weights for {name}")
                    param.data = QuantizationConverter._convert_tensor(
                        param.data,
                        source_format,
                        target_format
                    )
            
            return model
        except Exception as e:
            logger.error(f"Error converting weights: {str(e)}")
            raise
    
    @staticmethod
    def _convert_tensor(tensor: torch.Tensor, source_format: str, target_format: str) -> torch.Tensor:
        """
        Convert a single tensor from source format to target format.
        
        Args:
            tensor: The tensor to convert
            source_format: Source quantization format
            target_format: Target quantization format
            
        Returns:
            The converted tensor
        """
        # Convert to numpy for easier manipulation
        data = tensor.detach().cpu().numpy()
        
        if source_format == 'Q4_0':
            # Convert from Q4_0 to target format
            if target_format == 'w4a16':
                # Convert to 4-bit weights with 16-bit activations
                data = QuantizationConverter._convert_q4_to_w4a16(data)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
        elif source_format == 'Q8_0':
            # Convert from Q8_0 to target format
            if target_format == 'w8a8':
                # Convert to 8-bit weights with 8-bit activations
                data = QuantizationConverter._convert_q8_to_w8a8(data)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
        else:
            raise ValueError(f"Unsupported source format: {source_format}")
        
        # Convert back to torch tensor
        return torch.from_numpy(data).to(tensor.device)
    
    @staticmethod
    def _convert_q4_to_w4a16(data: np.ndarray) -> np.ndarray:
        """Convert from Q4_0 to w4a16 format."""
        # Scale the data to 4-bit range
        min_val = np.min(data)
        max_val = np.max(data)
        scale = (max_val - min_val) / 15  # 4-bit range is 0-15
        
        # Quantize to 4-bit
        quantized = np.round((data - min_val) / scale)
        
        # Clip to 4-bit range
        quantized = np.clip(quantized, 0, 15)
        
        # Convert back to float16 range
        return quantized * scale + min_val
    
    @staticmethod
    def _convert_q8_to_w8a8(data: np.ndarray) -> np.ndarray:
        """Convert from Q8_0 to w8a8 format."""
        # Scale the data to 8-bit range
        min_val = np.min(data)
        max_val = np.max(data)
        scale = (max_val - min_val) / 255  # 8-bit range is 0-255
        
        # Quantize to 8-bit
        quantized = np.round((data - min_val) / scale)
        
        # Clip to 8-bit range
        quantized = np.clip(quantized, 0, 255)
        
        # Convert back to float8 range
        return quantized * scale + min_val

def quantize_tensor(tensor: torch.Tensor, bits: int, group_size: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Quantize a tensor to a specified number of bits.
    
    Args:
        tensor: The tensor to quantize
        bits: Number of bits to quantize to
        group_size: Size of groups for grouped quantization (optional)
        
    Returns:
        Tuple of (quantized tensor, quantization metadata)
    """
    # Convert to numpy for easier manipulation
    data = tensor.detach().cpu().numpy()
    
    # Calculate scale and zero point
    min_val = np.min(data)
    max_val = np.max(data)
    scale = (max_val - min_val) / (2**bits - 1)
    zero_point = min_val
    
    # Quantize
    quantized = np.round((data - zero_point) / scale)
    
    # Clip to valid range
    quantized = np.clip(quantized, 0, 2**bits - 1)
    
    # Convert back to torch tensor
    quantized_tensor = torch.from_numpy(quantized).to(tensor.device)
    
    # Create metadata
    metadata = {
        "scale": scale,
        "zero_point": zero_point,
        "bits": bits,
        "group_size": group_size
    }
    
    return quantized_tensor, metadata 