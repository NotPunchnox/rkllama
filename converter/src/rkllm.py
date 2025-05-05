"""
RKLLM format converter module.
This module handles the conversion of quantized model weights to RKLLM format.
"""

import os
import struct
import numpy as np
import torch
from typing import Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RKLLMHeader:
    """Header structure for RKLLM format."""
    magic: bytes = b'RKLL'
    version: int = 1
    model_type: str = ""
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    max_seq_len: int = 0
    quantization: str = ""
    
    def to_bytes(self) -> bytes:
        """Convert header to bytes."""
        # Magic number (4 bytes)
        header = self.magic
        
        # Version (4 bytes)
        header += struct.pack('<I', self.version)
        
        # Model type (32 bytes, padded with zeros)
        model_type_bytes = self.model_type.encode('utf-8')
        header += model_type_bytes.ljust(32, b'\0')
        
        # Model parameters (4 bytes each)
        header += struct.pack('<IIII', 
            self.vocab_size,
            self.hidden_size,
            self.num_layers,
            self.max_seq_len
        )
        
        # Quantization type (16 bytes, padded with zeros)
        quant_bytes = self.quantization.encode('utf-8')
        header += quant_bytes.ljust(16, b'\0')
        
        return header

class RKLLMConverter:
    """Converts quantized model weights to RKLLM format."""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
    
    def convert(self, output_path: str) -> None:
        """Convert model to RKLLM format and save to file."""
        logger.info("Converting model to RKLLM format...")
        
        # Create header
        header = self._create_header()
        
        # Convert weights
        weights = self._convert_weights()
        
        # Write to file
        self._write_to_file(header, weights, output_path)
        
        logger.info(f"RKLLM file saved to {output_path}")
    
    def _create_header(self) -> RKLLMHeader:
        """Create RKLLM header from model configuration."""
        model_config = self.model.config
        
        return RKLLMHeader(
            model_type=model_config.model_type,
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_hidden_layers,
            max_seq_len=self.config.get('max_context_len', 4096),
            quantization=self.config.get('quantization', 'Q4_0')
        )
    
    def _convert_weights(self) -> bytes:
        """Convert model weights to RKLLM format."""
        weights_data = bytearray()
        
        # Convert each layer's weights
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                logger.info(f"Converting weights for layer: {name}")
                # Convert tensor to numpy array
                weight_array = param.detach().cpu().numpy()
                logger.info(f"Weight shape: {weight_array.shape}")
                
                # Convert to RKLLM format based on quantization
                quant_type = self.config.get('quantization', 'Q4_0')
                if quant_type == 'Q4_0':
                    weight_bytes = self._convert_to_q4_0(weight_array)
                elif quant_type == 'Q4_K_M':
                    weight_bytes = self._convert_to_q4_k_m(weight_array)
                elif quant_type == 'Q8_0':
                    weight_bytes = self._convert_to_q8_0(weight_array)
                elif quant_type == 'Q8_K_M':
                    weight_bytes = self._convert_to_q8_k_m(weight_array)
                else:
                    raise ValueError(f"Unsupported quantization type: {quant_type}")
                
                weights_data.extend(weight_bytes)
                logger.info(f"Converted {len(weight_bytes)} bytes")
        
        logger.info(f"Total converted size: {len(weights_data)} bytes")
        return bytes(weights_data)
    
    def _convert_to_q4_0(self, weight_array: np.ndarray) -> bytes:
        """Convert weights to Q4_0 format."""
        logger.info("Converting to Q4_0 format...")
        
        # Reshape array to 2D if needed
        original_shape = weight_array.shape
        logger.info(f"Original shape: {original_shape}")
        
        # Handle 1D tensors
        if len(original_shape) == 1:
            weight_array = weight_array.reshape(1, -1)
            logger.info(f"Reshaped 1D tensor to: {weight_array.shape}")
        elif len(original_shape) > 2:
            weight_array = weight_array.reshape(-1, original_shape[-1])
            logger.info(f"Reshaped to: {weight_array.shape}")
        
        # Ensure even number of columns for 4-bit packing
        if weight_array.shape[1] % 2 != 0:
            # Pad with zeros if needed
            pad_width = ((0, 0), (0, 1))
            weight_array = np.pad(weight_array, pad_width, mode='constant')
            logger.info(f"Padded shape: {weight_array.shape}")
        
        # Scale weights to int8 range (-8 to 7)
        weight_array = np.clip(weight_array * 8, -8, 7)
        
        # Convert to int8
        q4_weights = weight_array.astype(np.int8)
        
        # Pack two 4-bit values into each byte
        rows, cols = q4_weights.shape
        packed_weights = np.zeros((rows, cols // 2), dtype=np.uint8)
        logger.info(f"Packed shape: {packed_weights.shape}")
        
        try:
            for i in range(0, cols, 2):
                # Ensure values are in 4-bit range
                low_nibble = q4_weights[:, i] & 0x0F
                high_nibble = q4_weights[:, i + 1] & 0x0F
                
                # Pack into bytes
                packed_weights[:, i // 2] = (high_nibble << 4) | low_nibble
            
            # Add metadata
            metadata = struct.pack('<III', 
                original_shape[0] if len(original_shape) > 1 else 1,
                original_shape[1] if len(original_shape) > 1 else original_shape[0],
                len(original_shape)
            )
            logger.info(f"Metadata size: {len(metadata)} bytes")
            
            result = metadata + packed_weights.tobytes()
            logger.info(f"Total size: {len(result)} bytes")
            return result
            
        except Exception as e:
            logger.error(f"Error in Q4_0 conversion: {str(e)}")
            raise
    
    def _convert_to_q4_k_m(self, weight_array: np.ndarray) -> bytes:
        """Convert weights to Q4_K_M format."""
        # Similar to Q4_0 but with additional metadata
        # TODO: Implement Q4_K_M conversion
        raise NotImplementedError("Q4_K_M conversion not yet implemented")
    
    def _convert_to_q8_0(self, weight_array: np.ndarray) -> bytes:
        """Convert weights to Q8_0 format."""
        # Convert to 8-bit integers
        q8_weights = np.clip(weight_array * 128, -128, 127).astype(np.int8)
        return q8_weights.tobytes()
    
    def _convert_to_q8_k_m(self, weight_array: np.ndarray) -> bytes:
        """Convert weights to Q8_K_M format."""
        # Similar to Q8_0 but with additional metadata
        # TODO: Implement Q8_K_M conversion
        raise NotImplementedError("Q8_K_M conversion not yet implemented")
    
    def _write_to_file(self, header: RKLLMHeader, weights: bytes, output_path: str) -> None:
        """Write RKLLM format to file."""
        with open(output_path, 'wb') as f:
            # Write header
            f.write(header.to_bytes())
            
            # Write weights
            f.write(weights) 