"""
RKLlama Converter - Convert HuggingFace models to RKLLM format.

This package provides tools to convert HuggingFace transformer models
to RKLLM format for deployment on Rockchip NPU devices (RK3588/RK3576).
"""

__version__ = "1.2.3-2"

from .converter import ConversionConfig, HuggingFaceToRKLLMConverter
from .device import DeviceType, detect_device, get_device_info, get_torch_device
from .modelfile import ModelfileConfig, generate_modelfile
from .quantization import QuantizationConverter, quantize_tensor

__all__ = [
    "__version__",
    "ConversionConfig",
    "HuggingFaceToRKLLMConverter",
    "DeviceType",
    "detect_device",
    "get_device_info",
    "get_torch_device",
    "ModelfileConfig",
    "generate_modelfile",
    "QuantizationConverter",
    "quantize_tensor",
]
