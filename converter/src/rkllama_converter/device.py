"""
Device detection and management for model conversion.

Supports CUDA (NVIDIA), ROCm (AMD), and CPU backends with auto-detection.
"""

from enum import Enum
from typing import Any

import torch


class DeviceType(str, Enum):
    """Supported compute device types."""

    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"


def detect_device() -> DeviceType:
    """
    Auto-detect the best available compute device.

    Detection order:
    1. ROCm (AMD GPU) - detected via torch.version.hip
    2. CUDA (NVIDIA GPU) - detected via torch.cuda.is_available()
    3. CPU - fallback

    Returns:
        DeviceType: The detected device type.

    Example:
        >>> device = detect_device()
        >>> print(f"Using {device.value}")
        Using cuda
    """
    if torch.cuda.is_available():
        # ROCm builds of PyTorch report via torch.version.hip
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return DeviceType.ROCM
        return DeviceType.CUDA
    return DeviceType.CPU


def get_torch_device(device_type: DeviceType) -> torch.device:
    """
    Get a torch.device from a DeviceType.

    Note: ROCm uses the "cuda" device identifier in PyTorch.

    Args:
        device_type: The device type to convert.

    Returns:
        torch.device: The corresponding torch device.

    Example:
        >>> device = get_torch_device(DeviceType.ROCM)
        >>> model.to(device)
    """
    if device_type in (DeviceType.CUDA, DeviceType.ROCM):
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device_type: DeviceType) -> dict[str, Any]:
    """
    Get detailed information about a compute device.

    Args:
        device_type: The device type to query.

    Returns:
        Dict containing device information:
        - type: Device type string
        - name: Device name (for GPU devices)
        - memory_gb: Total memory in GB (for GPU devices)
        - compute_capability: CUDA compute capability (NVIDIA only)
        - hip_version: HIP version (AMD only)

    Example:
        >>> info = get_device_info(DeviceType.CUDA)
        >>> print(f"{info['name']}: {info['memory_gb']:.1f} GB")
        NVIDIA GeForce RTX 3080: 10.0 GB
    """
    info: dict[str, Any] = {"type": device_type.value}

    if device_type == DeviceType.CPU:
        import os

        info["cores"] = os.cpu_count()
        return info

    if device_type in (DeviceType.CUDA, DeviceType.ROCM):
        if not torch.cuda.is_available():
            info["available"] = False
            return info

        info["available"] = True
        info["name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["memory_gb"] = props.total_memory / (1024**3)
        info["multi_processor_count"] = props.multi_processor_count

        if device_type == DeviceType.CUDA:
            info["compute_capability"] = f"{props.major}.{props.minor}"
        elif device_type == DeviceType.ROCM:
            info["hip_version"] = getattr(torch.version, "hip", "unknown")

    return info


def validate_device(device_type: DeviceType) -> bool:
    """
    Validate that a device type is available on the current system.

    Args:
        device_type: The device type to validate.

    Returns:
        True if the device is available, False otherwise.

    Raises:
        ValueError: If the requested device is not available.
    """
    if device_type == DeviceType.CPU:
        return True

    if not torch.cuda.is_available():
        return False

    if device_type == DeviceType.ROCM:
        return hasattr(torch.version, "hip") and torch.version.hip is not None

    if device_type == DeviceType.CUDA:
        # Make sure it's not actually ROCm
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        return not is_rocm

    return False


def get_device_or_fallback(requested: DeviceType | None = None) -> DeviceType:
    """
    Get the requested device or fall back to an available one.

    Args:
        requested: The requested device type, or None for auto-detection.

    Returns:
        DeviceType: The device to use.

    Example:
        >>> # User requested CUDA but only has AMD GPU
        >>> device = get_device_or_fallback(DeviceType.CUDA)
        >>> # Returns DeviceType.ROCM or DeviceType.CPU
    """
    if requested is None:
        return detect_device()

    if validate_device(requested):
        return requested

    # Fall back to auto-detection
    return detect_device()
