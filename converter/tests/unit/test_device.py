"""
Unit tests for device detection module.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_device_type_values(self):
        """Test that DeviceType has expected values."""
        from rkllama_converter.device import DeviceType

        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.ROCM.value == "rocm"
        assert DeviceType.CPU.value == "cpu"


class TestDetectDevice:
    """Tests for detect_device function."""

    def test_detect_cpu_when_no_cuda(self, mock_torch_cuda_unavailable):
        """Test CPU is detected when CUDA is unavailable."""
        from rkllama_converter.device import DeviceType, detect_device

        device = detect_device()
        assert device == DeviceType.CPU

    def test_detect_cuda_when_available(self):
        """Test CUDA is detected when available and not ROCm."""
        from rkllama_converter.device import DeviceType, detect_device

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.version", MagicMock(hip=None)):
                device = detect_device()
                assert device == DeviceType.CUDA

    def test_detect_rocm_when_hip_available(self):
        """Test ROCm is detected when HIP version is present."""
        from rkllama_converter.device import DeviceType, detect_device

        mock_version = MagicMock()
        mock_version.hip = "6.2.0"

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.version", mock_version):
                device = detect_device()
                assert device == DeviceType.ROCM


class TestGetTorchDevice:
    """Tests for get_torch_device function."""

    def test_cuda_returns_cuda_device(self):
        """Test CUDA DeviceType returns cuda torch device."""
        from rkllama_converter.device import DeviceType, get_torch_device

        device = get_torch_device(DeviceType.CUDA)
        assert str(device) == "cuda"

    def test_rocm_returns_cuda_device(self):
        """Test ROCm DeviceType returns cuda torch device (ROCm uses cuda in PyTorch)."""
        from rkllama_converter.device import DeviceType, get_torch_device

        device = get_torch_device(DeviceType.ROCM)
        assert str(device) == "cuda"

    def test_cpu_returns_cpu_device(self):
        """Test CPU DeviceType returns cpu torch device."""
        from rkllama_converter.device import DeviceType, get_torch_device

        device = get_torch_device(DeviceType.CPU)
        assert str(device) == "cpu"


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_cpu_info_includes_cores(self):
        """Test CPU info includes core count."""
        from rkllama_converter.device import DeviceType, get_device_info

        info = get_device_info(DeviceType.CPU)
        assert info["type"] == "cpu"
        assert "cores" in info

    def test_cuda_info_when_available(self, mock_torch_cuda_available, mock_cuda_device_info):
        """Test CUDA info when GPU is available."""
        from rkllama_converter.device import DeviceType, get_device_info

        info = get_device_info(DeviceType.CUDA)
        assert info["type"] == "cuda"
        assert info["available"] is True
        assert "name" in info
        assert "memory_gb" in info

    def test_cuda_info_when_unavailable(self, mock_torch_cuda_unavailable):
        """Test CUDA info when GPU is unavailable."""
        from rkllama_converter.device import DeviceType, get_device_info

        info = get_device_info(DeviceType.CUDA)
        assert info["type"] == "cuda"
        assert info.get("available") is False


class TestValidateDevice:
    """Tests for validate_device function."""

    def test_cpu_always_valid(self):
        """Test CPU device is always valid."""
        from rkllama_converter.device import DeviceType, validate_device

        assert validate_device(DeviceType.CPU) is True

    def test_cuda_invalid_when_unavailable(self, mock_torch_cuda_unavailable):
        """Test CUDA is invalid when not available."""
        from rkllama_converter.device import DeviceType, validate_device

        assert validate_device(DeviceType.CUDA) is False


class TestGetDeviceOrFallback:
    """Tests for get_device_or_fallback function."""

    def test_returns_detected_when_none_requested(self, mock_torch_cuda_unavailable):
        """Test auto-detection when no device requested."""
        from rkllama_converter.device import DeviceType, get_device_or_fallback

        device = get_device_or_fallback(None)
        assert device == DeviceType.CPU

    def test_returns_requested_when_valid(self, mock_torch_cuda_unavailable):
        """Test returns requested device when valid."""
        from rkllama_converter.device import DeviceType, get_device_or_fallback

        device = get_device_or_fallback(DeviceType.CPU)
        assert device == DeviceType.CPU

    def test_falls_back_when_requested_invalid(self, mock_torch_cuda_unavailable):
        """Test falls back to auto-detect when requested device invalid."""
        from rkllama_converter.device import DeviceType, get_device_or_fallback

        device = get_device_or_fallback(DeviceType.CUDA)
        # Should fall back to CPU since CUDA is unavailable
        assert device == DeviceType.CPU
