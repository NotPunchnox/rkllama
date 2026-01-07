"""
Pytest configuration and fixtures for rkllama-converter tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_torch_cuda_available():
    """Mock torch.cuda.is_available() to return True."""
    with patch("torch.cuda.is_available", return_value=True):
        yield


@pytest.fixture
def mock_torch_cuda_unavailable():
    """Mock torch.cuda.is_available() to return False."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_torch_rocm():
    """Mock torch to appear as ROCm installation."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch.object(type(MagicMock()), "hip", "6.2.0", create=True):
            import torch

            original_hip = getattr(torch.version, "hip", None)
            torch.version.hip = "6.2.0"
            yield
            if original_hip is None:
                delattr(torch.version, "hip")
            else:
                torch.version.hip = original_hip


@pytest.fixture
def mock_cuda_device_info():
    """Mock CUDA device properties."""
    mock_props = MagicMock()
    mock_props.total_memory = 10 * (1024**3)  # 10 GB
    mock_props.multi_processor_count = 68
    mock_props.major = 8
    mock_props.minor = 6

    with patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3080"):
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            yield


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_modelfile_config():
    """Create a sample ModelfileConfig for testing."""
    from rkllama_converter.modelfile import ModelfileConfig

    return ModelfileConfig(
        model_file="test-model.rkllm",
        huggingface_path="test-org/test-model",
        system="You are a test assistant.",
        temperature=0.8,
        num_ctx=2048,
        max_new_tokens=512,
        top_k=50,
        top_p=0.95,
    )


@pytest.fixture
def mock_hf_model():
    """Mock a HuggingFace model for testing."""
    mock_model = MagicMock()
    mock_model.config.model_type = "llama"
    mock_model.config.vocab_size = 32000
    mock_model.config.hidden_size = 4096
    mock_model.config.num_hidden_layers = 32

    # Mock named_parameters
    mock_param = MagicMock()
    mock_param.data = MagicMock()
    mock_param.detach.return_value.cpu.return_value.numpy.return_value = MagicMock()
    mock_model.named_parameters.return_value = [("layer.weight", mock_param)]

    return mock_model


@pytest.fixture
def mock_hf_tokenizer():
    """Mock a HuggingFace tokenizer for testing."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 32000
    return mock_tokenizer
