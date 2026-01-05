"""Shared pytest fixtures for RKLlama tests."""

import pytest
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, AsyncMock, patch

# Check if RKLLM library is available (ARM-only)
RKLLM_AVAILABLE = False
try:
    from rkllama.api.classes import RKLLM_AVAILABLE
except (ImportError, OSError):
    pass

# Skip marker for tests requiring RKLLM library
requires_rkllm = pytest.mark.skipif(
    not RKLLM_AVAILABLE,
    reason="RKLLM native library not available (ARM-only)"
)


@pytest.fixture
def mock_worker_manager() -> MagicMock:
    """Mock WorkerManager that doesn't actually load models."""
    manager = MagicMock()
    manager.workers = {}
    manager.exists_model_loaded = MagicMock(return_value=False)
    manager.add_worker = MagicMock(return_value=True)
    manager.stop_worker = MagicMock(return_value=None)
    manager.stop_all = MagicMock(return_value=None)

    # Mock result queue for inference
    result_queue = MagicMock()
    result_queue.get = MagicMock(
        side_effect=["Hello", " ", "world", "!", "<RKLLM_TASK_FINISHED>"]
    )
    manager.get_result = MagicMock(return_value=result_queue)
    manager.get_finished_inference_token = MagicMock(
        return_value="<RKLLM_TASK_FINISHED>"
    )

    return manager


@pytest.fixture
def sample_modelfile(tmp_path: Path) -> Path:
    """Create a sample Modelfile for testing."""
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()

    modelfile = model_dir / "Modelfile"
    modelfile.write_text(
        '''FROM="test-model.rkllm"

HUGGINGFACE_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM="You are a helpful assistant."

TEMPERATURE=0.7

NUM_CTX=4096

MAX_NEW_TOKENS=1024

TOP_K=40

TOP_P=0.9

REPEAT_PENALTY=1.1

FREQUENCY_PENALTY=0.0

PRESENCE_PENALTY=0.0

MIROSTAT=0

MIROSTAT_TAU=3.0

MIROSTAT_ETA=0.1

ENABLE_THINKING=False
'''
    )

    # Create dummy .rkllm file
    (model_dir / "test-model.rkllm").write_bytes(b"dummy model data")

    return model_dir


@pytest.fixture
def models_path(tmp_path: Path, sample_modelfile: Path) -> Path:
    """Create a models directory with sample model."""
    # sample_modelfile is already in tmp_path/test-model/
    return tmp_path


@pytest.fixture
def test_client(mock_worker_manager: MagicMock, models_path: Path) -> Generator:
    """Create a FastAPI test client with mocked dependencies."""
    if not RKLLM_AVAILABLE:
        pytest.skip("RKLLM native library not available (ARM-only)")

    from fastapi.testclient import TestClient

    # Patch config before importing app
    with patch("rkllama.config.get_path") as mock_get_path, \
         patch("rkllama.config.is_debug_mode", return_value=False), \
         patch("rkllama.config.get", return_value="0.7"):

        mock_get_path.return_value = str(models_path)

        from rkllama.server.app import create_app

        app = create_app()
        app.state.worker_manager = mock_worker_manager

        with TestClient(app) as client:
            yield client


@pytest.fixture
def async_mock_worker_manager() -> MagicMock:
    """Async-compatible mock WorkerManager."""
    manager = MagicMock()
    manager.workers = {}
    manager.exists_model_loaded = MagicMock(return_value=False)
    manager.add_worker = MagicMock(return_value=True)
    manager.stop_worker = MagicMock(return_value=None)
    manager.stop_all = MagicMock(return_value=None)

    # Mock async result queue
    result_queue = AsyncMock()
    result_queue.get = AsyncMock(
        side_effect=["Hello", " ", "world", "!", "<RKLLM_TASK_FINISHED>"]
    )
    manager.get_result = MagicMock(return_value=result_queue)

    return manager
