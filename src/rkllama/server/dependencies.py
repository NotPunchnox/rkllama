"""FastAPI dependencies for RKLlama server."""

import asyncio
from typing import Any

from fastapi import Depends, HTTPException, Request

import rkllama.config
from rkllama.api.worker import WorkerManager


def get_worker_manager(request: Request) -> WorkerManager:
    """Get the WorkerManager instance from app state."""
    return request.app.state.worker_manager


def get_config() -> Any:
    """Get the config module."""
    return rkllama.config


def get_models_path() -> str:
    """Get the models directory path."""
    return rkllama.config.get_path("models")


def get_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return rkllama.config.is_debug_mode()


class RequestLock:
    """Context manager for request locking to ensure sequential model access."""

    _lock = asyncio.Lock()

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


async def get_request_lock() -> RequestLock:
    """Get a request lock for sequential model access."""
    return RequestLock()


def require_model_loaded(
    model_name: str,
    worker_manager: WorkerManager = Depends(get_worker_manager),
) -> None:
    """Dependency that checks if a model is loaded."""
    if not worker_manager.exists_model_loaded(model_name):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not loaded. Please load the model first.",
        )


def get_app_state(request: Request) -> dict:
    """Get common app state values."""
    return {
        "worker_manager": request.app.state.worker_manager,
        "model_id": request.app.state.model_id,
        "system": request.app.state.system,
        "model_config": request.app.state.model_config,
        "generation_complete": request.app.state.generation_complete,
        "stream_stats": request.app.state.stream_stats,
    }
