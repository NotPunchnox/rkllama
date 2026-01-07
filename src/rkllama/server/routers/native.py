"""Native RKLlama API routes."""

import os
import shutil
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

import rkllama.config
from rkllama.api.model_utils import get_model_full_options
from rkllama.api.schemas import (
    NativeDeleteRequest,
    NativeDeleteResponse,
    NativeLoadRequest,
    NativeLoadResponse,
    NativeModelsResponse,
    NativeUnloadAllResponse,
    NativeUnloadRequest,
    NativeUnloadResponse,
    PullRequest,
)
from rkllama.api.worker import WorkerManager
from rkllama.logging import get_logger
from rkllama.server.dependencies import get_debug_mode, get_models_path, get_worker_manager

logger = get_logger("rkllama.server.native")

router = APIRouter()


def create_modelfile(huggingface_path: str, from_value: str, model_name: str, system: str = "") -> None:
    """Create a Modelfile for a model."""
    struct_modelfile = f"""
FROM="{from_value}"

HUGGINGFACE_PATH="{huggingface_path}"

SYSTEM="{system}"

TEMPERATURE={rkllama.config.get("model", "default_temperature")}

ENABLE_THINKING={rkllama.config.get("model", "default_enable_thinking")}

NUM_CTX={rkllama.config.get("model", "default_num_ctx")}

MAX_NEW_TOKENS={rkllama.config.get("model", "default_max_new_tokens")}

TOP_K={rkllama.config.get("model", "default_top_k")}

TOP_P={rkllama.config.get("model", "default_top_p")}

REPEAT_PENALTY={rkllama.config.get("model", "default_repeat_penalty")}

FREQUENCY_PENALTY={rkllama.config.get("model", "default_frequency_penalty")}

PRESENCE_PENALTY={rkllama.config.get("model", "default_presence_penalty")}

MIROSTAT={rkllama.config.get("model", "default_mirostat")}

MIROSTAT_TAU={rkllama.config.get("model", "default_mirostat_tau")}

MIROSTAT_ETA={rkllama.config.get("model", "default_mirostat_eta")}


"""
    path = os.path.join(rkllama.config.get_path("models"), model_name)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "Modelfile"), "w") as f:
        f.write(struct_modelfile)


def load_model(
    model_name: str,
    worker_manager: WorkerManager,
    huggingface_path: str | None = None,
    system: str = "",
    from_value: str | None = None,
    request_options: dict | None = None,
) -> tuple[None, str | None]:
    """Load a model into memory."""
    from dotenv import load_dotenv

    model_dir = os.path.join(rkllama.config.get_path("models"), model_name)

    if not os.path.exists(model_dir):
        return None, f"Model directory '{model_name}' not found."

    modelfile_path = os.path.join(model_dir, "Modelfile")
    if not os.path.exists(modelfile_path) and (huggingface_path is None and from_value is None):
        return None, f"Modelfile not found in '{model_name}' directory."
    elif huggingface_path is not None and from_value is not None:
        create_modelfile(huggingface_path=huggingface_path, from_value=from_value, system=system, model_name=model_name)
        time.sleep(0.1)

    # Load modelfile
    load_dotenv(modelfile_path, override=True)

    from_value = os.getenv("FROM")
    huggingface_path = os.getenv("HUGGINGFACE_PATH")

    if not from_value or not huggingface_path:
        return None, "FROM or HUGGINGFACE_PATH not defined in Modelfile."

    # Get model parameters if not provided
    if not request_options:
        request_options = get_model_full_options(model_name, rkllama.config.get_path("models"), request_options)

    # Load model into memory
    model_loaded = worker_manager.add_worker(
        model_name, os.path.join(model_dir, from_value), model_dir, options=request_options
    )

    if not model_loaded:
        return None, f"Unexpected Error loading the model {model_name} into memory."
    return None, None


def unload_model(model_name: str, worker_manager: WorkerManager) -> None:
    """Release a model from memory."""
    worker_manager.stop_worker(model_name)


@router.get("/models")
async def list_models(models_path: str = Depends(get_models_path)) -> NativeModelsResponse:
    """List available models."""
    if not os.path.exists(models_path):
        raise HTTPException(status_code=500, detail=f"Models directory {models_path} not found.")

    # Move loose .rkllm files into directories
    direct_models = [f for f in os.listdir(models_path) if f.endswith(".rkllm")]
    for model in direct_models:
        model_name = os.path.splitext(model)[0]
        model_dir = os.path.join(models_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        shutil.move(os.path.join(models_path, model), os.path.join(model_dir, model))

    # Find model directories
    model_dirs = []
    for subdir in os.listdir(models_path):
        subdir_path = os.path.join(models_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    model_dirs.append(subdir)
                    break

    return NativeModelsResponse(models=model_dirs)


@router.delete("/rm")
async def rm_model(
    request: NativeDeleteRequest,
    wm: WorkerManager = Depends(get_worker_manager),
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> NativeDeleteResponse:
    """Delete a model."""
    model_path = os.path.join(models_path, request.model)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model directory for '{request.model}' not found")

    # Unload if loaded
    if wm.exists_model_loaded(request.model):
        if debug:
            logger.debug("Unloading model before deletion", model=request.model)
        unload_model(request.model, wm)

    try:
        if debug:
            logger.debug("Deleting model directory", path=model_path)
        shutil.rmtree(model_path)
        return NativeDeleteResponse(message="The model has been successfully deleted!")
    except Exception as e:
        logger.error("Failed to delete model", model=request.model, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")


@router.post("/pull")
async def pull_model(
    request: PullRequest,
    models_path: str = Depends(get_models_path),
) -> StreamingResponse:
    """Pull a model from various sources (HuggingFace, URL, S3)."""
    import json

    from rkllama.pull.base import PullSource as BasePullSource
    from rkllama.pull.base import get_handler

    async def generate_progress():
        model = request.model
        source_type = request.source.value.lower()
        model_name = request.model_name

        # Determine source type
        try:
            if source_type == "url" or model.startswith(("http://", "https://")):
                # Check if it's an S3 URL
                if "s3" in model and "amazonaws.com" in model:
                    pull_source = BasePullSource.S3
                elif model.startswith(("http://", "https://")):
                    pull_source = BasePullSource.URL
                else:
                    pull_source = BasePullSource.HUGGINGFACE
            elif source_type == "s3" or model.startswith("s3://"):
                pull_source = BasePullSource.S3
            else:
                pull_source = BasePullSource.HUGGINGFACE

            handler = get_handler(pull_source)

            async for progress in handler.pull(model, model_name, models_path):
                yield json.dumps(progress.to_dict()) + "\n"

        except Exception as e:
            yield json.dumps({"status": "error", "error": str(e)}) + "\n"

    return StreamingResponse(generate_progress(), media_type="application/x-ndjson")


@router.post("/load_model")
async def load_model_route(
    request: NativeLoadRequest,
    wm: WorkerManager = Depends(get_worker_manager),
) -> NativeLoadResponse:
    """Load a model into the NPU."""
    if wm.exists_model_loaded(request.model_name):
        return NativeLoadResponse(error="A model is already loaded. Nothing to do.")

    if request.from_value or request.huggingface_path:
        _, error = load_model(
            request.model_name, wm, from_value=request.from_value, huggingface_path=request.huggingface_path
        )
    else:
        _, error = load_model(request.model_name, wm)

    if error:
        raise HTTPException(status_code=400, detail=error)

    return NativeLoadResponse(message=f"Model {request.model_name} loaded successfully.")


@router.post("/unload_model")
async def unload_model_route(
    request: NativeUnloadRequest,
    wm: WorkerManager = Depends(get_worker_manager),
) -> NativeUnloadResponse:
    """Unload a model from the NPU."""
    if not wm.exists_model_loaded(request.model_name):
        raise HTTPException(status_code=400, detail=f"No model {request.model_name} is currently loaded.")

    unload_model(request.model_name, wm)
    return NativeUnloadResponse(message=f"Model {request.model_name} successfully unloaded!")


@router.post("/unload_models")
async def unload_models_route(wm: WorkerManager = Depends(get_worker_manager)) -> NativeUnloadAllResponse:
    """Unload all models."""
    wm.stop_all()
    return NativeUnloadAllResponse(message="Models successfully unloaded!")
