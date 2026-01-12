"""Ollama API compatible routes."""

import datetime
import os
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

import rkllama.config
from rkllama.api.format_utils import strtobool
from rkllama.api.model_utils import (
    extract_model_details,
    find_rkllm_model_name,
    get_huggingface_model_info,
    get_model_full_options,
    get_property_modelfile,
)
from rkllama.api.schemas.ollama import (
    ChatRequest,
    CreateRequest,
    CreateResponse,
    DeleteRequest,
    DeleteResponse,
    EmbeddingsRequest,
    GenerateRequest,
    LoadRequest,
    LoadResponse,
    PsResponse,
    PullRequest,
    ShowRequest,
    ShowResponse,
    TagsResponse,
    UnloadRequest,
    UnloadResponse,
    VersionResponse,
)
from rkllama.api.worker import WorkerManager
from rkllama.logging import get_logger
from rkllama.server.dependencies import get_debug_mode, get_models_path, get_worker_manager
from rkllama.server.routers.native import load_model, unload_model

logger = get_logger("rkllama.server.ollama")

router = APIRouter()


def strip_namespace(model_name: str) -> str:
    """Remove namespace from model name (e.g., 'namespace/model' -> 'model')."""
    match = re.search(r"/(.*)", model_name)
    return match.group(1) if match else model_name


@router.get("/tags", response_model=TagsResponse)
async def list_ollama_models(models_path: str = Depends(get_models_path)) -> TagsResponse:
    """List models in Ollama format."""
    if not os.path.exists(models_path):
        return TagsResponse(models=[])

    models = []
    for subdir in os.listdir(models_path):
        subdir_path = os.path.join(models_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    size = os.path.getsize(os.path.join(subdir_path, file))
                    model_details = extract_model_details(file)

                    models.append(
                        {
                            "name": subdir,
                            "model": subdir,
                            "modified_at": datetime.datetime.fromtimestamp(
                                os.path.getmtime(os.path.join(subdir_path, file))
                            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            "size": size,
                            "digest": "",
                            "details": {
                                "format": "rkllm",
                                "family": "llama",
                                "parameter_size": model_details.get("parameter_size", "Unknown"),
                                "quantization_level": model_details.get("quantization_level", "Unknown"),
                            },
                        }
                    )
                    break

    return TagsResponse(models=models)


@router.get("/ps", response_model=PsResponse)
async def get_current_models(
    wm: WorkerManager = Depends(get_worker_manager),
    models_path: str = Depends(get_models_path),
) -> PsResponse:
    """Get currently loaded models."""
    # Build model info map
    models_info = {}
    for subdir in os.listdir(models_path):
        subdir_path = os.path.join(models_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    size = os.path.getsize(os.path.join(subdir_path, file))
                    model_details = extract_model_details(file)

                    models_info[subdir] = {
                        "name": subdir,
                        "model": subdir,
                        "modified_at": datetime.datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(subdir_path, file))
                        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "size": size,
                        "digest": "",
                        "details": {
                            "format": "rkllm",
                            "family": "llama",
                            "parameter_size": model_details.get("parameter_size", "Unknown"),
                            "quantization_level": model_details.get("quantization_level", "Unknown"),
                        },
                    }
                    break

    # Get running models
    models_running = []
    for model in wm.workers:
        worker_model_info = wm.workers[model].worker_model_info
        if model in models_info:
            model_info = {
                "name": model,
                "model": model,
                "size": worker_model_info.size,
                "digest": models_info[model]["digest"],
                "details": {
                    "parent_model": "",
                    "format": models_info[model]["details"]["format"],
                    "family": models_info[model]["details"]["family"],
                    "families": [models_info[model]["details"]["family"]],
                    "parameter_size": models_info[model]["details"]["parameter_size"],
                    "quantization_level": models_info[model]["details"]["quantization_level"],
                },
                "expires_at": worker_model_info.expires_at.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "loaded_at": worker_model_info.loaded_at.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "base_domain_id": worker_model_info.base_domain_id,
                "last_call": worker_model_info.last_call.strftime("%Y-%m-%d %H:%M:%S.%f"),
            }
            models_running.append(model_info)

    return PsResponse(models=models_running)


@router.post("/show", response_model=ShowResponse)
async def show_model_info(
    request: ShowRequest,
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> ShowResponse:
    """Show detailed model information."""
    model_name = request.model
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing model name")

    model_name = strip_namespace(model_name)

    if debug:
        logger.debug("API show request", model=model_name)

    model_dir = os.path.join(models_path, model_name)
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    model_rkllm = find_rkllm_model_name(model_dir)

    # Read modelfile content
    modelfile_path = os.path.join(model_dir, "Modelfile")
    modelfile_content = ""
    system_prompt = ""
    template = "{{ .Prompt }}"
    license_text = ""
    huggingface_path = None

    if os.path.exists(modelfile_path):
        with open(modelfile_path) as f:
            modelfile_content = f.read()

            # Extract various fields
            system_match = re.search(r'SYSTEM="(.*?)"', modelfile_content, re.DOTALL)
            if system_match:
                system_prompt = system_match.group(1).strip()

            template_match = re.search(r'TEMPLATE="(.*?)"', modelfile_content, re.DOTALL)
            if template_match:
                template = template_match.group(1).strip()

            license_match = re.search(r'LICENSE="(.*?)"', modelfile_content, re.DOTALL)
            if license_match:
                license_text = license_match.group(1).strip()

            hf_path_match = re.search(r'HUGGINGFACE_PATH="(.*?)"', modelfile_content, re.DOTALL)
            if hf_path_match:
                huggingface_path = hf_path_match.group(1).strip()

    # Find .rkllm file
    model_file = None
    for file in os.listdir(model_dir):
        if file.endswith(".rkllm"):
            model_file = file
            break

    if not model_file:
        raise HTTPException(status_code=404, detail=f"Model file not found in '{model_name}' directory")

    file_path = os.path.join(model_dir, model_file)
    size = os.path.getsize(file_path)

    model_details = extract_model_details(model_rkllm)
    parameter_size = model_details.get("parameter_size", "Unknown")
    quantization_level = model_details.get("quantization_level", "Unknown")

    # Determine model family
    family = "llama"
    families = ["llama"]

    # Get HuggingFace metadata
    hf_metadata = get_huggingface_model_info(huggingface_path) if huggingface_path else None

    if hf_metadata:
        tags = hf_metadata.get("tags", [])
        if "qwen" in tags or "qwen2" in tags:
            family, families = "qwen2", ["qwen2"]
        elif "mistral" in tags:
            family, families = "mistral", ["mistral"]
        # ... additional family detection

    modified_at = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # Build response
    response = {
        "license": license_text or "Unknown",
        "modelfile": f"FROM {model_name}\n",
        "parameters": parameter_size,
        "template": template,
        "system": system_prompt,
        "name": model_name,
        "details": {
            "parent_model": huggingface_path or "",
            "format": "rkllm",
            "family": family,
            "families": families,
            "parameter_size": parameter_size,
            "quantization_level": quantization_level,
        },
        "size": size,
        "modified_at": modified_at,
    }

    return response


@router.post("/create", response_model=CreateResponse)
async def create_model(request: CreateRequest, models_path: str = Depends(get_models_path)) -> CreateResponse:
    """Create a new model from modelfile."""
    model_name = request.model
    modelfile = request.modelfile

    if not model_name:
        raise HTTPException(status_code=400, detail="Missing model name")

    model_name = strip_namespace(model_name)
    model_dir = os.path.join(models_path, model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "Modelfile"), "w") as f:
        f.write(modelfile)

    return CreateResponse(status="success", model=model_name)


@router.post("/pull")
async def pull_model_ollama(
    request: PullRequest,
    models_path: str = Depends(get_models_path),
) -> StreamingResponse:
    """Pull a model from HuggingFace, URL, or S3.

    Examples:
        - HuggingFace: `{"model": "owner/repo/model.rkllm", "model_name": "my-model"}`
        - URL: `{"model": "https://example.com/model.rkllm", "model_name": "my-model"}`
        - S3: `{"model": "s3://bucket/path/model.rkllm", "model_name": "my-model"}`
    """
    from rkllama.server.routers.native import pull_model

    if not request.model:
        raise HTTPException(status_code=400, detail="Missing model path")

    # Pass the request directly to native pull
    return await pull_model(request, models_path)


@router.delete("/delete", response_model=DeleteResponse)
async def delete_model_ollama(
    request: DeleteRequest,
    wm: WorkerManager = Depends(get_worker_manager),
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> DeleteResponse:
    """Delete a model (Ollama-compatible)."""
    if not request.model:
        raise HTTPException(status_code=400, detail="Missing model name")

    model_name = strip_namespace(request.model)
    model_path = os.path.join(models_path, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model directory for '{model_name}' not found")

    if wm.exists_model_loaded(model_name):
        if debug:
            logger.debug("Unloading model before deletion", model=model_name)
        unload_model(model_name, wm)

    try:
        import shutil

        shutil.rmtree(model_path)
        return DeleteResponse(status="success", message="The model has been successfully deleted!")
    except Exception as e:
        logger.error("Failed to delete model", model=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")


@router.post("/load", response_model=LoadResponse)
async def load_model_ollama(
    request: LoadRequest,
    wm: WorkerManager = Depends(get_worker_manager),
) -> LoadResponse:
    """Load a model (Ollama-compatible)."""
    model_name = strip_namespace(request.model)

    if wm.exists_model_loaded(model_name):
        return LoadResponse(status="success", message="Model already loaded", model=model_name)

    _, error = load_model(model_name, wm, request_options=request.options)
    if error:
        raise HTTPException(status_code=400, detail=error)

    return LoadResponse(status="success", model=model_name)


@router.post("/unload", response_model=UnloadResponse)
async def unload_model_ollama(
    request: UnloadRequest,
    wm: WorkerManager = Depends(get_worker_manager),
) -> UnloadResponse:
    """Unload a model (Ollama-compatible)."""
    if rkllama.config.get("model", "disable_model_unloading"):
        raise HTTPException(
            status_code=400,
            detail="Model unloading is disabled to prevent NPU memory leaks. Restart pod to change models.",
        )

    model_name = strip_namespace(request.model)

    if not wm.exists_model_loaded(model_name):
        return UnloadResponse(status="success", message="Model not loaded")

    unload_model(model_name, wm)
    return UnloadResponse(status="success")


@router.post(
    "/generate",
    responses={
        200: {
            "description": "Generated response",
            "content": {
                "application/x-ndjson": {
                    "schema": {"$ref": "#/components/schemas/GenerateStreamChunk"},
                    "example": {"model": "qwen", "response": "Hello", "done": False},
                },
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/GenerateResponse"},
                },
            },
        }
    },
)
def generate_ollama(
    request: GenerateRequest,
    req: Request,
    wm: WorkerManager = Depends(get_worker_manager),
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> Any:
    """Generate text (Ollama-compatible).

    When `stream=true`, returns NDJSON stream (application/x-ndjson).
    When `stream=false`, returns single JSON response (application/json).
    """
    model_name = strip_namespace(request.model)

    if debug:
        logger.debug("Generate request", model=model_name)

    # Get thinking setting from modelfile if not provided
    enable_thinking = request.think
    if enable_thinking is None:
        model_thinking_enabled = get_property_modelfile(model_name, "ENABLE_THINKING", models_path)
        enable_thinking = strtobool(model_thinking_enabled) if model_thinking_enabled else False

    # Get model options
    options = get_model_full_options(model_name, models_path, request.options or {})

    # Load model if needed
    if not wm.exists_model_loaded(model_name):
        _, error = load_model(model_name, wm, request_options=options)
        if error:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {error}")

    # Use existing handler
    from rkllama.api.server_utils import GenerateEndpointHandler

    return GenerateEndpointHandler.handle_request(
        model_name=model_name,
        prompt=request.prompt,
        system=request.system or "",
        stream=request.stream,
        format_spec=request.format,
        options=options,
        enable_thinking=enable_thinking,
        is_openai_request=False,
        images=request.images,
    )


@router.post(
    "/chat",
    responses={
        200: {
            "description": "Chat response",
            "content": {
                "application/x-ndjson": {
                    "schema": {"$ref": "#/components/schemas/ChatStreamChunk"},
                    "example": {
                        "model": "qwen",
                        "message": {"role": "assistant", "content": "Hello"},
                        "done": False,
                    },
                },
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ChatResponse"},
                },
            },
        }
    },
)
def chat_ollama(
    request: ChatRequest,
    req: Request,
    wm: WorkerManager = Depends(get_worker_manager),
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> Any:
    """Chat with model (Ollama-compatible).

    When `stream=true`, returns NDJSON stream (application/x-ndjson).
    When `stream=false`, returns single JSON response (application/json).
    """
    model_name = strip_namespace(request.model)

    if debug:
        logger.debug("Chat request", model=model_name)

    # Extract system message from messages
    system = request.system or ""
    messages = []
    images = []

    for msg in request.messages:
        if msg.role == "system":
            system = msg.content or ""
        else:
            messages.append({"role": msg.role, "content": msg.content})
            if msg.images:
                images.extend(msg.images)

    # Get thinking setting
    enable_thinking = request.think
    if enable_thinking is None:
        model_thinking_enabled = get_property_modelfile(model_name, "ENABLE_THINKING", models_path)
        enable_thinking = strtobool(model_thinking_enabled) if model_thinking_enabled else False

    # Get model options
    options = get_model_full_options(model_name, models_path, request.options or {})

    # Load model if needed
    if not wm.exists_model_loaded(model_name):
        _, error = load_model(model_name, wm, request_options=options)
        if error:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {error}")

    # Use existing handler
    from rkllama.api.server_utils import ChatEndpointHandler

    return ChatEndpointHandler.handle_request(
        model_name=model_name,
        messages=messages,
        system=system,
        stream=request.stream,
        format_spec=request.format,
        options=options,
        tools=request.tools,
        enable_thinking=enable_thinking,
        is_openai_request=False,
        images=images if images else None,
    )


@router.post("/embeddings")
@router.post("/embed")
def embeddings_ollama(
    request: EmbeddingsRequest,
    wm: WorkerManager = Depends(get_worker_manager),
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> Any:
    """Generate embeddings (Ollama-compatible)."""
    model_name = request.model
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing model name")

    model_name = strip_namespace(model_name)
    input_text = request.input

    if not input_text:
        raise HTTPException(status_code=400, detail="Missing input")

    truncate = request.truncate
    keep_alive = request.keep_alive
    options = get_model_full_options(model_name, models_path, request.options or {})

    # Load model if needed
    if not wm.exists_model_loaded(model_name):
        _, error = load_model(model_name, wm, request_options=options)
        if error:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {error}")

    from rkllama.api.server_utils import EmbedEndpointHandler

    return EmbedEndpointHandler.handle_request(
        model_name=model_name,
        input_text=input_text,
        truncate=truncate,
        keep_alive=keep_alive,
        options=options,
        is_openai_request=False,
    )


@router.get("/version", response_model=VersionResponse)
async def ollama_version() -> VersionResponse:
    """Return version for Ollama compatibility."""
    return VersionResponse(version="0.0.54")
