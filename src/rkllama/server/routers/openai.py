"""OpenAI API compatible routes."""

import datetime
import logging
import os
import random
import re
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import rkllama.config
from rkllama.api.format_utils import openai_to_ollama_chat_request, openai_to_ollama_generate_request, strtobool
from rkllama.api.model_utils import get_model_full_options, get_property_modelfile
from rkllama.api.schemas.openai import (
    OpenAIChatRequest,
    OpenAICompletionRequest,
    OpenAIEmbeddingRequest,
    OpenAIImageRequest,
    OpenAISpeechRequest,
)
from rkllama.api.worker import WorkerManager
from rkllama.server.dependencies import get_debug_mode, get_models_path, get_worker_manager
from rkllama.server.routers.native import load_model

logger = logging.getLogger("rkllama.server.openai")

router = APIRouter()


def strip_namespace(model_name: str) -> str:
    """Remove namespace from model name."""
    match = re.search(r"/(.*)", model_name)
    return match.group(1) if match else model_name


@router.get("/models")
async def list_openai_models(models_path: str = Depends(get_models_path)) -> dict:
    """List models in OpenAI format."""
    if not os.path.exists(models_path):
        return {"object": "list", "data": []}

    models = []
    for subdir in os.listdir(models_path):
        subdir_path = os.path.join(models_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                # Include RKLLM, RKNN (SD), and Piper models
                if file.endswith(".rkllm") or file.endswith(".rknn") or file == "unet":
                    models.append(
                        {
                            "id": subdir,
                            "object": "model",
                            "created": int(
                                datetime.datetime.fromtimestamp(
                                    os.path.getmtime(os.path.join(subdir_path, file))
                                ).timestamp()
                            ),
                            "owned_by": "rkllama",
                        }
                    )
                    break

    return {"object": "list", "data": models}


@router.get("/models/{model_name}")
async def get_openai_model(model_name: str, models_path: str = Depends(get_models_path)) -> dict:
    """Get a specific model in OpenAI format."""
    if not os.path.exists(models_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    for subdir in os.listdir(models_path):
        subdir_path = os.path.join(models_path, subdir)
        if os.path.isdir(subdir_path) and subdir == model_name:
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm") or file == "unet":
                    return {
                        "id": subdir,
                        "object": "model",
                        "created": int(
                            datetime.datetime.fromtimestamp(
                                os.path.getmtime(os.path.join(subdir_path, file))
                            ).timestamp()
                        ),
                        "owned_by": "rkllama",
                    }

    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


@router.post("/completions")
async def completions_openai(
    request: OpenAICompletionRequest,
    wm: WorkerManager = Depends(get_worker_manager),
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> Any:
    """Generate completions (OpenAI-compatible)."""
    model_name = strip_namespace(request.model)

    if debug:
        logger.debug(f"OpenAI completions request for model: {model_name}")

    # Convert to Ollama format
    ollama_data = openai_to_ollama_generate_request(request.model_dump())

    # Get model options
    options = get_model_full_options(model_name, models_path, {})

    # Get thinking setting
    enable_thinking = None
    model_thinking_enabled = get_property_modelfile(model_name, "ENABLE_THINKING", models_path)
    enable_thinking = strtobool(model_thinking_enabled) if model_thinking_enabled else False

    # Load model if needed
    if not wm.exists_model_loaded(model_name):
        _, error = load_model(model_name, wm, request_options=options)
        if error:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {error}")

    from rkllama.api.server_utils import GenerateEndpointHandler

    return GenerateEndpointHandler.handle_request(
        model_name=model_name,
        prompt=ollama_data.get("prompt", ""),
        system=ollama_data.get("system", ""),
        stream=ollama_data.get("stream", False),
        format_spec=None,
        options=options,
        enable_thinking=enable_thinking,
        is_openai_request=True,
        images=None,
    )


@router.post("/chat/completions")
async def chat_completions_openai(
    request: OpenAIChatRequest,
    wm: WorkerManager = Depends(get_worker_manager),
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> Any:
    """Chat completions (OpenAI-compatible)."""
    model_name = strip_namespace(request.model)

    if debug:
        logger.debug(f"OpenAI chat request for model: {model_name}")

    # Convert to Ollama format
    ollama_data = openai_to_ollama_chat_request(request.model_dump())

    # Extract messages and system
    messages = ollama_data.get("messages", [])
    system = ollama_data.get("system", "")

    # Extract system from messages
    filtered_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system = msg.get("content", "")
        else:
            filtered_messages.append(msg)

    # Get model options
    options = get_model_full_options(model_name, models_path, {})

    # Get thinking setting
    enable_thinking = None
    model_thinking_enabled = get_property_modelfile(model_name, "ENABLE_THINKING", models_path)
    enable_thinking = strtobool(model_thinking_enabled) if model_thinking_enabled else False

    # Load model if needed
    if not wm.exists_model_loaded(model_name):
        _, error = load_model(model_name, wm, request_options=options)
        if error:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {error}")

    from rkllama.api.server_utils import ChatEndpointHandler

    return ChatEndpointHandler.handle_request(
        model_name=model_name,
        messages=filtered_messages,
        system=system,
        stream=ollama_data.get("stream", False),
        format_spec=None,
        options=options,
        tools=ollama_data.get("tools"),
        enable_thinking=enable_thinking,
        is_openai_request=True,
        images=None,
    )


@router.post("/embeddings")
async def embeddings_openai(
    request: OpenAIEmbeddingRequest,
    wm: WorkerManager = Depends(get_worker_manager),
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> Any:
    """Generate embeddings (OpenAI-compatible)."""
    model_name = strip_namespace(request.model)

    if debug:
        logger.debug(f"OpenAI embedding request for model: {model_name}")

    options = get_model_full_options(model_name, models_path, {})

    # Load model if needed
    if not wm.exists_model_loaded(model_name):
        _, error = load_model(model_name, wm, request_options=options)
        if error:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {error}")

    from rkllama.api.server_utils import EmbedEndpointHandler

    return EmbedEndpointHandler.handle_request(
        model_name=model_name,
        input_text=request.input,
        truncate=True,
        keep_alive=False,
        options=options,
        is_openai_request=True,
    )


@router.post("/images/generations")
async def generate_image_openai(
    request: OpenAIImageRequest,
    debug: bool = Depends(get_debug_mode),
) -> Any:
    """Generate images (OpenAI-compatible)."""
    model_name = strip_namespace(request.model) if request.model else None

    if debug:
        logger.debug(f"OpenAI image generation request: {request.prompt[:50]}...")

    from rkllama.api.server_utils import GenerateImageEndpointHandler

    return GenerateImageEndpointHandler.handle_request(
        model_name=model_name,
        prompt=request.prompt,
        stream=False,
        size=request.size,
        response_format=request.response_format,
        output_format="png",
        num_images=request.n,
        seed=random.randint(1, 99),
        num_inference_steps=4,
        guidance_scale=7.5,
    )


@router.get("/files/{model_name}/images/{file_name}")
async def get_generated_image(model_name: str, file_name: str) -> FileResponse:
    """Get a generated image file."""
    model_dir = os.path.join(rkllama.config.get_path("models"), model_name)
    file_path = os.path.join(model_dir, "images", file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path, filename=file_name)


@router.post("/audio/speech")
async def generate_speech_openai(
    request: OpenAISpeechRequest,
    debug: bool = Depends(get_debug_mode),
) -> Any:
    """Generate speech (OpenAI-compatible)."""
    model_name = strip_namespace(request.model) if request.model else None

    # Calculate Piper length_scale from speed
    length_scale = 1 / request.speed if request.speed else None

    if debug:
        logger.debug(f"OpenAI speech request for model: {model_name}")

    from rkllama.api.server_utils import GenerateSpeechEndpointHandler

    return GenerateSpeechEndpointHandler.handle_request(
        model_name=model_name,
        input=request.input,
        voice=request.voice,
        response_format=request.response_format,
        stream_format="audio",
        volume=None,
        length_scale=length_scale,
        noise_scale=None,
        noise_w_scale=None,
        normalize_audio=None,
    )


@router.post("/audio/transcriptions")
async def generate_transcriptions_openai(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    response_format: str = Form("text"),
    stream: str = Form("false"),
    debug: bool = Depends(get_debug_mode),
) -> Any:
    """Generate transcriptions (OpenAI-compatible)."""
    model_name = strip_namespace(model) if model else None
    stream_bool = strtobool(stream) if stream else False

    if debug:
        logger.debug(f"OpenAI transcription request for model: {model_name}")

    from rkllama.api.server_utils import GenerateTranscriptionsEndpointHandler

    return GenerateTranscriptionsEndpointHandler.handle_request(
        model_name=model_name,
        file=file,
        language=language,
        response_format=response_format,
        stream=stream_bool,
    )
