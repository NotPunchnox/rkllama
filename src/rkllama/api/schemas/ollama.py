"""Pydantic schemas for Ollama-compatible API endpoints."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import ModelDetails

# ============================================================================
# Chat Endpoint Schemas
# ============================================================================


class ToolCall(BaseModel):
    """Tool call from assistant."""

    id: str | None = None
    type: str = "function"
    function: dict[str, Any]


class ChatMessage(BaseModel):
    """Chat message for Ollama API."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    images: list[str] | None = Field(None, description="Base64-encoded images for multimodal")
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class Tool(BaseModel):
    """Tool definition for function calling."""

    type: str = "function"
    function: dict[str, Any]


class ChatRequest(BaseModel):
    """Request schema for /api/chat endpoint."""

    model: str = Field(..., description="Model name", examples=["qwen:7b"])
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    stream: bool = Field(True, description="Stream response")
    format: str | dict[str, Any] | None = Field(None, description="Response format (json, etc.)", examples=["json"])
    options: dict[str, Any] | None = Field(
        None,
        description="Model options",
        examples=[{"temperature": 0.7, "top_p": 0.9, "num_ctx": 4096}],
    )
    tools: list[Tool] | None = Field(None, description="Available tools for function calling")
    keep_alive: str | None = Field(None, description="Keep model loaded duration", examples=["5m"])
    system: str | None = Field(None, description="System prompt override", examples=["You are a helpful assistant."])
    template: str | None = Field(None, description="Chat template override")
    think: bool | None = Field(None, alias="enable_thinking", description="Enable thinking mode")

    model_config = {"populate_by_name": True}


class ChatResponseMessage(BaseModel):
    """Response message in chat completion."""

    role: str = Field("assistant", description="Message role", examples=["assistant"])
    content: str = Field("", description="Message content", examples=["Hello! How can I help you today?"])
    tool_calls: list[ToolCall] | None = None


class ChatResponse(BaseModel):
    """Response schema for /api/chat endpoint (non-streaming)."""

    model: str = Field(..., description="Model name", examples=["qwen:7b"])
    created_at: str = Field(..., description="Timestamp", examples=["2024-01-15T10:30:00.000Z"])
    message: ChatResponseMessage = Field(..., description="Response message")
    done: bool = Field(True, description="Whether generation is complete")
    done_reason: str = Field("stop", description="Reason for completion", examples=["stop", "length"])
    total_duration: int = Field(0, description="Total duration in nanoseconds")
    load_duration: int = Field(100000000, description="Model load duration in nanoseconds")
    prompt_eval_count: int = Field(0, description="Number of prompt tokens evaluated")
    prompt_eval_duration: int = Field(0, description="Prompt evaluation duration in nanoseconds")
    eval_count: int = Field(0, description="Number of tokens generated")
    eval_duration: int = Field(0, description="Generation duration in nanoseconds")


class ChatStreamChunk(BaseModel):
    """Streaming chunk for /api/chat endpoint."""

    model: str
    created_at: str
    message: ChatResponseMessage
    done: bool = False
    done_reason: str | None = None
    # Metrics only in final chunk
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


# ============================================================================
# Generate Endpoint Schemas
# ============================================================================


class GenerateRequest(BaseModel):
    """Request schema for /api/generate endpoint."""

    model: str = Field(..., description="Model name", examples=["qwen:7b"])
    prompt: str = Field(..., description="Input prompt", examples=["Why is the sky blue?"])
    stream: bool = Field(True, description="Stream response")
    raw: bool = Field(False, description="Skip chat template, use raw prompt")
    format: str | dict[str, Any] | None = Field(None, description="Response format", examples=["json"])
    options: dict[str, Any] | None = Field(
        None,
        description="Model options",
        examples=[{"temperature": 0.7, "top_p": 0.9}],
    )
    system: str | None = Field(None, description="System prompt", examples=["You are a helpful assistant."])
    template: str | None = Field(None, description="Chat template override")
    context: list[int] | None = Field(None, description="Context from previous response")
    images: list[str] | None = Field(None, description="Base64-encoded images")
    keep_alive: str | None = Field(None, description="Keep model loaded duration", examples=["5m"])
    think: bool | None = Field(None, alias="enable_thinking", description="Enable thinking mode")

    model_config = {"populate_by_name": True}


class GenerateResponse(BaseModel):
    """Response schema for /api/generate endpoint (non-streaming)."""

    model: str = Field(..., description="Model name", examples=["qwen:7b"])
    created_at: str = Field(..., description="Timestamp", examples=["2024-01-15T10:30:00.000Z"])
    response: str = Field(
        ..., description="Generated text", examples=["The sky appears blue due to Rayleigh scattering..."]
    )
    done: bool = Field(True, description="Whether generation is complete")
    done_reason: str = Field("stop", description="Reason for completion", examples=["stop", "length"])
    context: list[int] = Field(default_factory=list, description="Context for follow-up requests")
    total_duration: int = Field(0, description="Total duration in nanoseconds")
    load_duration: int = Field(100000000, description="Model load duration in nanoseconds")
    prompt_eval_count: int = Field(0, description="Number of prompt tokens evaluated")
    prompt_eval_duration: int = Field(0, description="Prompt evaluation duration in nanoseconds")
    eval_count: int = Field(0, description="Number of tokens generated")
    eval_duration: int = Field(0, description="Generation duration in nanoseconds")


class GenerateStreamChunk(BaseModel):
    """Streaming chunk for /api/generate endpoint."""

    model: str
    created_at: str
    response: str = ""
    done: bool = False
    done_reason: str | None = None
    context: list[int] | None = None
    # Metrics only in final chunk
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


# ============================================================================
# Embeddings Endpoint Schemas
# ============================================================================


class EmbeddingsRequest(BaseModel):
    """Request schema for /api/embeddings endpoint."""

    model: str = Field(..., description="Model name", examples=["qwen:7b"])
    input: str | list[str] = Field(..., alias="prompt", description="Text to embed", examples=["Hello, world!"])
    truncate: bool = Field(True, description="Truncate input to fit context")
    keep_alive: str | None = Field(None, description="Keep model loaded duration", examples=["5m"])
    options: dict[str, Any] | None = Field(None, description="Model options")

    model_config = {"populate_by_name": True}


class EmbeddingsResponse(BaseModel):
    """Response schema for /api/embeddings endpoint."""

    model: str = Field(..., description="Model name", examples=["qwen:7b"])
    embeddings: list[list[float]] = Field(..., description="Embedding vectors")
    total_duration: int = Field(0, description="Total duration in nanoseconds")
    load_duration: int = Field(100000000, description="Model load duration in nanoseconds")
    prompt_eval_count: int = Field(0, description="Number of prompt tokens evaluated")


# ============================================================================
# Model Management Schemas
# ============================================================================


class ShowRequest(BaseModel):
    """Request schema for /api/show endpoint."""

    model: str = Field(..., alias="name", description="Model name", examples=["qwen:7b"])
    verbose: bool = Field(False, description="Include verbose details")

    model_config = {"populate_by_name": True}


class ShowResponse(BaseModel):
    """Response schema for /api/show endpoint."""

    modelfile: str = Field("", description="Modelfile content")
    parameters: str = Field("", description="Model parameters")
    template: str = Field("", description="Chat template")
    details: ModelDetails = Field(default_factory=ModelDetails, description="Model details")
    model_info: dict[str, Any] = Field(default_factory=dict, description="Additional model info")
    license: str | None = Field(None, description="Model license")
    system: str | None = Field(None, description="System prompt", examples=["You are a helpful assistant."])


class CreateRequest(BaseModel):
    """Request schema for /api/create endpoint."""

    model: str = Field(..., alias="name", description="Model name", examples=["my-model:latest"])
    modelfile: str = Field(
        "", description="Modelfile content", examples=['FROM="/models/model.rkllm"\nSYSTEM="You are helpful."']
    )
    stream: bool = Field(False, description="Stream progress")
    path: str | None = Field(None, description="Path to Modelfile (alternative to content)")

    model_config = {"populate_by_name": True}


class CreateResponse(BaseModel):
    """Response schema for /api/create endpoint."""

    status: str = Field("success", description="Creation status", examples=["success"])
    model: str | None = Field(None, description="Created model name", examples=["my-model:latest"])


class DeleteRequest(BaseModel):
    """Request schema for /api/delete endpoint."""

    model: str = Field(..., alias="name", description="Model name to delete", examples=["qwen:7b"])

    model_config = {"populate_by_name": True}


class DeleteResponse(BaseModel):
    """Response schema for /api/delete endpoint."""

    status: str = Field("success", description="Deletion status", examples=["success"])
    message: str | None = Field(None, description="Status message", examples=["Model deleted successfully"])


# ============================================================================
# Load/Unload Schemas (NEW)
# ============================================================================


class LoadRequest(BaseModel):
    """Request schema for /api/load endpoint."""

    model: str = Field(..., alias="name", description="Model name to load", examples=["qwen:7b"])
    options: dict[str, Any] | None = Field(
        None,
        description="Model options",
        examples=[{"temperature": 0.7, "num_ctx": 4096}],
    )
    keep_alive: str | None = Field(None, description="Keep model loaded duration", examples=["5m", "1h"])

    model_config = {"populate_by_name": True}


class LoadResponse(BaseModel):
    """Response schema for /api/load endpoint."""

    status: str = Field("success", description="Load status", examples=["success"])
    message: str | None = Field(None, description="Status message", examples=["Model loaded successfully"])
    model: str | None = Field(None, description="Loaded model name", examples=["qwen:7b"])


class UnloadRequest(BaseModel):
    """Request schema for /api/unload endpoint."""

    model: str = Field(..., alias="name", description="Model name to unload", examples=["qwen:7b"])

    model_config = {"populate_by_name": True}


class UnloadResponse(BaseModel):
    """Response schema for /api/unload endpoint."""

    status: str = Field("success", description="Unload status", examples=["success"])
    message: str | None = Field(None, description="Status message", examples=["Model unloaded successfully"])


# ============================================================================
# Pull Schemas
# ============================================================================


class PullSource(str, Enum):
    """Source type for pulling models."""

    HUGGINGFACE = "huggingface"
    URL = "url"
    S3 = "s3"


class PullRequest(BaseModel):
    """Request schema for /api/pull endpoint.

    Examples:
        HuggingFace: {"model": "owner/repo/model.rkllm", "model_name": "my-model"}
        URL: {"model": "https://example.com/model.rkllm", "model_name": "my-model"}
        S3: {"model": "s3://bucket/path/model.rkllm", "model_name": "my-model"}
    """

    model: str = Field(
        ...,
        alias="name",
        description="Model source path. For HuggingFace: 'owner/repo/file.rkllm'. For URL: full https:// URL. For S3: 's3://bucket/path'",
        examples=["punchnox/TinyLlama-1.1B-rk3588/TinyLlama-1.1B-Chat-v1.0.rkllm"],
    )
    model_name: str | None = Field(
        None,
        description="Custom name to save the model as locally (e.g., 'tinyllama:1.1b'). If not provided, derived from filename.",
        examples=["tinyllama:1.1b", "qwen:7b"],
    )
    source: PullSource = Field(
        PullSource.HUGGINGFACE,
        description="Source type. Auto-detected from model path if not specified.",
    )
    stream: bool = Field(True, description="Stream progress updates")

    model_config = {"populate_by_name": True}


class PullProgress(BaseModel):
    """Progress update during model pull."""

    status: str = Field(..., description="Progress status", examples=["downloading", "verifying", "success"])
    digest: str | None = Field(None, description="File digest/hash")
    total: int | None = Field(None, description="Total bytes to download")
    completed: int | None = Field(None, description="Bytes downloaded so far")


class PullResponse(BaseModel):
    """Final response for /api/pull endpoint."""

    status: str = Field("success", description="Pull status", examples=["success"])
    model: str | None = Field(None, description="Pulled model name", examples=["qwen:7b"])


# ============================================================================
# List/Status Schemas
# ============================================================================


class ModelTag(BaseModel):
    """Model info for /api/tags response."""

    name: str = Field(..., description="Model name with tag", examples=["qwen:7b"])
    model: str = Field(..., description="Model identifier", examples=["qwen:7b"])
    size: int = Field(0, description="Model size in bytes")
    digest: str = Field("", description="Model digest/hash")
    details: ModelDetails = Field(default_factory=ModelDetails, description="Model details")
    modified_at: str = Field("", description="Last modification timestamp", examples=["2024-01-15T10:30:00.000Z"])


class TagsResponse(BaseModel):
    """Response schema for /api/tags endpoint."""

    models: list[ModelTag] = Field(default_factory=list, description="List of available models")


class RunningModel(BaseModel):
    """Running model info for /api/ps response."""

    name: str = Field(..., description="Model name", examples=["qwen:7b"])
    model: str = Field(..., description="Model identifier", examples=["qwen:7b"])
    size: int = Field(0, description="Model size in bytes")
    digest: str = Field("", description="Model digest/hash")
    details: ModelDetails = Field(default_factory=ModelDetails, description="Model details")
    expires_at: str = Field("", description="When model will be unloaded")
    loaded_at: str = Field("", description="When model was loaded", examples=["2024-01-15T10:30:00.000Z"])
    base_domain_id: int = Field(1, description="NPU domain ID")
    last_call: str = Field("", description="Last inference timestamp")
    size_vram: int | None = Field(None, description="VRAM usage in bytes")


class PsResponse(BaseModel):
    """Response schema for /api/ps endpoint."""

    models: list[RunningModel] = Field(default_factory=list, description="List of running models")


class VersionResponse(BaseModel):
    """Response schema for /api/version endpoint."""

    version: str = Field(..., description="RKLlama version", examples=["0.2.0"])
