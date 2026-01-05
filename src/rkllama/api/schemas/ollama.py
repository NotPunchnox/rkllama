"""Pydantic schemas for Ollama-compatible API endpoints."""

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

    model: str = Field(..., description="Model name")
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    stream: bool = Field(True, description="Stream response")
    format: str | dict[str, Any] | None = Field(None, description="Response format (json, etc.)")
    options: dict[str, Any] | None = Field(None, description="Model options")
    tools: list[Tool] | None = Field(None, description="Available tools for function calling")
    keep_alive: str | None = Field(None, description="Keep model loaded duration")
    system: str | None = Field(None, description="System prompt override")
    template: str | None = Field(None, description="Chat template override")
    think: bool | None = Field(None, alias="enable_thinking", description="Enable thinking mode")

    model_config = {"populate_by_name": True}


class ChatResponseMessage(BaseModel):
    """Response message in chat completion."""

    role: str = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] | None = None


class ChatResponse(BaseModel):
    """Response schema for /api/chat endpoint (non-streaming)."""

    model: str
    created_at: str
    message: ChatResponseMessage
    done: bool = True
    done_reason: str = "stop"
    total_duration: int = 0
    load_duration: int = 100000000
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


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

    model: str = Field(..., description="Model name")
    prompt: str = Field(..., description="Input prompt")
    stream: bool = Field(True, description="Stream response")
    raw: bool = Field(False, description="Skip chat template, use raw prompt")
    format: str | dict[str, Any] | None = Field(None, description="Response format")
    options: dict[str, Any] | None = Field(None, description="Model options")
    system: str | None = Field(None, description="System prompt")
    template: str | None = Field(None, description="Chat template override")
    context: list[int] | None = Field(None, description="Context from previous response")
    images: list[str] | None = Field(None, description="Base64-encoded images")
    keep_alive: str | None = Field(None, description="Keep model loaded duration")
    think: bool | None = Field(None, alias="enable_thinking", description="Enable thinking mode")

    model_config = {"populate_by_name": True}


class GenerateResponse(BaseModel):
    """Response schema for /api/generate endpoint (non-streaming)."""

    model: str
    created_at: str
    response: str
    done: bool = True
    done_reason: str = "stop"
    context: list[int] = Field(default_factory=list)
    total_duration: int = 0
    load_duration: int = 100000000
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


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

    model: str = Field(..., description="Model name")
    input: str | list[str] = Field(..., alias="prompt", description="Text to embed")
    truncate: bool = Field(True, description="Truncate input to fit context")
    keep_alive: str | None = Field(None, description="Keep model loaded duration")
    options: dict[str, Any] | None = Field(None, description="Model options")

    model_config = {"populate_by_name": True}


class EmbeddingsResponse(BaseModel):
    """Response schema for /api/embeddings endpoint."""

    model: str
    embeddings: list[list[float]]
    total_duration: int = 0
    load_duration: int = 100000000
    prompt_eval_count: int = 0


# ============================================================================
# Model Management Schemas
# ============================================================================


class ShowRequest(BaseModel):
    """Request schema for /api/show endpoint."""

    model: str = Field(..., alias="name", description="Model name")
    verbose: bool = Field(False, description="Include verbose details")

    model_config = {"populate_by_name": True}


class ShowResponse(BaseModel):
    """Response schema for /api/show endpoint."""

    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: ModelDetails = Field(default_factory=ModelDetails)
    model_info: dict[str, Any] = Field(default_factory=dict)
    license: str | None = None
    system: str | None = None


class CreateRequest(BaseModel):
    """Request schema for /api/create endpoint."""

    model: str = Field(..., alias="name", description="Model name")
    modelfile: str = Field("", description="Modelfile content")
    stream: bool = Field(False, description="Stream progress")
    path: str | None = Field(None, description="Path to Modelfile (alternative to content)")

    model_config = {"populate_by_name": True}


class CreateResponse(BaseModel):
    """Response schema for /api/create endpoint."""

    status: str = "success"
    model: str | None = None


class DeleteRequest(BaseModel):
    """Request schema for /api/delete endpoint."""

    model: str = Field(..., alias="name", description="Model name to delete")

    model_config = {"populate_by_name": True}


class DeleteResponse(BaseModel):
    """Response schema for /api/delete endpoint."""

    status: str = "success"
    message: str | None = None


# ============================================================================
# Load/Unload Schemas (NEW)
# ============================================================================


class LoadRequest(BaseModel):
    """Request schema for /api/load endpoint."""

    model: str = Field(..., alias="name", description="Model name to load")
    options: dict[str, Any] | None = Field(None, description="Model options")
    keep_alive: str | None = Field(None, description="Keep model loaded duration")

    model_config = {"populate_by_name": True}


class LoadResponse(BaseModel):
    """Response schema for /api/load endpoint."""

    status: str = "success"
    message: str | None = None
    model: str | None = None


class UnloadRequest(BaseModel):
    """Request schema for /api/unload endpoint."""

    model: str = Field(..., alias="name", description="Model name to unload")

    model_config = {"populate_by_name": True}


class UnloadResponse(BaseModel):
    """Response schema for /api/unload endpoint."""

    status: str = "success"
    message: str | None = None


# ============================================================================
# Pull Schemas
# ============================================================================


class PullRequest(BaseModel):
    """Request schema for /api/pull endpoint."""

    model: str = Field(..., alias="name", description="Model name or HuggingFace path")
    insecure: bool = Field(False, description="Allow insecure connections")
    stream: bool = Field(True, description="Stream progress")

    model_config = {"populate_by_name": True}


class PullProgress(BaseModel):
    """Progress update during model pull."""

    status: str
    digest: str | None = None
    total: int | None = None
    completed: int | None = None


class PullResponse(BaseModel):
    """Final response for /api/pull endpoint."""

    status: str = "success"
    model: str | None = None


# ============================================================================
# List/Status Schemas
# ============================================================================


class ModelTag(BaseModel):
    """Model info for /api/tags response."""

    name: str
    model: str
    size: int = 0
    digest: str = ""
    details: ModelDetails = Field(default_factory=ModelDetails)
    modified_at: str = ""


class TagsResponse(BaseModel):
    """Response schema for /api/tags endpoint."""

    models: list[ModelTag] = Field(default_factory=list)


class RunningModel(BaseModel):
    """Running model info for /api/ps response."""

    name: str
    model: str
    size: int = 0
    digest: str = ""
    details: ModelDetails = Field(default_factory=ModelDetails)
    expires_at: str = ""
    loaded_at: str = ""
    base_domain_id: int = 1
    last_call: str = ""
    size_vram: int | None = None


class PsResponse(BaseModel):
    """Response schema for /api/ps endpoint."""

    models: list[RunningModel] = Field(default_factory=list)


class VersionResponse(BaseModel):
    """Response schema for /api/version endpoint."""

    version: str
