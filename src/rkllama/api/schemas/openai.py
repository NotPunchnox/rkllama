"""Pydantic schemas for OpenAI-compatible API endpoints."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Chat Completions Schemas
# ============================================================================


class OpenAIToolCall(BaseModel):
    """Tool call in OpenAI format."""

    id: str
    type: str = "function"
    function: dict[str, Any]


class OpenAIChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None


class OpenAITool(BaseModel):
    """Tool definition in OpenAI format."""

    type: str = "function"
    function: dict[str, Any]


class OpenAIChatRequest(BaseModel):
    """Request schema for /v1/chat/completions endpoint."""

    model: str
    messages: list[OpenAIChatMessage]
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, description="Number of completions")
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(None, alias="max_completion_tokens")
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[OpenAITool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    seed: int | None = None

    model_config = {"populate_by_name": True}


class OpenAIChatChoice(BaseModel):
    """Choice in chat completion response."""

    index: int = 0
    message: OpenAIChatMessage
    finish_reason: str | None = "stop"
    logprobs: Any | None = None


class OpenAIUsage(BaseModel):
    """Token usage in OpenAI format."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    """Response schema for /v1/chat/completions endpoint."""

    id: str = Field(default="chatcmpl-rkllama")
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatChoice]
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)
    system_fingerprint: str | None = None


class OpenAIChatStreamDelta(BaseModel):
    """Delta in streaming chat completion."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAIChatStreamChoice(BaseModel):
    """Choice in streaming chat completion."""

    index: int = 0
    delta: OpenAIChatStreamDelta
    finish_reason: str | None = None
    logprobs: Any | None = None


class OpenAIChatStreamChunk(BaseModel):
    """Streaming chunk for /v1/chat/completions endpoint."""

    id: str = Field(default="chatcmpl-rkllama")
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIChatStreamChoice]
    system_fingerprint: str | None = None


# ============================================================================
# Completions Schemas
# ============================================================================


class OpenAICompletionRequest(BaseModel):
    """Request schema for /v1/completions endpoint."""

    model: str
    prompt: str | list[str]
    suffix: str | None = None
    max_tokens: int | None = Field(None, alias="max_completion_tokens")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    n: int = Field(1, ge=1)
    stream: bool = False
    logprobs: int | None = None
    echo: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    best_of: int | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    seed: int | None = None

    model_config = {"populate_by_name": True}


class OpenAICompletionChoice(BaseModel):
    """Choice in completion response."""

    text: str
    index: int = 0
    logprobs: Any | None = None
    finish_reason: str | None = "stop"


class OpenAICompletionResponse(BaseModel):
    """Response schema for /v1/completions endpoint."""

    id: str = Field(default="cmpl-rkllama")
    object: str = "text_completion"
    created: int
    model: str
    choices: list[OpenAICompletionChoice]
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)
    system_fingerprint: str | None = None


# ============================================================================
# Embeddings Schemas
# ============================================================================


class OpenAIEmbeddingRequest(BaseModel):
    """Request schema for /v1/embeddings endpoint."""

    model: str
    input: str | list[str]
    encoding_format: str = Field("float", description="Encoding format (float or base64)")
    dimensions: int | None = None
    user: str | None = None


class OpenAIEmbeddingData(BaseModel):
    """Embedding data in response."""

    object: str = "embedding"
    embedding: list[float]
    index: int = 0


class OpenAIEmbeddingResponse(BaseModel):
    """Response schema for /v1/embeddings endpoint."""

    object: str = "list"
    data: list[OpenAIEmbeddingData]
    model: str
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)


# ============================================================================
# Models Schemas
# ============================================================================


class OpenAIModel(BaseModel):
    """Model info in OpenAI format."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "rkllama"
    permission: list[dict[str, Any]] = Field(default_factory=list)
    root: str | None = None
    parent: str | None = None


class OpenAIModelList(BaseModel):
    """Response schema for /v1/models endpoint."""

    object: str = "list"
    data: list[OpenAIModel]


# ============================================================================
# Image Generation Schemas
# ============================================================================


class OpenAIImageRequest(BaseModel):
    """Request schema for /v1/images/generations endpoint."""

    model: str | None = None
    prompt: str
    n: int = Field(1, ge=1, le=10)
    size: str = Field("512x512", description="Image size (e.g., 512x512, 1024x1024)")
    quality: str = Field("standard", description="Image quality")
    response_format: str = Field("b64_json", description="Response format (b64_json or url)")
    style: str | None = None
    user: str | None = None
    # RKNN-specific
    seed: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    output_format: str = Field("png", description="Output format (png, jpeg)")


class OpenAIImageData(BaseModel):
    """Image data in response."""

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class OpenAIImageResponse(BaseModel):
    """Response schema for /v1/images/generations endpoint."""

    created: int
    data: list[OpenAIImageData]


# ============================================================================
# Audio Schemas
# ============================================================================


class OpenAISpeechRequest(BaseModel):
    """Request schema for /v1/audio/speech endpoint."""

    model: str
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field("default", description="Voice to use")
    response_format: str = Field("mp3", description="Audio format")
    speed: float = Field(1.0, ge=0.25, le=4.0)
    # Piper-specific
    volume: float | None = None
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    normalize_audio: bool | None = None


class OpenAITranscriptionRequest(BaseModel):
    """Request schema for /v1/audio/transcriptions endpoint."""

    model: str
    # file is handled separately as multipart form data
    language: str | None = None
    prompt: str | None = None
    response_format: str = Field("json", description="Response format")
    temperature: float | None = Field(None, ge=0.0, le=1.0)
    timestamp_granularities: list[str] | None = None


class OpenAITranscriptionResponse(BaseModel):
    """Response schema for /v1/audio/transcriptions endpoint."""

    text: str
    task: str | None = None
    language: str | None = None
    duration: float | None = None
    words: list[dict[str, Any]] | None = None
    segments: list[dict[str, Any]] | None = None
