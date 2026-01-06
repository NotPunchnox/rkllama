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

    model: str = Field(..., description="Model ID", examples=["qwen:7b"])
    messages: list[OpenAIChatMessage] = Field(..., description="List of chat messages")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature", examples=[0.7])
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling", examples=[0.9])
    n: int = Field(1, ge=1, description="Number of completions")
    stream: bool = Field(False, description="Stream response")
    stop: str | list[str] | None = Field(None, description="Stop sequences", examples=[["\\n", "END"]])
    max_tokens: int | None = Field(
        None, alias="max_completion_tokens", description="Max tokens to generate", examples=[1024]
    )
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: dict[str, float] | None = Field(None, description="Token logit biases")
    user: str | None = Field(None, description="User identifier")
    tools: list[OpenAITool] | None = Field(None, description="Available tools")
    tool_choice: str | dict[str, Any] | None = Field(None, description="Tool selection mode")
    seed: int | None = Field(None, description="Random seed for reproducibility")

    model_config = {"populate_by_name": True}


class OpenAIChatChoice(BaseModel):
    """Choice in chat completion response."""

    index: int = Field(0, description="Choice index")
    message: OpenAIChatMessage = Field(..., description="Response message")
    finish_reason: str | None = Field("stop", description="Finish reason", examples=["stop", "length", "tool_calls"])
    logprobs: Any | None = Field(None, description="Log probabilities")


class OpenAIUsage(BaseModel):
    """Token usage in OpenAI format."""

    prompt_tokens: int = Field(0, description="Tokens in prompt", examples=[25])
    completion_tokens: int = Field(0, description="Tokens in completion", examples=[100])
    total_tokens: int = Field(0, description="Total tokens", examples=[125])


class OpenAIChatResponse(BaseModel):
    """Response schema for /v1/chat/completions endpoint."""

    id: str = Field(default="chatcmpl-rkllama", description="Unique response ID", examples=["chatcmpl-abc123"])
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp", examples=[1704067200])
    model: str = Field(..., description="Model used", examples=["qwen:7b"])
    choices: list[OpenAIChatChoice] = Field(..., description="Completion choices")
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage, description="Token usage")
    system_fingerprint: str | None = Field(None, description="System fingerprint")


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

    model: str = Field(..., description="Model ID", examples=["qwen:7b"])
    prompt: str | list[str] = Field(..., description="Text prompt(s)", examples=["Once upon a time"])
    suffix: str | None = Field(None, description="Text to append after completion")
    max_tokens: int | None = Field(None, alias="max_completion_tokens", description="Max tokens", examples=[256])
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature", examples=[0.7])
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling", examples=[0.9])
    n: int = Field(1, ge=1, description="Number of completions")
    stream: bool = Field(False, description="Stream response")
    logprobs: int | None = Field(None, description="Include log probabilities")
    echo: bool = Field(False, description="Echo prompt in response")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    best_of: int | None = Field(None, description="Best of N completions")
    logit_bias: dict[str, float] | None = Field(None, description="Token logit biases")
    user: str | None = Field(None, description="User identifier")
    seed: int | None = Field(None, description="Random seed")

    model_config = {"populate_by_name": True}


class OpenAICompletionChoice(BaseModel):
    """Choice in completion response."""

    text: str = Field(..., description="Generated text", examples=["there lived a princess..."])
    index: int = Field(0, description="Choice index")
    logprobs: Any | None = Field(None, description="Log probabilities")
    finish_reason: str | None = Field("stop", description="Finish reason", examples=["stop", "length"])


class OpenAICompletionResponse(BaseModel):
    """Response schema for /v1/completions endpoint."""

    id: str = Field(default="cmpl-rkllama", description="Unique response ID", examples=["cmpl-abc123"])
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp", examples=[1704067200])
    model: str = Field(..., description="Model used", examples=["qwen:7b"])
    choices: list[OpenAICompletionChoice] = Field(..., description="Completion choices")
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage, description="Token usage")
    system_fingerprint: str | None = Field(None, description="System fingerprint")


# ============================================================================
# Embeddings Schemas
# ============================================================================


class OpenAIEmbeddingRequest(BaseModel):
    """Request schema for /v1/embeddings endpoint."""

    model: str = Field(..., description="Model ID", examples=["qwen:7b"])
    input: str | list[str] = Field(..., description="Text to embed", examples=["Hello, world!"])
    encoding_format: str = Field("float", description="Encoding format (float or base64)")
    dimensions: int | None = Field(None, description="Output dimensions")
    user: str | None = Field(None, description="User identifier")


class OpenAIEmbeddingData(BaseModel):
    """Embedding data in response."""

    object: str = Field("embedding", description="Object type")
    embedding: list[float] = Field(..., description="Embedding vector")
    index: int = Field(0, description="Index in batch")


class OpenAIEmbeddingResponse(BaseModel):
    """Response schema for /v1/embeddings endpoint."""

    object: str = Field("list", description="Object type")
    data: list[OpenAIEmbeddingData] = Field(..., description="Embedding data")
    model: str = Field(..., description="Model used", examples=["qwen:7b"])
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage, description="Token usage")


# ============================================================================
# Models Schemas
# ============================================================================


class OpenAIModel(BaseModel):
    """Model info in OpenAI format."""

    id: str = Field(..., description="Model ID", examples=["qwen:7b"])
    object: str = Field("model", description="Object type")
    created: int = Field(0, description="Creation timestamp", examples=[1704067200])
    owned_by: str = Field("rkllama", description="Model owner", examples=["rkllama"])
    permission: list[dict[str, Any]] = Field(default_factory=list, description="Permissions")
    root: str | None = Field(None, description="Root model")
    parent: str | None = Field(None, description="Parent model")


class OpenAIModelList(BaseModel):
    """Response schema for /v1/models endpoint."""

    object: str = Field("list", description="Object type")
    data: list[OpenAIModel] = Field(..., description="List of models")


# ============================================================================
# Image Generation Schemas
# ============================================================================


class OpenAIImageRequest(BaseModel):
    """Request schema for /v1/images/generations endpoint."""

    model: str | None = Field(None, description="Model ID", examples=["stable-diffusion"])
    prompt: str = Field(..., description="Image prompt", examples=["A sunset over mountains"])
    n: int = Field(1, ge=1, le=10, description="Number of images")
    size: str = Field("512x512", description="Image size (e.g., 512x512, 1024x1024)", examples=["512x512"])
    quality: str = Field("standard", description="Image quality", examples=["standard", "hd"])
    response_format: str = Field("b64_json", description="Response format (b64_json or url)")
    style: str | None = Field(None, description="Image style")
    user: str | None = Field(None, description="User identifier")
    # RKNN-specific
    seed: int | None = Field(None, description="Random seed for reproducibility")
    num_inference_steps: int | None = Field(None, description="Inference steps", examples=[4])
    guidance_scale: float | None = Field(None, description="Guidance scale", examples=[7.5])
    output_format: str = Field("png", description="Output format (png, jpeg)")


class OpenAIImageData(BaseModel):
    """Image data in response."""

    b64_json: str | None = Field(None, description="Base64 encoded image")
    url: str | None = Field(None, description="Image URL")
    revised_prompt: str | None = Field(None, description="Revised prompt used")


class OpenAIImageResponse(BaseModel):
    """Response schema for /v1/images/generations endpoint."""

    created: int = Field(..., description="Unix timestamp", examples=[1704067200])
    data: list[OpenAIImageData] = Field(..., description="Generated images")


# ============================================================================
# Audio Schemas
# ============================================================================


class OpenAISpeechRequest(BaseModel):
    """Request schema for /v1/audio/speech endpoint."""

    model: str = Field(..., description="Model ID", examples=["piper-voice"])
    input: str = Field(..., description="Text to synthesize", examples=["Hello, how are you today?"])
    voice: str = Field("default", description="Voice to use", examples=["default", "en_US-amy-medium"])
    response_format: str = Field("mp3", description="Audio format", examples=["mp3", "wav", "opus"])
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speech speed", examples=[1.0])
    # Piper-specific
    volume: float | None = Field(None, description="Audio volume")
    length_scale: float | None = Field(None, description="Piper length scale")
    noise_scale: float | None = Field(None, description="Piper noise scale")
    noise_w_scale: float | None = Field(None, description="Piper noise W scale")
    normalize_audio: bool | None = Field(None, description="Normalize audio output")


class OpenAITranscriptionRequest(BaseModel):
    """Request schema for /v1/audio/transcriptions endpoint."""

    model: str = Field(..., description="Model ID", examples=["whisper"])
    # file is handled separately as multipart form data
    language: str | None = Field(None, description="Audio language", examples=["en", "es", "fr"])
    prompt: str | None = Field(None, description="Transcription prompt/context")
    response_format: str = Field("json", description="Response format", examples=["json", "text", "srt", "vtt"])
    temperature: float | None = Field(None, ge=0.0, le=1.0, description="Sampling temperature")
    timestamp_granularities: list[str] | None = Field(None, description="Timestamp granularities")


class OpenAITranscriptionResponse(BaseModel):
    """Response schema for /v1/audio/transcriptions endpoint."""

    text: str = Field(..., description="Transcribed text", examples=["Hello, how are you today?"])
    task: str | None = Field(None, description="Task type", examples=["transcribe"])
    language: str | None = Field(None, description="Detected language", examples=["english"])
    duration: float | None = Field(None, description="Audio duration in seconds", examples=[5.5])
    words: list[dict[str, Any]] | None = Field(None, description="Word-level timestamps")
    segments: list[dict[str, Any]] | None = Field(None, description="Segment-level timestamps")
