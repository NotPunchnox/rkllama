"""Common Pydantic schemas shared across API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class Options(BaseModel):
    """Model inference options."""

    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature", examples=[0.7])
    top_k: int | None = Field(None, ge=1, description="Top-K sampling", examples=[40])
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-P (nucleus) sampling", examples=[0.9])
    num_predict: int | None = Field(None, alias="max_tokens", description="Max tokens to generate", examples=[256])
    num_ctx: int | None = Field(None, description="Context window size", examples=[4096])
    repeat_penalty: float | None = Field(None, ge=0.0, description="Repetition penalty", examples=[1.1])
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Frequency penalty", examples=[0.0])
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Presence penalty", examples=[0.0])
    mirostat: int | None = Field(None, ge=0, le=2, description="Mirostat mode (0=disabled)", examples=[0])
    mirostat_tau: float | None = Field(None, description="Mirostat tau", examples=[5.0])
    mirostat_eta: float | None = Field(None, description="Mirostat eta", examples=[0.1])
    seed: int | None = Field(None, description="Random seed for reproducibility", examples=[42])
    stop: list[str] | None = Field(None, description="Stop sequences", examples=[["\\n", "END"]])

    model_config = {"populate_by_name": True, "extra": "allow"}


class ModelDetails(BaseModel):
    """Model detail information."""

    parent_model: str = Field("", description="Parent model name")
    format: str = Field("rkllm", description="Model format", examples=["rkllm", "gguf"])
    family: str = Field("", description="Model family", examples=["qwen", "llama"])
    families: list[str] = Field(default_factory=list, description="Model families")
    parameter_size: str = Field("", description="Parameter size", examples=["7B", "13B"])
    quantization_level: str = Field("", description="Quantization level", examples=["w8a8", "w4a16"])


class ModelInfo(BaseModel):
    """Full model metadata."""

    name: str = Field(..., description="Model name", examples=["qwen:7b"])
    model: str = Field(..., description="Model identifier", examples=["qwen:7b"])
    size: int = Field(0, description="Model size in bytes")
    digest: str = Field("", description="Model digest/hash")
    details: ModelDetails = Field(default_factory=ModelDetails, description="Model details")
    modified_at: str | None = Field(
        None, description="Last modification timestamp", examples=["2024-01-15T10:30:00.000Z"]
    )

    # Extended info
    license: str | None = Field(None, description="Model license")
    modelfile: str | None = Field(None, description="Modelfile content")
    parameters: str | None = Field(None, description="Model parameters")
    template: str | None = Field(None, description="Chat template")
    system: str | None = Field(None, description="Default system prompt")


class UsageStats(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(0, description="Tokens in prompt", examples=[25])
    completion_tokens: int = Field(0, description="Tokens in completion", examples=[100])
    total_tokens: int = Field(0, description="Total tokens", examples=[125])


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(..., description="Error message", examples=["Model not found"])
    code: str | None = Field(None, description="Error code", examples=["MODEL_NOT_FOUND"])
    details: dict[str, Any] | None = Field(None, description="Additional error details")


class DurationMetrics(BaseModel):
    """Duration metrics for inference."""

    total_duration: int = Field(description="Total duration in nanoseconds")
    load_duration: int = Field(default=100000000, description="Model load duration in nanoseconds")
    prompt_eval_count: int = Field(default=0, description="Number of prompt tokens evaluated")
    prompt_eval_duration: int = Field(default=0, description="Prompt evaluation duration in nanoseconds")
    eval_count: int = Field(default=0, description="Number of tokens generated")
    eval_duration: int = Field(default=0, description="Generation duration in nanoseconds")
