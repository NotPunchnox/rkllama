"""Common Pydantic schemas shared across API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class Options(BaseModel):
    """Model inference options."""

    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int | None = Field(None, ge=1, description="Top-K sampling")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-P (nucleus) sampling")
    num_predict: int | None = Field(None, alias="max_tokens", description="Max tokens to generate")
    num_ctx: int | None = Field(None, description="Context window size")
    repeat_penalty: float | None = Field(None, ge=0.0, description="Repetition penalty")
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    mirostat: int | None = Field(None, ge=0, le=2, description="Mirostat mode (0=disabled)")
    mirostat_tau: float | None = Field(None, description="Mirostat tau")
    mirostat_eta: float | None = Field(None, description="Mirostat eta")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    stop: list[str] | None = Field(None, description="Stop sequences")

    model_config = {"populate_by_name": True, "extra": "allow"}


class ModelDetails(BaseModel):
    """Model detail information."""

    parent_model: str = ""
    format: str = "rkllm"
    family: str = ""
    families: list[str] = Field(default_factory=list)
    parameter_size: str = ""
    quantization_level: str = ""


class ModelInfo(BaseModel):
    """Full model metadata."""

    name: str
    model: str
    size: int = 0
    digest: str = ""
    details: ModelDetails = Field(default_factory=ModelDetails)
    modified_at: str | None = None

    # Extended info
    license: str | None = None
    modelfile: str | None = None
    parameters: str | None = None
    template: str | None = None
    system: str | None = None


class UsageStats(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str
    code: str | None = None
    details: dict[str, Any] | None = None


class DurationMetrics(BaseModel):
    """Duration metrics for inference."""

    total_duration: int = Field(description="Total duration in nanoseconds")
    load_duration: int = Field(default=100000000, description="Model load duration in nanoseconds")
    prompt_eval_count: int = Field(default=0, description="Number of prompt tokens evaluated")
    prompt_eval_duration: int = Field(default=0, description="Prompt evaluation duration in nanoseconds")
    eval_count: int = Field(default=0, description="Number of tokens generated")
    eval_duration: int = Field(default=0, description="Generation duration in nanoseconds")
