"""Pydantic schemas for Modelfile CRUD API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class ModelfileProperty(BaseModel):
    """Single property from a Modelfile."""

    name: str = Field(..., description="Property name (e.g., TEMPERATURE)")
    value: str | int | float | bool = Field(..., description="Property value")
    description: str | None = Field(None, description="Property description")


class ModelfileResponse(BaseModel):
    """Response schema for GET /api/modelfile/{model}."""

    model: str = Field(..., description="Model name")
    path: str = Field(..., description="Path to Modelfile")
    properties: dict[str, Any] = Field(default_factory=dict, description="All properties")

    # Parsed/validated properties (optional, for convenience)
    from_: str | None = Field(None, alias="FROM", description="Model file path")
    huggingface_path: str | None = Field(None, alias="HUGGINGFACE_PATH")
    system: str | None = Field(None, alias="SYSTEM")
    temperature: float | None = Field(None, alias="TEMPERATURE")
    num_ctx: int | None = Field(None, alias="NUM_CTX")
    max_new_tokens: int | None = Field(None, alias="MAX_NEW_TOKENS")
    top_k: int | None = Field(None, alias="TOP_K")
    top_p: float | None = Field(None, alias="TOP_P")
    repeat_penalty: float | None = Field(None, alias="REPEAT_PENALTY")
    frequency_penalty: float | None = Field(None, alias="FREQUENCY_PENALTY")
    presence_penalty: float | None = Field(None, alias="PRESENCE_PENALTY")
    mirostat: int | None = Field(None, alias="MIROSTAT")
    mirostat_tau: float | None = Field(None, alias="MIROSTAT_TAU")
    mirostat_eta: float | None = Field(None, alias="MIROSTAT_ETA")
    enable_thinking: bool | None = Field(None, alias="ENABLE_THINKING")

    model_config = {"populate_by_name": True}


class ModelfilePatchRequest(BaseModel):
    """Request schema for PATCH /api/modelfile/{model}."""

    properties: dict[str, Any] = Field(
        ...,
        description="Properties to update (key-value pairs)",
        examples=[
            {"TEMPERATURE": 0.8, "TOP_K": 40},
            {"SYSTEM": "You are a helpful assistant.", "NUM_CTX": 4096},
        ],
    )


class ModelfilePropertyDeleteRequest(BaseModel):
    """Request schema for DELETE /api/modelfile/{model}/{property}."""

    # No body needed - property name is in URL path
    pass


class ModelfileCreateRequest(BaseModel):
    """Request schema for creating a new Modelfile."""

    model: str = Field(..., description="Model name")
    from_: str = Field(..., alias="FROM", description="Path to .rkllm model file")
    huggingface_path: str = Field(..., alias="HUGGINGFACE_PATH", description="HuggingFace model ID for tokenizer")
    system: str = Field("", alias="SYSTEM", description="Default system prompt")
    temperature: float = Field(0.7, alias="TEMPERATURE", ge=0.0, le=2.0)
    num_ctx: int = Field(4096, alias="NUM_CTX", ge=128)
    max_new_tokens: int = Field(1024, alias="MAX_NEW_TOKENS", ge=1)
    top_k: int = Field(40, alias="TOP_K", ge=1)
    top_p: float = Field(0.9, alias="TOP_P", ge=0.0, le=1.0)
    repeat_penalty: float = Field(1.1, alias="REPEAT_PENALTY", ge=0.0)
    frequency_penalty: float = Field(0.0, alias="FREQUENCY_PENALTY", ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, alias="PRESENCE_PENALTY", ge=-2.0, le=2.0)
    mirostat: int = Field(0, alias="MIROSTAT", ge=0, le=2)
    mirostat_tau: float = Field(3.0, alias="MIROSTAT_TAU")
    mirostat_eta: float = Field(0.1, alias="MIROSTAT_ETA")
    enable_thinking: bool = Field(False, alias="ENABLE_THINKING")

    model_config = {"populate_by_name": True}


class ModelfileListResponse(BaseModel):
    """Response schema for listing all models with Modelfiles."""

    models: list[str] = Field(default_factory=list, description="List of model names")
    count: int = Field(0, description="Total number of models")


# ============================================================================
# Validation Constants
# ============================================================================


MODELFILE_REQUIRED_PROPERTIES = ["FROM", "HUGGINGFACE_PATH"]

MODELFILE_OPTIONAL_PROPERTIES = [
    "SYSTEM",
    "TEMPERATURE",
    "NUM_CTX",
    "MAX_NEW_TOKENS",
    "TOP_K",
    "TOP_P",
    "REPEAT_PENALTY",
    "FREQUENCY_PENALTY",
    "PRESENCE_PENALTY",
    "MIROSTAT",
    "MIROSTAT_TAU",
    "MIROSTAT_ETA",
    "ENABLE_THINKING",
    "VISION_ENCODER",
    "TOKENIZER",
    "TEMPLATE",
    "LICENSE",
]

MODELFILE_ALL_PROPERTIES = MODELFILE_REQUIRED_PROPERTIES + MODELFILE_OPTIONAL_PROPERTIES


def validate_property_name(name: str) -> bool:
    """Validate that a property name is allowed."""
    return name.upper() in MODELFILE_ALL_PROPERTIES


def validate_property_value(name: str, value: Any) -> tuple[bool, str | None]:
    """
    Validate a property value based on its name.

    Returns:
        Tuple of (is_valid, error_message)
    """
    name = name.upper()

    if name == "TEMPERATURE":
        if not isinstance(value, int | float) or not (0.0 <= float(value) <= 2.0):
            return False, "TEMPERATURE must be between 0.0 and 2.0"

    elif name == "NUM_CTX":
        if not isinstance(value, int) or value < 128:
            return False, "NUM_CTX must be an integer >= 128"

    elif name in ("TOP_P",):
        if not isinstance(value, int | float) or not (0.0 <= float(value) <= 1.0):
            return False, f"{name} must be between 0.0 and 1.0"

    elif name == "MIROSTAT":
        if not isinstance(value, int) or value not in (0, 1, 2):
            return False, "MIROSTAT must be 0, 1, or 2"

    elif name == "ENABLE_THINKING" and not isinstance(value, bool) and value not in ("True", "False", "true", "false"):
        return False, "ENABLE_THINKING must be a boolean"

    return True, None
