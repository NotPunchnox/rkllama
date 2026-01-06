"""Pydantic schemas for native RKLlama API endpoints."""

from pydantic import BaseModel, Field

# ============================================================================
# Models Endpoint Schemas
# ============================================================================


class NativeModelsResponse(BaseModel):
    """Response for GET /models endpoint."""

    models: list[str] = Field(
        default_factory=list,
        description="List of available model names",
        examples=[["qwen:7b", "llama3:8b", "tinyllama:1.1b"]],
    )


# ============================================================================
# Delete Endpoint Schemas
# ============================================================================


class NativeDeleteRequest(BaseModel):
    """Request for DELETE /rm endpoint."""

    model: str = Field(
        ...,
        description="Name of the model to delete",
        examples=["qwen:7b"],
    )


class NativeDeleteResponse(BaseModel):
    """Response for DELETE /rm endpoint."""

    message: str = Field(
        ...,
        description="Success message",
        examples=["The model has been successfully deleted!"],
    )


# ============================================================================
# Load Endpoint Schemas
# ============================================================================


class NativeLoadRequest(BaseModel):
    """Request for POST /load_model endpoint."""

    model_name: str = Field(
        ...,
        description="Name of the model to load into the NPU",
        examples=["qwen:7b"],
    )
    huggingface_path: str | None = Field(
        None,
        description="HuggingFace path for model (only for new models)",
        examples=["Qwen/Qwen2.5-7B-Instruct-GGUF"],
    )
    from_value: str | None = Field(
        None,
        alias="from",
        description="Path to the .rkllm file relative to model directory",
        examples=["model.rkllm"],
    )

    model_config = {"populate_by_name": True}


class NativeLoadResponse(BaseModel):
    """Response for POST /load_model endpoint."""

    message: str | None = Field(
        None,
        description="Success message when model loads",
        examples=["Model qwen:7b loaded successfully."],
    )
    error: str | None = Field(
        None,
        description="Error message if already loaded",
        examples=["A model is already loaded. Nothing to do."],
    )


# ============================================================================
# Unload Endpoint Schemas
# ============================================================================


class NativeUnloadRequest(BaseModel):
    """Request for POST /unload_model endpoint."""

    model_name: str = Field(
        ...,
        description="Name of the model to unload from the NPU",
        examples=["qwen:7b"],
    )


class NativeUnloadResponse(BaseModel):
    """Response for POST /unload_model endpoint."""

    message: str = Field(
        ...,
        description="Success message",
        examples=["Model qwen:7b successfully unloaded!"],
    )


class NativeUnloadAllResponse(BaseModel):
    """Response for POST /unload_models endpoint."""

    message: str = Field(
        ...,
        description="Success message",
        examples=["Models successfully unloaded!"],
    )
