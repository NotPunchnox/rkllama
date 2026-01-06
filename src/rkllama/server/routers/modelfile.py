"""Modelfile CRUD API routes."""

import os
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from rkllama.api.schemas.modelfile import (
    ModelfilePatchRequest,
    ModelfilePropertyName,
    ModelfilePropertyResponse,
    ModelfileResponse,
    validate_property_name,
    validate_property_value,
)
from rkllama.logging import get_logger
from rkllama.server.dependencies import get_debug_mode, get_models_path

logger = get_logger("rkllama.server.modelfile")

router = APIRouter()


def parse_modelfile(modelfile_path: str) -> dict[str, Any]:
    """Parse a Modelfile into a dictionary of properties."""
    properties = {}

    if not os.path.exists(modelfile_path):
        return properties

    with open(modelfile_path) as f:
        content = f.read()

    # Match KEY="value" or KEY=value patterns
    pattern = r'^([A-Z_]+)=(?:"([^"]*)"|(.+))$'

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = re.match(pattern, line)
        if match:
            key = match.group(1)
            # Value is either in group 2 (quoted) or group 3 (unquoted)
            value = match.group(2) if match.group(2) is not None else match.group(3)
            properties[key] = value

    return properties


def write_modelfile(modelfile_path: str, properties: dict[str, Any]) -> None:
    """Write properties to a Modelfile."""
    lines = []

    # Define the order of properties
    order = [
        "FROM",
        "HUGGINGFACE_PATH",
        "VISION_ENCODER",
        "SYSTEM",
        "TEMPLATE",
        "LICENSE",
        "TEMPERATURE",
        "ENABLE_THINKING",
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
    ]

    # Write properties in order
    for key in order:
        if key in properties:
            value = properties[key]
            # Quote string values that contain spaces or special characters
            if isinstance(value, str) and (
                " " in value
                or '"' in value
                or "\n" in value
                or key in ["FROM", "HUGGINGFACE_PATH", "SYSTEM", "TEMPLATE", "LICENSE", "VISION_ENCODER"]
            ):
                lines.append(f'{key}="{value}"')
            else:
                lines.append(f"{key}={value}")
            lines.append("")  # Empty line after each property

    # Write any remaining properties not in the order list
    for key, value in properties.items():
        if key not in order:
            if isinstance(value, str) and " " in value:
                lines.append(f'{key}="{value}"')
            else:
                lines.append(f"{key}={value}")
            lines.append("")

    with open(modelfile_path, "w") as f:
        f.write("\n".join(lines))


@router.get("/{model}")
async def get_modelfile(
    model: str,
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> ModelfileResponse:
    """Get all properties from a model's Modelfile."""
    model_dir = os.path.join(models_path, model)
    modelfile_path = os.path.join(model_dir, "Modelfile")

    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    if not os.path.exists(modelfile_path):
        raise HTTPException(status_code=404, detail=f"Modelfile not found for model '{model}'")

    if debug:
        logger.debug("Reading Modelfile", model=model)

    properties = parse_modelfile(modelfile_path)

    return ModelfileResponse(
        model=model,
        path=modelfile_path,
        properties=properties,
    )


@router.get("/{model}/{property_name}", response_model=ModelfilePropertyResponse)
async def get_modelfile_property(
    model: str,
    property_name: ModelfilePropertyName,
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> ModelfilePropertyResponse:
    """Get a specific property from a model's Modelfile."""
    model_dir = os.path.join(models_path, model)
    modelfile_path = os.path.join(model_dir, "Modelfile")

    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    if not os.path.exists(modelfile_path):
        raise HTTPException(status_code=404, detail=f"Modelfile not found for model '{model}'")

    if debug:
        logger.debug("Reading property", model=model, property=property_name.value)

    properties = parse_modelfile(modelfile_path)

    if property_name.value not in properties:
        raise HTTPException(
            status_code=404,
            detail=f"Property '{property_name.value}' not found in Modelfile for model '{model}'",
        )

    return ModelfilePropertyResponse(
        model=model,
        property=property_name,
        value=properties[property_name.value],
    )


@router.patch("/{model}")
async def update_modelfile(
    model: str,
    request: ModelfilePatchRequest,
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> ModelfileResponse:
    """Update properties in a model's Modelfile."""
    model_dir = os.path.join(models_path, model)
    modelfile_path = os.path.join(model_dir, "Modelfile")

    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    if not os.path.exists(modelfile_path):
        raise HTTPException(status_code=404, detail=f"Modelfile not found for model '{model}'")

    if debug:
        logger.debug("Updating Modelfile", model=model, properties=request.properties)

    # Validate all properties before applying any changes
    errors = []
    normalized_properties = {}

    for key, value in request.properties.items():
        key_upper = key.upper()

        # Validate property name
        if not validate_property_name(key_upper):
            errors.append(f"Unknown property: {key_upper}")
            continue

        # Validate property value
        is_valid, error_msg = validate_property_value(key_upper, value)
        if not is_valid:
            errors.append(f"{key_upper}: {error_msg}")
            continue

        normalized_properties[key_upper] = value

    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Validation errors", "errors": errors},
        )

    # Read current properties
    current_properties = parse_modelfile(modelfile_path)

    # Merge with new properties
    current_properties.update(normalized_properties)

    # Write back
    write_modelfile(modelfile_path, current_properties)

    return ModelfileResponse(
        model=model,
        path=modelfile_path,
        properties=current_properties,
    )


class ModelfilePropertyDeleteResponse(BaseModel):
    """Response for deleting a property."""

    model: str
    property: ModelfilePropertyName
    deleted_value: str | int | float | bool
    message: str


@router.delete("/{model}/{property_name}", response_model=ModelfilePropertyDeleteResponse)
async def delete_modelfile_property(
    model: str,
    property_name: ModelfilePropertyName,
    models_path: str = Depends(get_models_path),
    debug: bool = Depends(get_debug_mode),
) -> ModelfilePropertyDeleteResponse:
    """Delete a property from a model's Modelfile."""
    model_dir = os.path.join(models_path, model)
    modelfile_path = os.path.join(model_dir, "Modelfile")

    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    if not os.path.exists(modelfile_path):
        raise HTTPException(status_code=404, detail=f"Modelfile not found for model '{model}'")

    # Prevent deletion of required properties
    required_properties = [ModelfilePropertyName.FROM, ModelfilePropertyName.HUGGINGFACE_PATH]
    if property_name in required_properties:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete required property '{property_name.value}'",
        )

    if debug:
        logger.debug("Deleting property", model=model, property=property_name.value)

    properties = parse_modelfile(modelfile_path)

    if property_name.value not in properties:
        raise HTTPException(
            status_code=404,
            detail=f"Property '{property_name.value}' not found in Modelfile for model '{model}'",
        )

    deleted_value = properties.pop(property_name.value)
    write_modelfile(modelfile_path, properties)

    return ModelfilePropertyDeleteResponse(
        model=model,
        property=property_name,
        deleted_value=deleted_value,
        message=f"Property '{property_name.value}' deleted successfully",
    )
