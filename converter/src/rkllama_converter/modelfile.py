"""
Modelfile generation for converted RKLLM models.

The Modelfile is a configuration file that tells RKLlama how to load and
configure a model. It contains the model path, HuggingFace reference for
tokenizer loading, and generation parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelfileConfig:
    """
    Complete configuration for Modelfile generation.

    This maps to the Modelfile schema used by RKLlama server.
    All fields have sensible defaults that work well for most models.

    Attributes:
        model_file: Filename of the .rkllm model file (required)
        huggingface_path: HuggingFace model ID for tokenizer (required)
        system: System prompt for the model
        temperature: Sampling temperature (0.0-2.0)
        num_ctx: Context window size in tokens
        max_new_tokens: Maximum tokens to generate
        top_k: Top-K sampling parameter
        top_p: Nucleus sampling parameter
        repeat_penalty: Repetition penalty
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        mirostat: Mirostat sampling mode (0=disabled, 1=Mirostat, 2=Mirostat 2.0)
        mirostat_tau: Mirostat target entropy
        mirostat_eta: Mirostat learning rate
        enable_thinking: Enable reasoning/thinking mode
        template: Custom chat template (optional)
        tokenizer: Custom tokenizer path (optional)
        license: Model license (optional)

    Example:
        >>> config = ModelfileConfig(
        ...     model_file="qwen2.5-7b.rkllm",
        ...     huggingface_path="Qwen/Qwen2.5-7B-Instruct",
        ...     temperature=0.8,
        ...     enable_thinking=True,
        ... )
        >>> content = generate_modelfile(config)
    """

    # Required fields
    model_file: str
    huggingface_path: str

    # Generation parameters
    system: str = "You are a helpful AI assistant."
    temperature: float = 0.7
    num_ctx: int = 4096
    max_new_tokens: int = 1024
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Mirostat sampling
    mirostat: int = 0
    mirostat_tau: float = 3.0
    mirostat_eta: float = 0.1

    # Advanced options
    enable_thinking: bool = False
    template: str | None = None
    tokenizer: str | None = None
    license: str | None = None

    # Vision model options (for multimodal models)
    vision_encoder: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    n_image_tokens: int | None = None
    img_start: str | None = None
    img_end: str | None = None
    img_content: str | None = None


def generate_modelfile(config: ModelfileConfig) -> str:
    """
    Generate Modelfile content from configuration.

    Args:
        config: ModelfileConfig with all settings.

    Returns:
        String content for the Modelfile.

    Example:
        >>> config = ModelfileConfig(
        ...     model_file="model.rkllm",
        ...     huggingface_path="Qwen/Qwen2.5-7B",
        ... )
        >>> print(generate_modelfile(config))
        FROM="model.rkllm"
        HUGGINGFACE_PATH="Qwen/Qwen2.5-7B"
        ...
    """
    lines = [
        f'FROM="{config.model_file}"',
        f'HUGGINGFACE_PATH="{config.huggingface_path}"',
        "",
        "# System prompt",
        f'SYSTEM="{_escape_string(config.system)}"',
        "",
        "# Generation parameters",
        f"TEMPERATURE={config.temperature}",
        f"NUM_CTX={config.num_ctx}",
        f"MAX_NEW_TOKENS={config.max_new_tokens}",
        f"TOP_K={config.top_k}",
        f"TOP_P={config.top_p}",
        f"REPEAT_PENALTY={config.repeat_penalty}",
    ]

    # Add frequency/presence penalties if non-zero
    if config.frequency_penalty != 0.0:
        lines.append(f"FREQUENCY_PENALTY={config.frequency_penalty}")
    if config.presence_penalty != 0.0:
        lines.append(f"PRESENCE_PENALTY={config.presence_penalty}")

    # Add mirostat settings if enabled
    if config.mirostat > 0:
        lines.extend([
            "",
            "# Mirostat sampling",
            f"MIROSTAT={config.mirostat}",
            f"MIROSTAT_TAU={config.mirostat_tau}",
            f"MIROSTAT_ETA={config.mirostat_eta}",
        ])

    # Add thinking mode if enabled
    if config.enable_thinking:
        lines.extend([
            "",
            "# Reasoning mode",
            "ENABLE_THINKING=true",
        ])

    # Add optional custom template
    if config.template:
        lines.extend([
            "",
            "# Custom chat template",
            f'TEMPLATE="{_escape_string(config.template)}"',
        ])

    # Add optional custom tokenizer
    if config.tokenizer:
        lines.extend([
            "",
            f'TOKENIZER="{config.tokenizer}"',
        ])

    # Add license if specified
    if config.license:
        lines.extend([
            "",
            f'LICENSE="{config.license}"',
        ])

    # Add vision encoder settings for multimodal models
    if config.vision_encoder:
        lines.extend([
            "",
            "# Vision encoder settings",
            f'VISION_ENCODER="{config.vision_encoder}"',
        ])
        if config.image_width:
            lines.append(f"IMAGE_WIDTH={config.image_width}")
        if config.image_height:
            lines.append(f"IMAGE_HEIGHT={config.image_height}")
        if config.n_image_tokens:
            lines.append(f"N_IMAGE_TOKENS={config.n_image_tokens}")
        if config.img_start:
            lines.append(f"IMG_START={config.img_start}")
        if config.img_end:
            lines.append(f"IMG_END={config.img_end}")
        if config.img_content:
            lines.append(f"IMG_CONTENT={config.img_content}")

    return "\n".join(lines) + "\n"


def _escape_string(s: str) -> str:
    """Escape special characters in string values."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def parse_modelfile(content: str) -> dict[str, Any]:
    """
    Parse a Modelfile into a dictionary.

    Args:
        content: Raw Modelfile content.

    Returns:
        Dictionary of parsed key-value pairs.

    Example:
        >>> content = 'FROM="model.rkllm"\\nTEMPERATURE=0.7'
        >>> parse_modelfile(content)
        {'FROM': 'model.rkllm', 'TEMPERATURE': 0.7}
    """
    result: dict[str, Any] = {}

    for line in content.split("\n"):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Parse key=value pairs
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Remove quotes from string values
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
            # Unescape special characters
            value = value.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
            result[key] = value
        else:
            # Try to parse as number or boolean
            if value.lower() == "true":
                result[key] = True
            elif value.lower() == "false":
                result[key] = False
            else:
                try:
                    if "." in value:
                        result[key] = float(value)
                    else:
                        result[key] = int(value)
                except ValueError:
                    result[key] = value

    return result


def save_modelfile(config: ModelfileConfig, output_dir: str | Path) -> Path:
    """
    Generate and save a Modelfile to disk.

    Args:
        config: ModelfileConfig with all settings.
        output_dir: Directory to save the Modelfile in.

    Returns:
        Path to the saved Modelfile.

    Example:
        >>> config = ModelfileConfig(model_file="m.rkllm", huggingface_path="org/model")
        >>> path = save_modelfile(config, "/models/mymodel")
        >>> print(path)
        /models/mymodel/Modelfile
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    modelfile_path = output_path / "Modelfile"
    content = generate_modelfile(config)

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)

    return modelfile_path
