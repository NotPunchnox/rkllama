"""Pydantic-based configuration schema for RKLlama."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Server configuration."""

    model_config = SettingsConfigDict(env_prefix="RKLLAMA_SERVER_")

    port: int = Field(default=8080, ge=1, le=65535, description="Server port number")
    host: str = Field(default="0.0.0.0", description="Server host address")
    debug: bool = Field(default=False, description="Enable debug mode")


class PathSettings(BaseSettings):
    """Path configuration."""

    model_config = SettingsConfigDict(env_prefix="RKLLAMA_PATHS_")

    models: str = Field(default="models", description="Path to model files")
    logs: str = Field(default="logs", description="Path to log files")
    data: str = Field(default="data", description="Path to data files")
    src: str = Field(default="src", description="Path to source files")
    lib: str = Field(default="lib", description="Path to library files")
    temp: str = Field(default="temp", description="Path to temporary files")


class ModelSettings(BaseSettings):
    """Model default parameters."""

    model_config = SettingsConfigDict(env_prefix="RKLLAMA_MODEL_")

    default: str = Field(default="", description="Default model to use")
    default_temperature: float = Field(
        default=0.5, ge=0.0, le=2.0, description="Default temperature for inference"
    )
    default_enable_thinking: bool = Field(
        default=False, description="Enable thinking/reasoning mode by default"
    )
    default_num_ctx: int = Field(
        default=4096, ge=512, le=131072, description="Default context window size in tokens"
    )
    default_max_new_tokens: int = Field(
        default=1024, ge=1, le=32768, description="Maximum tokens to generate per response"
    )
    default_top_k: int = Field(default=7, ge=1, le=100, description="Top-K sampling parameter")
    default_top_p: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Top-P (nucleus) sampling parameter"
    )
    default_repeat_penalty: float = Field(
        default=1.1, ge=0.0, le=2.0, description="Penalty for token repetition"
    )
    default_frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty for sampling"
    )
    default_presence_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Presence penalty for sampling"
    )
    default_mirostat: int = Field(
        default=0, ge=0, le=2, description="Mirostat sampling mode (0=disabled, 1=v1, 2=v2)"
    )
    default_mirostat_tau: float = Field(
        default=3.0, ge=0.0, le=10.0, description="Mirostat target entropy"
    )
    default_mirostat_eta: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Mirostat learning rate"
    )
    max_minutes_loaded_in_memory: int = Field(
        default=30, ge=1, le=1440, description="Max idle minutes before unloading model"
    )
    max_number_models_loaded_in_memory: int = Field(
        default=10, ge=1, le=100, description="Max concurrent loaded models"
    )


class PlatformSettings(BaseSettings):
    """Platform configuration."""

    model_config = SettingsConfigDict(env_prefix="RKLLAMA_PLATFORM_")

    processor: Literal["rk3588", "rk3576"] = Field(
        default="rk3588", description="Target processor"
    )


class RKLlamaSettings(BaseSettings):
    """Main RKLlama configuration combining all sections."""

    model_config = SettingsConfigDict(
        env_prefix="RKLLAMA_",
        env_nested_delimiter="_",
        extra="allow",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    platform: PlatformSettings = Field(default_factory=PlatformSettings)

    def get_section(self, section: str) -> BaseSettings | None:
        """Get a settings section by name."""
        return getattr(self, section, None)

    def resolve_path(self, path: str, app_root: Path) -> str | None:
        """Resolve a path relative to the application root."""
        import os

        if not path:
            return None

        path_obj = Path(path)

        if path_obj.is_absolute():
            return str(path_obj)
        elif "$" in path or "~" in path:
            expanded_path = os.path.expanduser(os.path.expandvars(path))
            if os.path.isabs(expanded_path):
                return expanded_path
            return str(app_root / expanded_path)
        else:
            return str(app_root / path)
