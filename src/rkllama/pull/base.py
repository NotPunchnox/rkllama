"""Base class for model pull handlers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum


class PullSource(str, Enum):
    """Supported pull sources."""

    HUGGINGFACE = "huggingface"
    URL = "url"
    S3 = "s3"


@dataclass
class PullProgress:
    """Progress update during model pull."""

    status: str
    completed: int = 0
    total: int = 0
    digest: str = ""
    error: str | None = None

    @property
    def percentage(self) -> int:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0
        return int((self.completed / self.total) * 100)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        result = {"status": self.status}
        if self.total > 0:
            result["completed"] = self.completed
            result["total"] = self.total
        if self.digest:
            result["digest"] = self.digest
        if self.error:
            result["error"] = self.error
        return result


class PullHandler(ABC):
    """Abstract base class for model pull handlers."""

    @abstractmethod
    async def pull(
        self,
        source: str,
        model_name: str,
        models_path: str,
    ) -> AsyncGenerator[PullProgress, None]:
        """
        Pull a model from the source.

        Args:
            source: Source identifier (HF repo, URL, S3 URI)
            model_name: Name to give the downloaded model
            models_path: Directory to store models

        Yields:
            PullProgress updates during download
        """
        pass

    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """
        Validate that the source is valid for this handler.

        Args:
            source: Source identifier to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    def create_modelfile(
        self,
        models_path: str,
        model_name: str,
        from_value: str,
        huggingface_path: str = "",
    ) -> None:
        """Create a Modelfile for the downloaded model."""
        import os

        import rkllama.config

        struct_modelfile = f"""FROM="{from_value}"

HUGGINGFACE_PATH="{huggingface_path}"

SYSTEM=""

TEMPERATURE={rkllama.config.get("model", "default_temperature")}

ENABLE_THINKING={rkllama.config.get("model", "default_enable_thinking")}

NUM_CTX={rkllama.config.get("model", "default_num_ctx")}

MAX_NEW_TOKENS={rkllama.config.get("model", "default_max_new_tokens")}

TOP_K={rkllama.config.get("model", "default_top_k")}

TOP_P={rkllama.config.get("model", "default_top_p")}

REPEAT_PENALTY={rkllama.config.get("model", "default_repeat_penalty")}

FREQUENCY_PENALTY={rkllama.config.get("model", "default_frequency_penalty")}

PRESENCE_PENALTY={rkllama.config.get("model", "default_presence_penalty")}

MIROSTAT={rkllama.config.get("model", "default_mirostat")}

MIROSTAT_TAU={rkllama.config.get("model", "default_mirostat_tau")}

MIROSTAT_ETA={rkllama.config.get("model", "default_mirostat_eta")}

"""
        model_dir = os.path.join(models_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, "Modelfile"), "w") as f:
            f.write(struct_modelfile)


def get_handler(source: PullSource) -> PullHandler:
    """Get the appropriate pull handler for a source type."""
    from rkllama.pull.huggingface import HuggingFacePullHandler
    from rkllama.pull.s3 import S3PullHandler
    from rkllama.pull.url import URLPullHandler

    handlers = {
        PullSource.HUGGINGFACE: HuggingFacePullHandler,
        PullSource.URL: URLPullHandler,
        PullSource.S3: S3PullHandler,
    }

    handler_class = handlers.get(source)
    if handler_class is None:
        raise ValueError(f"Unknown pull source: {source}")

    return handler_class()
