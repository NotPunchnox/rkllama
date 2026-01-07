"""
RKLLM format converter module.
This module wraps the official rkllm-toolkit for model conversion.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RKLLMConfig:
    """Configuration for RKLLM conversion."""

    model_path: str
    output_path: str
    target_platform: str = "rk3588"
    quantized_dtype: str = "w8a8"
    num_npu_core: int = 3
    max_context: int = 4096
    optimization_level: int = 1
    device: str = "cuda"
    dtype: str = "float16"
    dataset: str | None = None
    hybrid_rate: float = 0.0

    # Quantization type mapping from user-friendly names
    QUANT_MAPPING = {
        "Q4_0": "w4a16",
        "Q4_K_M": "w4a16_g128",
        "Q8_0": "w8a8",
        "Q8_K_M": "w8a8_g512",
    }

    @classmethod
    def from_quantization(cls, quant: str) -> str:
        """Convert user-friendly quantization name to RKLLM format."""
        return cls.QUANT_MAPPING.get(quant, quant)


class RKLLMConverter:
    """Converts HuggingFace models to RKLLM format using official toolkit."""

    def __init__(self, config: RKLLMConfig):
        self.config = config
        self._rkllm = None

    def convert(self) -> None:
        """Run the full conversion pipeline."""
        logger.info("Starting RKLLM conversion with official toolkit...")

        # Import here to avoid import errors when toolkit not installed
        from rkllm.api import RKLLM

        self._rkllm = RKLLM()

        # Step 1: Load model
        self._load_model()

        # Step 2: Build/quantize
        self._build_model()

        # Step 3: Export
        self._export_model()

        logger.info("RKLLM conversion completed successfully")

    def _load_model(self) -> None:
        """Load the HuggingFace model."""
        logger.info(f"Loading model from {self.config.model_path}...")

        ret = self._rkllm.load_huggingface(
            model=self.config.model_path,
            model_lora=None,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        if ret != 0:
            raise RuntimeError(f"Failed to load model: return code {ret}")

        logger.info("Model loaded successfully")

    def _build_model(self) -> None:
        """Build and quantize the model."""
        logger.info(f"Building model with {self.config.quantized_dtype} quantization...")

        build_kwargs: dict[str, Any] = {
            "do_quantization": True,
            "optimization_level": self.config.optimization_level,
            "quantized_dtype": self.config.quantized_dtype,
            "quantized_algorithm": "normal",
            "num_npu_core": self.config.num_npu_core,
            "target_platform": self.config.target_platform,
            "max_context": self.config.max_context,
            "hybrid_rate": self.config.hybrid_rate,
        }

        # Add dataset if provided for calibration
        if self.config.dataset:
            build_kwargs["dataset"] = self.config.dataset

        ret = self._rkllm.build(**build_kwargs)

        if ret != 0:
            raise RuntimeError(f"Failed to build model: return code {ret}")

        logger.info("Model built successfully")

    def _export_model(self) -> None:
        """Export the model to RKLLM format."""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to {output_path}...")

        ret = self._rkllm.export_rkllm(export_path=str(output_path))

        if ret != 0:
            raise RuntimeError(f"Failed to export model: return code {ret}")

        logger.info(f"Model exported to {output_path}")


def convert_huggingface_to_rkllm(
    model_path: str,
    output_path: str,
    quantization: str = "Q8_0",
    target_platform: str = "rk3588",
    max_context: int = 4096,
    num_npu_core: int = 3,
    device: str = "cuda",
    dtype: str = "float16",
    dataset: str | None = None,
) -> None:
    """
    Convenience function to convert a HuggingFace model to RKLLM format.

    Args:
        model_path: Path to HuggingFace model directory or model ID
        output_path: Path for output .rkllm file
        quantization: Quantization type (Q4_0, Q4_K_M, Q8_0, Q8_K_M)
        target_platform: Target platform (rk3588, rk3576, rk3562, rv1126b)
        max_context: Maximum context length (up to 16384, must align to 32)
        num_npu_core: Number of NPU cores (RK3588: 1-3, RK3576: 1-2)
        device: Device for conversion (cuda, cpu)
        dtype: Weight dtype (float16, float32, bfloat16)
        dataset: Optional JSON dataset for quantization calibration
    """
    config = RKLLMConfig(
        model_path=model_path,
        output_path=output_path,
        quantized_dtype=RKLLMConfig.from_quantization(quantization),
        target_platform=target_platform.lower(),
        max_context=max_context,
        num_npu_core=num_npu_core,
        device=device,
        dtype=dtype,
        dataset=dataset,
    )

    converter = RKLLMConverter(config)
    converter.convert()
