"""
RKLLM format converter module.
This module wraps the official rkllm-toolkit for model conversion.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)


class ConversionProgress:
    """Progress tracking for RKLLM conversion with Rich integration."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._progress: Progress | None = None
        self._task_id: int | None = None
        self._start_time: float = 0
        self._stage_times: dict[str, float] = {}

    def __enter__(self) -> "ConversionProgress":
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self._progress.__enter__()
        self._start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._progress:
            self._progress.__exit__(*args)
        total_time = time.time() - self._start_time
        self.console.print(f"\n[dim]Total conversion time: {total_time:.1f}s[/]")

    def start_stage(self, description: str, total: int = 100) -> None:
        """Start a new progress stage."""
        if self._progress:
            self._task_id = self._progress.add_task(description, total=total)
            self._stage_times[description] = time.time()

    def update(self, advance: int = 1, description: str | None = None) -> None:
        """Update current stage progress."""
        if self._progress and self._task_id is not None:
            if description:
                self._progress.update(self._task_id, description=description)
            self._progress.advance(self._task_id, advance)

    def complete_stage(self, message: str | None = None) -> None:
        """Mark current stage as complete."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, completed=100)
            if message:
                self.console.print(f"  [green]✓[/] {message}")


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

    def __init__(self, config: RKLLMConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self._rkllm = None
        self._stage_callback: Callable[[str, int, int], None] | None = None

    def set_stage_callback(self, callback: Callable[[str, int, int], None]) -> None:
        """Set callback for stage progress updates: callback(stage_name, current, total)."""
        self._stage_callback = callback

    def _log_stage(self, stage: str, current: int = 0, total: int = 0) -> None:
        """Log stage progress and call callback if set."""
        if self._stage_callback:
            self._stage_callback(stage, current, total)

    def convert(self) -> None:
        """Run the full conversion pipeline."""
        logger.info("Starting RKLLM conversion with official toolkit...")
        self.console.print("\n[bold cyan]Starting RKLLM Conversion Pipeline[/]\n")

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
        self.console.print("\n[bold green]RKLLM conversion completed successfully![/]")

    def _load_model(self) -> None:
        """Load the HuggingFace model."""
        logger.info(f"Loading model from {self.config.model_path}...")
        self._log_stage("load", 0, 100)

        start_time = time.time()
        self.console.print(f"[cyan]Step 1/3:[/] Loading model from [dim]{self.config.model_path}[/]")
        self.console.print(f"         Device: [yellow]{self.config.device}[/], Dtype: [yellow]{self.config.dtype}[/]")

        ret = self._rkllm.load_huggingface(
            model=self.config.model_path,
            model_lora=None,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        if ret != 0:
            self.console.print(f"[red]✗ Failed to load model (return code: {ret})[/]")
            raise RuntimeError(f"Failed to load model: return code {ret}")

        elapsed = time.time() - start_time
        logger.info("Model loaded successfully")
        self.console.print(f"         [green]✓ Model loaded[/] [dim]({elapsed:.1f}s)[/]\n")
        self._log_stage("load", 100, 100)

    def _build_model(self) -> None:
        """Build and quantize the model."""
        logger.info(f"Building model with {self.config.quantized_dtype} quantization...")
        self._log_stage("build", 0, 100)

        start_time = time.time()
        self.console.print("[cyan]Step 2/3:[/] Building and quantizing model")
        self.console.print(f"         Quantization: [yellow]{self.config.quantized_dtype}[/]")
        self.console.print(f"         Target: [yellow]{self.config.target_platform}[/] ({self.config.num_npu_core} NPU cores)")
        self.console.print(f"         Context: [yellow]{self.config.max_context:,}[/] tokens")
        self.console.print()

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
            self.console.print(f"         [dim]Using calibration dataset: {self.config.dataset}[/]")

        ret = self._rkllm.build(**build_kwargs)

        if ret != 0:
            self.console.print(f"\n[red]✗ Failed to build model (return code: {ret})[/]")
            raise RuntimeError(f"Failed to build model: return code {ret}")

        elapsed = time.time() - start_time
        logger.info("Model built successfully")
        self.console.print(f"\n         [green]✓ Model built[/] [dim]({elapsed:.1f}s)[/]\n")
        self._log_stage("build", 100, 100)

    def _export_model(self) -> None:
        """Export the model to RKLLM format."""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_stage("export", 0, 100)

        start_time = time.time()
        logger.info(f"Exporting model to {output_path}...")
        self.console.print("[cyan]Step 3/3:[/] Exporting RKLLM binary")
        self.console.print(f"         Output: [dim]{output_path}[/]")

        ret = self._rkllm.export_rkllm(export_path=str(output_path))

        if ret != 0:
            self.console.print(f"[red]✗ Failed to export model (return code: {ret})[/]")
            raise RuntimeError(f"Failed to export model: return code {ret}")

        elapsed = time.time() - start_time
        # Show file size if available
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            self.console.print(f"         [green]✓ Exported[/] [dim]({elapsed:.1f}s, {size_mb:.1f} MB)[/]")
        else:
            self.console.print(f"         [green]✓ Exported[/] [dim]({elapsed:.1f}s)[/]")

        logger.info(f"Model exported to {output_path}")
        self._log_stage("export", 100, 100)


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
