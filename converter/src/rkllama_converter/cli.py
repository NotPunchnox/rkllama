"""
Command-line interface for RKLlama Converter.

Provides a modern CLI with rich output for converting HuggingFace models
to RKLLM format for Rockchip NPU deployment.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .device import DeviceType, detect_device, get_device_info, get_device_or_fallback
from .modelfile import ModelfileConfig, save_modelfile

app = typer.Typer(
    name="rkllama-convert",
    help="Convert HuggingFace models to RKLLM format for Rockchip NPU.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


class Quantization(str, Enum):
    """Supported quantization formats."""

    Q4_0 = "Q4_0"
    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"
    Q8_K_M = "Q8_K_M"


class CliDeviceType(str, Enum):
    """Device types for CLI."""

    cuda = "cuda"
    rocm = "rocm"
    cpu = "cpu"
    auto = "auto"


def _show_device_info(device: DeviceType) -> None:
    """Display device information panel."""
    info = get_device_info(device)

    if device == DeviceType.CPU:
        console.print(f"[cyan]Device:[/] CPU ({info.get('cores', '?')} cores)")
    else:
        name = info.get("name", "Unknown")
        memory = info.get("memory_gb", 0)
        console.print(f"[cyan]Device:[/] {name} ({memory:.1f} GB)")

        if device == DeviceType.ROCM:
            console.print(f"[dim]HIP Version: {info.get('hip_version', 'unknown')}[/]")
        elif device == DeviceType.CUDA:
            console.print(f"[dim]Compute Capability: {info.get('compute_capability', 'unknown')}[/]")


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"rkllama-convert version {__version__}")
        raise typer.Exit()


@app.command()
def convert(
    model_id: Annotated[
        str,
        typer.Argument(help="HuggingFace model ID (e.g., Qwen/Qwen2.5-7B-Instruct)"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for converted model"),
    ] = Path("./output"),
    quantization: Annotated[
        Quantization,
        typer.Option("--quant", "-q", help="Quantization format"),
    ] = Quantization.Q4_0,
    max_context: Annotated[
        int,
        typer.Option("--context", "-c", help="Maximum context length in tokens"),
    ] = 4096,
    dtype: Annotated[
        str,
        typer.Option("--dtype", "-d", help="Model dtype: float16 or float32"),
    ] = "float16",
    device: Annotated[
        CliDeviceType,
        typer.Option("--device", help="Compute device: cuda, rocm, cpu, or auto"),
    ] = CliDeviceType.auto,
    token: Annotated[
        str | None,
        typer.Option("--token", "-t", help="HuggingFace token for private models", envvar="HF_TOKEN"),
    ] = None,
    # Modelfile options
    system_prompt: Annotated[
        str | None,
        typer.Option("--system", "-s", help="System prompt for the model"),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option("--temp", help="Generation temperature"),
    ] = 0.7,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Top-K sampling parameter"),
    ] = 40,
    top_p: Annotated[
        float,
        typer.Option("--top-p", help="Nucleus sampling parameter"),
    ] = 0.9,
    enable_thinking: Annotated[
        bool,
        typer.Option("--thinking/--no-thinking", help="Enable reasoning mode"),
    ] = False,
    # Target platform options
    target_platform: Annotated[
        str,
        typer.Option("--platform", "-p", help="Target platform: rk3588, rk3576, rk3562, rv1126b"),
    ] = "rk3588",
    num_npu_core: Annotated[
        int,
        typer.Option("--npu-cores", "-n", help="Number of NPU cores (RK3588: 1-3, RK3576: 1-2)"),
    ] = 3,
) -> None:
    """
    Convert a HuggingFace model to RKLLM format.

    This command downloads a model from HuggingFace, quantizes it using
    the official rkllm-toolkit, and generates an RKLLM binary file along
    with a Modelfile for use with RKLlama server.

    Example:
        rkllama-convert Qwen/Qwen2.5-7B-Instruct -q Q8_0 -o ./models/qwen -p rk3588
    """
    # Resolve device
    if device == CliDeviceType.auto:
        actual_device = detect_device()
    else:
        actual_device = get_device_or_fallback(DeviceType(device.value))

    # Show conversion info panel
    model_name = model_id.split("/")[-1]
    console.print()
    console.print(
        Panel(
            f"[bold]Model:[/] {model_id}\n"
            f"[bold]Output:[/] {output_dir / model_name}\n"
            f"[bold]Quantization:[/] {quantization.value}\n"
            f"[bold]Target Platform:[/] {target_platform.upper()}\n"
            f"[bold]NPU Cores:[/] {num_npu_core}\n"
            f"[bold]Context Length:[/] {max_context:,} tokens",
            title="[bold blue]RKLlama Converter (Official Toolkit)[/]",
            border_style="blue",
        )
    )

    _show_device_info(actual_device)
    console.print()

    # Import here to avoid slow startup
    from .converter import ConversionConfig, HuggingFaceToRKLLMConverter

    # Create configuration
    config = ConversionConfig(
        model_id=model_id,
        output_dir=str(output_dir),
        quantization=quantization.value,
        max_context_len=max_context,
        dtype=dtype,
        device=actual_device.value if actual_device != DeviceType.ROCM else "cuda",
        token=token,
        target_platform=target_platform,
        num_npu_core=num_npu_core,
    )

    # Run conversion
    import time
    start_time = time.time()

    try:
        converter = HuggingFaceToRKLLMConverter(config)

        # Create output directory
        Path(config.output_path).mkdir(parents=True, exist_ok=True)

        # Step 1: Download/prepare model
        console.print("[bold]Phase 1: Model Preparation[/]\n")
        with console.status("[cyan]Downloading/preparing model...[/]", spinner="dots"):
            converter._prepare_model()
        console.print("[green]✓[/] Model prepared\n")

        # Step 2: Convert to RKLLM (this has its own detailed progress)
        console.print("[bold]Phase 2: RKLLM Conversion[/]")
        console.print("[dim]This step may take several minutes depending on model size...[/]\n")
        converter._generate_rkllm_file()

        # Step 3: Create Modelfile
        console.print("\n[bold]Phase 3: Finalization[/]\n")
        console.print("[cyan]Creating Modelfile...[/]")
        modelfile_config = ModelfileConfig(
            model_file=f"{model_name}.rkllm",
            huggingface_path=model_id,
            system=system_prompt or "You are a helpful AI assistant.",
            temperature=temperature,
            num_ctx=max_context,
            top_k=top_k,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        save_modelfile(modelfile_config, config.output_path)
        console.print("[green]✓[/] Modelfile created")

        # Step 4: Save metadata
        console.print("[cyan]Saving metadata...[/]")
        converter._save_metadata(config.output_path)
        console.print("[green]✓[/] Metadata saved")

        total_time = time.time() - start_time
        console.print(f"\n[dim]Total time: {total_time:.1f}s[/]")

    except Exception as e:
        console.print(f"\n[red bold]Error:[/] {e}")
        raise typer.Exit(1) from None

    # Success message
    console.print()
    console.print(
        Panel(
            f"[green]Model converted successfully![/]\n\n"
            f"Output directory: [cyan]{config.output_path}[/]\n\n"
            f"Files created:\n"
            f"  - {model_name}.rkllm\n"
            f"  - Modelfile\n"
            f"  - metadata.json\n\n"
            f"[dim]Copy to your RKLlama models directory to use.[/]",
            title="[bold green]Conversion Complete[/]",
            border_style="green",
        )
    )


@app.command()
def info(
    model_id: Annotated[
        str,
        typer.Argument(help="HuggingFace model ID to inspect"),
    ],
    token: Annotated[
        str | None,
        typer.Option("--token", "-t", help="HuggingFace token", envvar="HF_TOKEN"),
    ] = None,
) -> None:
    """
    Show information about a HuggingFace model before conversion.

    This fetches model metadata from HuggingFace and displays architecture,
    size, and recommended conversion settings.
    """
    console.print(f"\n[cyan]Fetching info for:[/] {model_id}\n")

    try:
        from huggingface_hub import model_info as hf_model_info

        with console.status("[cyan]Loading model info...[/]"):
            info = hf_model_info(model_id, token=token)

        table = Table(title=f"Model: {model_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Model ID", info.id)
        table.add_row("Author", info.author or "Unknown")
        table.add_row("Downloads", f"{info.downloads:,}" if info.downloads else "N/A")
        table.add_row("Likes", str(info.likes) if info.likes else "N/A")
        table.add_row("License", info.card_data.license if info.card_data and info.card_data.license else "Unknown")
        table.add_row("Pipeline Tag", info.pipeline_tag or "N/A")

        if info.safetensors:
            total_size = sum(info.safetensors.parameters.values())
            table.add_row("Parameters", f"{total_size / 1e9:.2f}B")

        console.print(table)

        # Recommendations
        console.print("\n[bold]Recommended settings:[/]")
        console.print("  Quantization: Q4_0 (smallest) or Q8_0 (highest quality)")
        console.print("  Context: 4096 (default) or check model card for max")

    except Exception as e:
        console.print(f"[red]Error fetching model info:[/] {e}")
        raise typer.Exit(1) from None


@app.command(name="list-quants")
def list_quants() -> None:
    """
    List available quantization formats and their characteristics.

    Shows the supported quantization schemes with their RKLLM mapping,
    typical size reduction, and quality tradeoffs.
    """
    table = Table(title="Quantization Formats")
    table.add_column("Format", style="cyan", no_wrap=True)
    table.add_column("RKLLM Type", style="green")
    table.add_column("Bits", justify="center")
    table.add_column("Size vs FP16", justify="center")
    table.add_column("Quality", style="yellow")
    table.add_column("Status")

    table.add_row("Q4_0", "w4a16", "4", "~25%", "Good", "[green]Supported[/]")
    table.add_row("Q4_K_M", "w4a16_g128", "4", "~25%", "Better", "[yellow]Partial[/]")
    table.add_row("Q8_0", "w8a8", "8", "~50%", "Excellent", "[green]Supported[/]")
    table.add_row("Q8_K_M", "w8a8_g512", "8", "~50%", "Best", "[yellow]Partial[/]")

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Note: K_M variants use grouped quantization for better accuracy.[/]")
    console.print("[dim]w4a16 = 4-bit weights, 16-bit activations[/]")
    console.print("[dim]w8a8 = 8-bit weights, 8-bit activations[/]")


@app.command(name="list-devices")
def list_devices() -> None:
    """
    List available compute devices for model conversion.

    Detects available GPU and CPU devices and shows their capabilities.
    """
    console.print("\n[bold]Available Devices[/]\n")

    detected = detect_device()
    console.print(f"[green]Auto-detected:[/] {detected.value}\n")

    # Always show CPU
    cpu_info = get_device_info(DeviceType.CPU)
    console.print(f"[cyan]CPU:[/] {cpu_info.get('cores', '?')} cores")

    # Check for GPU
    try:
        import torch

        if torch.cuda.is_available():
            gpu_device = detected if detected in (DeviceType.CUDA, DeviceType.ROCM) else DeviceType.CUDA
            gpu_info = get_device_info(gpu_device)

            if gpu_info.get("available", False):
                console.print(f"[cyan]{gpu_device.value.upper()}:[/] {gpu_info.get('name', 'Unknown')}")
                console.print(f"  Memory: {gpu_info.get('memory_gb', 0):.1f} GB")

                if gpu_device == DeviceType.ROCM:
                    console.print(f"  HIP Version: {gpu_info.get('hip_version', 'unknown')}")
                else:
                    console.print(f"  Compute Capability: {gpu_info.get('compute_capability', 'unknown')}")
        else:
            console.print("[yellow]No GPU detected[/]")
    except ImportError:
        console.print("[yellow]PyTorch not installed - GPU detection unavailable[/]")

    console.print()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-v", callback=version_callback, is_eager=True, help="Show version"),
    ] = None,
) -> None:
    """
    RKLlama Converter - Convert HuggingFace models to RKLLM format.

    This tool converts transformer models from HuggingFace to RKLLM binary
    format for deployment on Rockchip NPU devices (RK3588/RK3576).

    For more information, see: https://github.com/notpunchnox/rkllama
    """
    pass


if __name__ == "__main__":
    app()
