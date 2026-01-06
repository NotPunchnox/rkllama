"""FastAPI application for RKLlama server."""

import argparse
import os
import resource
import subprocess
import sys
from contextlib import asynccontextmanager
from importlib import resources as importlib_resources

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import rkllama.config
from rkllama.api.worker import WorkerManager
from rkllama.logging import get_logger, setup_logging

# Set up structured logging
DEBUG_MODE = rkllama.config.is_debug_mode()
setup_logging(
    json_logs=not DEBUG_MODE,  # JSON in production, console in debug
    log_level="DEBUG" if DEBUG_MODE else "INFO",
)
logger = get_logger("rkllama.server")


def print_color(message: str, color: str) -> None:
    """Display colored messages in terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, colors['reset'])}{message}{colors['reset']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    logger.info("Starting RKLlama server")
    app.state.worker_manager = WorkerManager()

    # IMPORTANT: Share the same WorkerManager instance with server_utils
    # This bridges the FastAPI routers with the legacy server_utils code
    import rkllama.api.variables as variables
    variables._worker_manager_rkllm = app.state.worker_manager

    app.state.model_id = ""
    app.state.system = "You are an AI assistant."
    app.state.model_config = {}
    app.state.generation_complete = False
    app.state.stream_stats = {
        "total_requests": 0,
        "successful_responses": 0,
        "failed_responses": 0,
        "incomplete_streams": 0,
    }
    logger.info("WorkerManager initialized")

    yield

    # Shutdown
    logger.info("Shutting down RKLlama server")
    if hasattr(app.state, "worker_manager"):
        app.state.worker_manager.stop_all()
    logger.info("Server shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from rkllama import __version__

    app = FastAPI(
        title="RKLlama",
        description="Ollama-compatible API for Rockchip NPU",
        version=__version__,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and register routers
    from rkllama.server.routers import modelfile, native, ollama, openai

    app.include_router(native.router, tags=["Native"])
    app.include_router(ollama.router, prefix="/api", tags=["Ollama API"])
    app.include_router(openai.router, prefix="/v1", tags=["OpenAI API"])
    app.include_router(modelfile.router, prefix="/api/modelfile", tags=["Modelfile"])

    @app.get("/")
    async def root():
        return {
            "message": "Welcome to RKLlama with Ollama API compatibility!",
            "github": "https://github.com/notpunhnox/rkllama",
        }

    @app.get("/health", tags=["Health"])
    async def health_live():
        """Liveness probe - is the server process running?"""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["Health"])
    async def health_ready():
        """Readiness probe - is the server ready to accept requests?"""
        from rkllama import __version__

        checks = {
            "worker_manager": False,
            "models_path": False,
        }

        # Check worker manager is initialized
        if hasattr(app.state, "worker_manager") and app.state.worker_manager is not None:
            checks["worker_manager"] = True

        # Check models directory is accessible
        models_path = rkllama.config.get_path("models")
        if os.path.exists(models_path) and os.path.isdir(models_path):
            checks["models_path"] = True

        all_healthy = all(checks.values())

        return {
            "status": "ok" if all_healthy else "degraded",
            "version": __version__,
            "checks": checks,
        }

    return app


# Create the app instance
app = create_app()


def main():
    """Entry point for the server."""
    import uvicorn

    # Define CLI arguments
    parser = argparse.ArgumentParser(description="RKLLM server initialization with configurable options.")
    parser.add_argument("--processor", type=str, help="Processor: rk3588/rk3576.")
    parser.add_argument("--port", type=str, help="Port for the server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--models", type=str, help="Path where models will be loaded from")
    args = parser.parse_args()

    # Load arguments into config
    rkllama.config.load_args(args)
    rkllama.config.validate()

    # Update debug mode
    global DEBUG_MODE
    DEBUG_MODE = rkllama.config.is_debug_mode()
    if DEBUG_MODE:
        setup_logging(json_logs=False, log_level="DEBUG")
        print_color("Debug mode enabled", "yellow")
        rkllama.config.display()
        os.environ["RKLLAMA_DEBUG"] = "1"

    # Get server configuration
    port = int(rkllama.config.get("server", "port", "8080"))
    host = rkllama.config.get("server", "host", "0.0.0.0")

    # Check and configure processor
    processor = rkllama.config.get("platform", "processor", None)
    if not processor:
        print_color("Error: processor not configured", "red")
        sys.exit(1)

    if processor not in ["rk3588", "rk3576"]:
        print_color("Error: Invalid processor. Please enter rk3588 or rk3576.", "red")
        sys.exit(1)

    if os.getuid() == 0:
        print_color(f"Setting the frequency for the {processor} platform...", "cyan")
        library_path = importlib_resources.files("rkllama.lib") / f"fix_freq_{processor}.sh"
        debug_param = "1" if DEBUG_MODE else "0"
        command = f"bash {library_path} {debug_param}"
        subprocess.run(command, shell=True)

    # Set resource limits if running as root
    if os.getuid() == 0:
        resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    print_color(f"Starting RKLlama API at http://{host}:{port}", "blue")

    # Run with uvicorn
    # Disable uvicorn's default access log - we handle logging via structlog
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if DEBUG_MODE else "info",
        access_log=False,  # Disable default access log format
    )


if __name__ == "__main__":
    main()
