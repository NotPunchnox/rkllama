"""FastAPI application for RKLlama server."""

import argparse
import os
import resource
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from importlib import resources as importlib_resources

import structlog
from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id.context import correlation_id
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.protocols.utils import get_path_with_query_string

import rkllama.config
from rkllama.api.worker import WorkerManager
from rkllama.logging import get_logger, setup_logging
from rkllama.telemetry import (
    instrument_fastapi,
    setup_telemetry,
    shutdown_telemetry,
)

# Set up structured logging
DEBUG_MODE = rkllama.config.is_debug_mode()
setup_logging(
    json_logs=not DEBUG_MODE,  # JSON in production, console in debug
    log_level="DEBUG" if DEBUG_MODE else "INFO",
)
logger = get_logger("rkllama.server")
access_logger = structlog.stdlib.get_logger("api.access")
# Telemetry providers (set during startup)
_tracer_provider = None
_meter_provider = None


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
    global _tracer_provider, _meter_provider

    # Startup
    logger.info("Starting RKLlama server")

    # Initialize telemetry
    from rkllama import __version__
    _tracer_provider, _meter_provider = setup_telemetry(
        service_name="rkllama",
        service_version=__version__,
        service_namespace="rkllama",
    )

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

    # Shutdown telemetry (flush pending spans/metrics)
    shutdown_telemetry(_tracer_provider, _meter_provider)
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

    # Instrument FastAPI with OpenTelemetry
    instrument_fastapi(app)

    # Add logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next) -> Response:
        structlog.contextvars.clear_contextvars()
        # These context vars will be added to all log entries emitted during the request
        request_id = correlation_id.get()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Log request start (skip health checks to reduce noise)
        if not request.url.path.startswith("/health"):
            access_logger.info(
                f"Request started: {request.method} {request.url.path}",
                http={"method": request.method, "path": request.url.path, "request_id": request_id},
                event_type="request_started",
            )

        start_time = time.perf_counter_ns()
        response = Response(status_code=500)
        try:
            response = await call_next(request)
        except Exception as e:
            structlog.stdlib.get_logger("api.error").exception(
                "Uncaught exception",
                exception_type=type(e).__name__,
                exception_message=str(e),
                request_path=str(request.url.path),
                request_method=request.method,
            )
            raise
        finally:
            process_time_ns = time.perf_counter_ns() - start_time
            process_time_ms = process_time_ns / 1_000_000
            status_code = response.status_code
            url = get_path_with_query_string(request.scope)  # type: ignore[arg-type]
            client_host = request.client.host if request.client else "unknown"
            client_port = request.client.port if request.client else 0
            http_method = request.method
            http_version = request.scope["http_version"]

            access_logger.info(
                f'{client_host}:{client_port} - "{http_method} {url} HTTP/{http_version}" {status_code}',
                http={
                    "url": str(request.url),
                    "status_code": status_code,
                    "method": http_method,
                    "request_id": request_id,
                    "version": http_version,
                },
                network={"client": {"ip": client_host, "port": client_port}},
                duration_ms=process_time_ms,
            )

            response.headers["X-Request-ID"] = request_id or ""
            response.headers["X-Process-Time"] = str(process_time_ms)
        return response

    # Add correlation ID middleware (must be after logging middleware)
    app.add_middleware(CorrelationIdMiddleware)

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
