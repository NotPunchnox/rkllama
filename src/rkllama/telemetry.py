"""
OpenTelemetry telemetry setup for RKLlama.

This module provides distributed tracing and metrics collection via OpenTelemetry,
with OTLP gRPC export for Grafana Alloy, Jaeger, or any OTLP-compatible backend.

Features:
    - Distributed tracing with automatic FastAPI instrumentation
    - Application metrics with periodic OTLP export
    - Service resource attributes (name, version, environment)
    - Automatic no-op mode during pytest runs
    - Graceful shutdown with span/metric flushing

Quick Start:
    Telemetry is automatically configured when starting the RKLlama server.
    Set the OTLP endpoint via environment variable:

        export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
        rkllama_server --port 8080

Custom Spans:
    Add tracing to your own code:

        from rkllama.telemetry import get_tracer

        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("model-inference") as span:
            span.set_attribute("model.name", "qwen:7b")
            result = run_inference(prompt)

Custom Metrics:
    Add application metrics:

        from rkllama.telemetry import get_meter

        meter = get_meter(__name__)
        inference_counter = meter.create_counter(
            name="rkllama_inferences_total",
            description="Total inference requests",
            unit="1",
        )
        inference_counter.add(1, {"model": "qwen:7b", "status": "success"})

Span Attributes:
    Add context to the current span:

        from rkllama.telemetry import add_span_attributes

        add_span_attributes(
            model_name="qwen:7b",
            prompt_tokens=128,
            user_id="user-123",
        )

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP gRPC endpoint (default: Grafana Alloy k8s service)
    ENVIRONMENT: Deployment environment label (default: "local")
    PYTEST_CURRENT_TEST: If set, uses no-op providers (no export)

Kubernetes:
    For k8s deployments, telemetry exports to the Grafana Alloy receiver service.
    Override the endpoint if using a different collector:

        env:
          - name: OTEL_EXPORTER_OTLP_ENDPOINT
            value: "http://jaeger-collector:4317"
"""
# ruff: noqa: PLC0415

import os
from typing import Any

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = structlog.get_logger(__name__)


def create_resource_attributes(
    service_name: str,
    service_version: str | None = None,
    service_namespace: str | None = None,
    deployment_environment: str | None = None,
    **extra_attributes: Any,
) -> Resource:
    """
    Create OpenTelemetry resource attributes for the service.

    Args:
        service_name: Name of the service (e.g., "rkllama")
        service_version: Version of the service (e.g., "1.2.3-1")
        service_namespace: Namespace/group for the service (default: "rkllama")
        deployment_environment: Environment (e.g., "production", "staging", "local")
        **extra_attributes: Additional custom attributes

    Returns:
        Resource with configured attributes
    """
    attributes = {
        "service.name": service_name,
    }

    if service_version:
        attributes["service.version"] = service_version

    if service_namespace:
        attributes["service.namespace"] = service_namespace

    if deployment_environment:
        attributes["deployment.environment"] = deployment_environment
    else:
        attributes["deployment.environment"] = os.environ.get("ENVIRONMENT", "local")

    # Add any extra custom attributes
    attributes.update(extra_attributes)

    # Add hostname and process info
    attributes["host.name"] = os.environ.get("HOSTNAME", "unknown")
    attributes["process.pid"] = str(os.getpid())

    return Resource.create(attributes)


def setup_telemetry(
    service_name: str,
    service_version: str | None = None,
    service_namespace: str = "rkllama",
    deployment_environment: str | None = None,
    otlp_endpoint: str | None = None,
    enable_traces: bool = True,
    enable_metrics: bool = True,
    enable_console_export: bool = False,
    **resource_attributes: Any,
) -> tuple[TracerProvider | None, MeterProvider | None]:
    """
    Set up OpenTelemetry tracing and metrics with OTLP exporters.

    Called automatically by RKLlama server during startup. You typically don't
    need to call this directly unless building custom tooling.

    Args:
        service_name: Name of the service (default: "rkllama")
        service_version: Version of the service (from __version__)
        service_namespace: Namespace for grouping services (default: "rkllama")
        deployment_environment: Environment label (reads from ENVIRONMENT env var if not set)
        otlp_endpoint: OTLP gRPC endpoint (reads from OTEL_EXPORTER_OTLP_ENDPOINT env var if not set)
        enable_traces: Enable distributed tracing (default: True)
        enable_metrics: Enable metrics collection (default: True)
        enable_console_export: Also print spans/metrics to console for debugging
        **resource_attributes: Additional resource attributes to attach

    Returns:
        Tuple of (TracerProvider, MeterProvider) - store these for shutdown

    Environment Variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP gRPC endpoint URL
        ENVIRONMENT: Deployment environment (production, staging, local)
        PYTEST_CURRENT_TEST: If set, uses no-op providers (no network calls)

    Example:
        tracer_provider, meter_provider = setup_telemetry(
            service_name="rkllama",
            service_version="1.2.3-1",
            otlp_endpoint="http://localhost:4317",
        )
    """
    # Skip telemetry setup during tests - use no-op providers
    if os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("Running in pytest - using no-op telemetry providers")
        from opentelemetry.sdk.metrics import MeterProvider as NoOpMeterProvider
        from opentelemetry.sdk.trace import TracerProvider as NoOpTracerProvider

        # Create minimal no-op providers (won't export anything)
        tracer_provider = NoOpTracerProvider()
        meter_provider = NoOpMeterProvider()

        # Set them globally so get_tracer/get_meter work
        trace.set_tracer_provider(tracer_provider)
        metrics.set_meter_provider(meter_provider)

        return tracer_provider, meter_provider

    # Get deployment environment from env var if not provided
    if deployment_environment is None:
        deployment_environment = os.environ.get("ENVIRONMENT", "local")

    # Get OTLP endpoint from env var if not provided
    if otlp_endpoint is None:
        otlp_endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://grafana-k8s-monitoring-alloy-receiver.observability.svc.cluster.local:4317",
        )

    logger.info(
        "Initializing OpenTelemetry",
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        environment=deployment_environment,
        traces_enabled=enable_traces,
        metrics_enabled=enable_metrics,
    )

    # Create resource with service metadata
    resource = create_resource_attributes(
        service_name=service_name,
        service_version=service_version,
        service_namespace=service_namespace,
        deployment_environment=deployment_environment,
        **resource_attributes,
    )

    tracer_provider = None
    meter_provider = None

    # Setup Traces
    if enable_traces:
        tracer_provider = TracerProvider(resource=resource)

        # Add OTLP exporter
        otlp_span_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,  # Use insecure for internal cluster communication
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_span_exporter))

        # Add console exporter for debugging
        if enable_console_export:
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
            )

            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

        # Set the global tracer provider
        trace.set_tracer_provider(tracer_provider)
        logger.info("OpenTelemetry tracing initialized", endpoint=otlp_endpoint)

    # Setup Metrics
    if enable_metrics:
        # Create OTLP metric exporter
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint,
            insecure=True,
        )

        # Create metric reader with periodic export
        metric_reader = PeriodicExportingMetricReader(
            otlp_metric_exporter,
            export_interval_millis=60000,  # Export every 60 seconds
        )

        readers = [metric_reader]

        # Add console exporter for debugging
        if enable_console_export:
            from opentelemetry.sdk.metrics.export import (
                ConsoleMetricExporter,
            )

            console_metric_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=60000,
            )
            readers.append(console_metric_reader)

        # Create and set meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=readers,
        )
        metrics.set_meter_provider(meter_provider)
        logger.info("OpenTelemetry metrics initialized", endpoint=otlp_endpoint)

    return tracer_provider, meter_provider


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer instance for creating custom spans.

    Args:
        name: Name for the tracer, typically __name__ of the calling module

    Returns:
        Tracer instance

    Example:
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("my-operation"):
            # your code here
            pass
    """
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """
    Get a meter instance for creating custom metrics.

    Args:
        name: Name for the meter, typically __name__ of the calling module

    Returns:
        Meter instance

    Example:
        meter = get_meter(__name__)
        counter = meter.create_counter(
            name="requests_total",
            description="Total number of requests",
            unit="1"
        )
        counter.add(1, {"endpoint": "/api/feeds"})
    """
    return metrics.get_meter(name)


def instrument_fastapi(app: Any) -> None:
    """
    Automatically instrument a FastAPI application with OpenTelemetry.

    Called automatically by RKLlama server. Adds automatic tracing for all
    HTTP requests including method, URL, status code, and duration.

    Args:
        app: FastAPI application instance
    """
    from opentelemetry.instrumentation.fastapi import (
        FastAPIInstrumentor,
    )

    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI instrumentation enabled")

def shutdown_telemetry(
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
) -> None:
    """
    Gracefully shutdown telemetry providers, flushing pending spans and metrics.

    Called automatically by RKLlama server during shutdown. Ensures all telemetry
    data is exported before the process exits.

    Args:
        tracer_provider: TracerProvider instance from setup_telemetry
        meter_provider: MeterProvider instance from setup_telemetry
    """
    if tracer_provider:
        tracer_provider.shutdown()
        logger.info("Tracer provider shutdown complete")

    if meter_provider:
        meter_provider.shutdown()
        logger.info("Meter provider shutdown complete")


def add_span_attributes(**attributes: Any) -> None:
    """
    Add custom attributes to the current span.

    Use this to enrich traces with business context for filtering and debugging
    in your observability tool.

    Args:
        **attributes: Key-value pairs to add as span attributes

    Example:
        from rkllama.telemetry import add_span_attributes

        async def generate_response(model: str, prompt: str):
            add_span_attributes(
                model_name=model,
                prompt_tokens=len(prompt.split()),
                operation="inference",
            )
            # ... run inference ...
    """
    span = trace.get_current_span()
    if span.is_recording():
        for key, value in attributes.items():
            # Convert value to string if it's not a basic type
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, str(value))
