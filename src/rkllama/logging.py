"""Structured logging configuration for RKLlama."""

import logging
import sys

import structlog
from structlog.types import EventDict, Processor


# https://github.com/hynek/structlog/issues/35#issuecomment-591321744
def rename_event_key(_, __, event_dict: EventDict) -> EventDict:
    """
    Log entries keep the text message in the `event` field, but some log aggregators
    use the `message` field. This processor moves the value from one field to
    the other.
    See https://github.com/hynek/structlog/issues/35#issuecomment-591321744
    """
    event_dict["message"] = event_dict.pop("event")
    return event_dict


def drop_color_message_key(_, __, event_dict: EventDict) -> EventDict:
    """
    Uvicorn logs the message a second time in the extra `color_message`, but we don't
    need it. This processor drops the key from the event dict if it exists.
    """
    event_dict.pop("color_message", None)
    return event_dict


def tracer_injection(_, __, event_dict: EventDict) -> EventDict:
    """
    Inject OpenTelemetry trace context into log events if available.
    """
    try:
        from opentelemetry import trace

        # Get current span context
        span = trace.get_current_span()
        span_context = span.get_span_context()

        # Add trace and span IDs to structlog event dictionary
        if span_context.is_valid:
            # Format as hex strings (standard OTEL format)
            event_dict["trace_id"] = format(span_context.trace_id, "032x")
            event_dict["span_id"] = format(span_context.span_id, "016x")
            event_dict["trace_flags"] = format(span_context.trace_flags, "02x")
        else:
            event_dict["trace_id"] = "0" * 32
            event_dict["span_id"] = "0" * 16
            event_dict["trace_flags"] = "00"
    except ImportError:
        # OpenTelemetry not installed, skip trace injection
        pass

    return event_dict


def setup_logging(json_logs: bool = False, log_level: str = "INFO"):
    """Configure structlog for the application.

    Args:
        json_logs: If True, output JSON formatted logs. If False, use console renderer.
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        A structlog logger instance
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        drop_color_message_key,
        tracer_injection,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        # We rename the `event` key to `message` only in JSON logs,
        # as some log aggregators look for the `message` key
        # but the pretty ConsoleRenderer looks for `event`
        shared_processors.append(rename_event_key)
        # Format the exception only for JSON logs, as we want to pretty-print them when
        # using the ConsoleRenderer
        shared_processors.append(structlog.processors.dict_tracebacks)

    structlog.configure(
        processors=shared_processors
        + [
            # Prepare event dict for `ProcessorFormatter`.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    log_renderer: structlog.types.Processor
    if json_logs:
        log_renderer = structlog.processors.JSONRenderer()
    else:
        log_renderer = structlog.dev.ConsoleRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONLY on `logging` entries that do NOT originate within
        # structlog.
        foreign_pre_chain=shared_processors,
        # These run on ALL entries after the pre_chain is done.
        processors=[
            # Remove _record & _from_structlog.
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            log_renderer,
        ],
    )

    handler = logging.StreamHandler()
    # Use OUR `ProcessorFormatter` to format all `logging` entries.
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    # Clear and propagate third-party loggers
    for _log in ["uvicorn", "uvicorn.error", "httpx"]:
        # Clear the log handlers for uvicorn loggers, and enable propagation
        # so the messages are caught by our root logger and formatted correctly
        # by structlog
        logging.getLogger(_log).handlers.clear()
        logging.getLogger(_log).propagate = True

    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Since we re-create the access logs ourselves, to add all information
    # in the structured log, we clear the handlers and prevent the logs
    # to propagate to a logger higher up in the hierarchy
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False

    def handle_exception(exc_type, exc_value, exc_traceback):
        """
        Log any uncaught exception instead of letting it be printed by Python
        (but leave KeyboardInterrupt untouched to allow users to Ctrl+C to stop)
        See https://stackoverflow.com/a/16993115/3641865
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        root_logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception

    # Return a structlog logger instance
    return structlog.get_logger()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger with the given name.

    Args:
        name: Logger name, typically __name__ or a dotted path like 'rkllama.api.worker'

    Returns:
        A bound structlog logger instance
    """
    return structlog.get_logger(name)
