"""RKLlama API module."""

# Lazy imports to avoid loading native libraries and Flask dependencies at import time
# This allows schemas to be imported independently for testing


def __getattr__(name):
    """Lazy import handler."""
    if name in (
        "RKLLMParam",
        "RKLLMInput",
        "RKLLMInferParam",
        "RKLLM_Handle_t",
        "userdata",
        "LLMCallState",
        "RKLLMInputType",
        "RKLLMInferMode",
        "RKLLM_AVAILABLE",
        "rkllm_lib",
    ):
        from . import classes

        return getattr(classes, name)
    elif name in ("callback", "callback_type"):
        from . import callback

        return getattr(callback, name)
    elif name == "Request":
        from . import process

        return getattr(process, name)
    elif name == "RKLLM":
        from . import rkllm

        return getattr(rkllm, name)
    elif name in (
        "worker_manager_rkllm",
        "verrou",
        "model_id",
        "system",
        "model_config",
        "generation_complete",
        "debug_mode",
        "stream_stats",
    ):
        from . import variables

        return getattr(variables, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
