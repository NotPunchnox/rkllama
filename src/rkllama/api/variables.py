import threading
import logging
from rkllama.config import is_debug_mode
from rkllama.api.worker import WorkerManager

logger = logging.getLogger("rkllama.variables")

isLocked = False

# Worker variables
worker_manager_rkllm = WorkerManager()


# Legacy global lock - kept for backward compatibility with process.py
# but should no longer be used directly. Use get_model_lock() instead.
verrou = threading.Lock()

# Per-model lock manager: each model gets its own lock so that
# requests to different models can proceed concurrently.
_model_locks = {}
_model_locks_meta_lock = threading.Lock()  # protects _model_locks dict itself


def get_model_lock(model_name: str) -> threading.Lock:
    """
    Return a per-model threading.Lock, creating one if it does not exist.
    Requests to different models will use different locks and can run in parallel.
    Requests to the same model will be serialized (RKLLM cannot handle concurrent
    requests on the same model handle).
    """
    if model_name not in _model_locks:
        with _model_locks_meta_lock:
            # Double-check inside the meta lock
            if model_name not in _model_locks:
                _model_locks[model_name] = threading.Lock()
                logger.debug(f"Created per-model lock for '{model_name}'")
    return _model_locks[model_name]


def remove_model_lock(model_name: str):
    """
    Remove the per-model lock when a model is unloaded.
    """
    with _model_locks_meta_lock:
        _model_locks.pop(model_name, None)

model_id = ""
system = "Tu es un assistant artificiel."
model_config = {}  # For storing model-specific configuration
generation_complete = False  # Flag to track completion status
debug_mode = is_debug_mode()  
stream_stats = {
    "total_requests": 0,
    "successful_responses": 0,
    "failed_responses": 0,
    "incomplete_streams": 0  # Streams that didn't receive done=true
}