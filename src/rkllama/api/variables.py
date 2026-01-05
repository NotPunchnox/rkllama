import threading
from rkllama.config import is_debug_mode

isLocked = False

# Worker variables (lazy initialization to avoid circular import)
_worker_manager_rkllm = None


def get_worker_manager():
    """Get or create the global WorkerManager instance."""
    global _worker_manager_rkllm
    if _worker_manager_rkllm is None:
        from rkllama.api.worker import WorkerManager
        _worker_manager_rkllm = WorkerManager()
    return _worker_manager_rkllm


# For backwards compatibility - will be initialized on first access
class _LazyWorkerManager:
    def __getattr__(self, name):
        return getattr(get_worker_manager(), name)


worker_manager_rkllm = _LazyWorkerManager()


verrou = threading.Lock()

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
