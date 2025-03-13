import threading
from config import is_debug_mode

isLocked = False
global_status = -1
global_text = []
split_byte_data = bytes(b"")

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