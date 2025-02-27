import os
import json
import datetime
import logging
import threading
import time

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser("~/RKLLAMA/rkllama_debug.log"))
    ]
)
logger = logging.getLogger("rkllama.debug")

class StreamDebugger:
    """
    Utility class to debug streaming responses
    """
    def __init__(self, model_name, enable_logging=True):
        self.model_name = model_name
        self.enable_logging = enable_logging
        self.start_time = time.time()
        self.chunks = []
        self.done_received = False
        self.total_tokens = 0
        
    def log_chunk(self, chunk, is_done=False):
        """Log a streaming chunk"""
        if not self.enable_logging:
            return
        
        self.chunks.append(chunk)
        self.total_tokens += 1
        
        if is_done:
            self.done_received = True
            logger.debug(f"DONE message received for model {self.model_name}")
            
        if len(self.chunks) % 10 == 0:  # Log every 10 chunks
            elapsed = time.time() - self.start_time
            logger.debug(f"Stream progress: {len(self.chunks)} chunks, {elapsed:.2f}s elapsed, done={self.done_received}")
    
    def finalize(self):
        """Called at the end of streaming to log final stats"""
        if not self.enable_logging:
            return
            
        elapsed = time.time() - self.start_time
        logger.debug(f"Stream completed: {len(self.chunks)} total chunks, {elapsed:.2f}s elapsed, done={self.done_received}")
        
        if not self.done_received:
            logger.warning(f"Stream for {self.model_name} completed without receiving done=true message")
            
        return {
            "model": self.model_name,
            "chunks": len(self.chunks),
            "elapsed_time": elapsed,
            "done_received": self.done_received,
            "tokens_per_second": self.total_tokens / elapsed if elapsed > 0 else 0
        }

def check_response_format(response_text):
    """
    Utility function to analyze response format and find issues
    """
    lines = response_text.split("\n")
    issues = []
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        
        try:
            data = json.loads(line)
            # Check for required fields
            if "model" not in data:
                issues.append(f"Line {i+1}: Missing 'model' field")
            if "message" not in data:
                issues.append(f"Line {i+1}: Missing 'message' field")
            if "done" not in data:
                issues.append(f"Line {i+1}: Missing 'done' field")
                
            # Check if this is the last line
            if i == len(lines) - 2 or (i == len(lines) - 1 and lines[-1].strip()):
                if not data.get("done", False):
                    issues.append(f"Last line doesn't have done=true")
        except json.JSONDecodeError:
            issues.append(f"Line {i+1}: Invalid JSON: {line}")
    
    # Check if any line has done=true
    has_done = any('"done":true' in line or '"done": true' in line for line in lines)
    if not has_done:
        issues.append("No line has done=true")
        
    return issues
