import ctypes
import numpy as np
from .classes import *
from .variables import *
from rkllama.logging import get_logger

logger = get_logger("rkllama.callback")

global_status = -1
global_text = []
split_byte_data = bytes(b"")
last_embeddings = []
last_perf_stats = None  # Performance stats from last inference


def callback_impl(result, userdata, status):
    """RKLLM callback for token streaming and embeddings."""
    global split_byte_data, global_status, global_text, last_embeddings, last_perf_stats

    if status == LLMCallState.RKLLM_RUN_FINISH:
        global_status = status

        # Capture performance stats on completion
        if result and result.contents:
            try:
                perf = result.contents.perf
                last_perf_stats = {
                    "prefill_time_ms": perf.prefill_time_ms,
                    "prefill_tokens": perf.prefill_tokens,
                    "generate_time_ms": perf.generate_time_ms,
                    "generate_tokens": perf.generate_tokens,
                    "memory_usage_mb": perf.memory_usage_mb,
                }
            except Exception:
                last_perf_stats = None

    elif status == LLMCallState.RKLLM_RUN_ERROR:
        global_status = status
        logger.error("RKLLM execution error")

    elif status == LLMCallState.RKLLM_RUN_NORMAL:
        global_status = status
        try:
            # Add defensive checks to prevent None concatenation
            if result and result.contents and result.contents.text:
                text_bytes = result.contents.text
                if not isinstance(text_bytes, bytes):
                    try:
                        text_bytes = bytes(text_bytes)
                    except:
                        text_bytes = b""

                # Now safely concatenate
                try:
                    decoded_text = (split_byte_data + text_bytes).decode('utf-8')
                    global_text.append(decoded_text)
                    split_byte_data = bytes(b"")
                except UnicodeDecodeError:
                    # Handle incomplete UTF-8 sequences
                    split_byte_data += text_bytes
            else:
                # Handle case where text is None
                if split_byte_data:
                    try:
                        decoded_text = split_byte_data.decode('utf-8')
                        global_text.append(decoded_text)
                        split_byte_data = bytes(b"")
                    except UnicodeDecodeError:
                        pass

            # --- EMBEDDINGS Part---
            if result and result.contents and result.contents.last_hidden_layer.hidden_states:
                num_tokens = result.contents.last_hidden_layer.num_tokens
                embd_size = result.contents.last_hidden_layer.embd_size
                total_size = num_tokens * embd_size

                # Convert pointer to numpy
                array_type = ctypes.c_float * total_size
                raw = array_type.from_address(
                    ctypes.addressof(result.contents.last_hidden_layer.hidden_states.contents)
                )
                embeddings = np.ctypeslib.as_array(raw)
                embeddings = embeddings.reshape(num_tokens, embd_size)

                last_embeddings.append(embeddings)
                logger.debug("Embeddings generated", shape=embeddings.shape)

        except Exception as e:
            logger.error("Error processing callback", error=str(e))
