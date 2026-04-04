import ctypes, sys
import numpy as np
from .classes import *
from .variables import *

global_status = -1
global_text = []
split_byte_data = bytes(b"")
last_embeddings = []
last_logits = []
global_metrics = []


# Definir la fonction de rappel
def callback_impl(result, userdata, status):
    global split_byte_data, global_status, global_text, last_embeddings, last_logits, global_metrics

    if status == LLMCallState.RKLLM_RUN_FINISH:
        global_status = status
        
        # Get the metrics for the current inference
        prefill_tokens = result.contents.perf.prefill_tokens
        generate_tokens = result.contents.perf.generate_tokens
        prefill_time_ms = int(result.contents.perf.prefill_time_ms) # Float to Int for compliance Ollama Pyhton package
        generate_time_ms = int(result.contents.perf.generate_time_ms) # Float to Int for compliance Ollama Pyhton package

        # Add the metrics to the global variable
        global_metrics.clear()
        global_metrics.append(prefill_tokens)
        global_metrics.append(generate_tokens)
        global_metrics.append(prefill_time_ms)
        global_metrics.append(generate_time_ms)
        
        print("\n")
        sys.stdout.flush()
    elif status == LLMCallState.RKLLM_RUN_ERROR:
        global_status = status
        print("Execution Error")
        sys.stdout.flush()
    elif status == LLMCallState.RKLLM_RUN_NORMAL:
        # Sauvegarder le texte du token de sortie et l'status d'execution de RKLLM
        global_status = status
        # Check if result or result.contents or result.contents.text is None
        try:
            # Add defensive checks to prevent None concatenation
            if result and result.contents and result.contents.text:
                text_bytes = result.contents.text
                if not isinstance(text_bytes, bytes):
                    # If not bytes, try to convert or use empty bytes
                    try:
                        text_bytes = bytes(text_bytes)
                    except:
                        text_bytes = b""
                        
                # Now safely concatenate
                try:
                    decoded_text = (split_byte_data + text_bytes).decode('utf-8')
                    global_text.append(decoded_text)
                    print(decoded_text, end='')
                    split_byte_data = bytes(b"")
                except UnicodeDecodeError:
                    # Handle incomplete UTF-8 sequences
                    split_byte_data += text_bytes
            else:
                # Handle case where text is None
                if split_byte_data:
                    try:
                        # Try to decode any accumulated bytes
                        decoded_text = split_byte_data.decode('utf-8')
                        global_text.append(decoded_text)
                        print(decoded_text, end='')
                        split_byte_data = bytes(b"")
                    except UnicodeDecodeError:
                        # Still incomplete, keep for next time
                        pass
            
            # --- EMBEDDINGS Part---
            if result and result.contents and result.contents.last_hidden_layer.hidden_states and result.contents.last_hidden_layer.embd_size > 0:
                num_tokens = result.contents.last_hidden_layer.num_tokens
                embd_size = result.contents.last_hidden_layer.embd_size

                # Only if tokens generated
                if num_tokens > 0:
                    # Construct the embed array
                    last_token_embedding = np.array([
                        result.contents.last_hidden_layer.hidden_states[(num_tokens-1) * embd_size + i] 
                        for i in range(embd_size)
                    ])
                    # Save the values
                    embeddings = {
                        'embedding': last_token_embedding,
                        'embd_size': embd_size,
                        'num_tokens': num_tokens
                    }

                    # Send to the global variable
                    last_embeddings.append(embeddings)
                    print(f"\n✅ Embeddings Shape: {last_token_embedding.shape}")

            # --- LOGITS Part ---
            if result and result.contents and result.contents.logits.logits and result.contents.logits.vocab_size > 0:
                vocab_size = result.contents.logits.vocab_size
                num_tokens = result.contents.logits.num_tokens

                if vocab_size > 0:
                    # Copy the logits array from C memory
                    total_size = vocab_size * max(num_tokens, 1)
                    array_type = ctypes.c_float * total_size
                    raw = array_type.from_address(ctypes.addressof(result.contents.logits.logits.contents))
                    logits_array = np.ctypeslib.as_array(raw).copy()

                    # If multiple tokens, take the last token's logits
                    if num_tokens > 1:
                        logits_array = logits_array[(num_tokens - 1) * vocab_size : num_tokens * vocab_size]
                    else:
                        logits_array = logits_array[:vocab_size]

                    last_logits.append({
                        'logits': logits_array,
                        'vocab_size': vocab_size,
                        'num_tokens': num_tokens
                    })
                    print(f"\n✅ Logits captured: vocab_size={vocab_size}, num_tokens={num_tokens}")

        except Exception as e:
            print(f"\nError processing callback: {str(e)}", end='')
            
        sys.stdout.flush()    
