import ctypes, sys
import time
from .classes import *
from .variables import *

# Definir la fonction de rappel
def callback_impl(result, donnees_utilisateur, status):
    global split_byte_data

    if status == LLMCallState.RKLLM_RUN_FINISH:
        global_status = status
        print("\n")
        sys.stdout.flush()
    elif status == LLMCallState.RKLLM_RUN_ERROR:
        global_status = status
        print("erreur d'execution")
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
        except Exception as e:
            print(f"\nError processing callback: {str(e)}", end='')
            
        sys.stdout.flush()