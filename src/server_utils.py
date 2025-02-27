import threading
import json
import time
import datetime
import logging
import os
from transformers import AutoTokenizer
from flask import jsonify, Response
import src.variables as variables

# Check for debug mode
DEBUG_MODE = os.environ.get("RKLLAMA_DEBUG", "0").lower() in ["1", "true", "yes", "on"]

# Set up logging based on debug mode
logging_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser("~/RKLLAMA/rkllama_debug.log")) if DEBUG_MODE else logging.NullHandler()
    ]
)
logger = logging.getLogger("rkllama.server_utils")

class RequestWrapper:
    """A class that mimics Flask's request object for custom request handling"""
    def __init__(self, json_data, path="/"):
        self.json = json_data
        self.path = path


def process_ollama_chat_request(modele_rkllm, model_name, messages, system="", stream=True):
    """
    Process an Ollama-style chat request
    
    Args:
        modele_rkllm: The model instance
        model_name: Name of the model
        messages: List of message objects
        system: System prompt
        stream: Whether to stream the response
        
    Returns:
        Flask response with generated text
    """
    if DEBUG_MODE:
        logger.debug(f"Processing Ollama chat request for model: {model_name}, stream: {stream}")
    
    # Save original system prompt and set new one if provided
    original_system = variables.system
    if system:
        if DEBUG_MODE:
            logger.debug(f"Setting system prompt: {system}")
        variables.system = system
        
    try:
        # Reset global status for new request
        variables.global_status = -1
        variables.generation_complete = False
        
        # Set up tokenizer
        if DEBUG_MODE:
            logger.debug(f"Setting up tokenizer for model_id: {variables.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(variables.model_id, trust_remote_code=True)
        supports_system_role = "raise_exception('System role not supported')" not in tokenizer.chat_template
        
        # Prepare prompt
        if variables.system and supports_system_role:
            if DEBUG_MODE:
                logger.debug("Adding system prompt to messages")
            prompt = [{"role": "system", "content": variables.system}] + messages
        else:
            prompt = messages
            
        # Validate message sequence
        for i in range(1, len(prompt)):
            if prompt[i]["role"] == prompt[i - 1]["role"]:
                err_msg = "Roles must alternate between 'user' and 'assistant'"
                if DEBUG_MODE:
                    logger.error(err_msg)
                raise ValueError(err_msg)
                
        # Apply chat template
        if DEBUG_MODE:
            logger.debug("Applying chat template")
        prompt_tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        prompt_token_count = len(prompt_tokens)
        if DEBUG_MODE:
            logger.debug(f"Prompt token count: {prompt_token_count}")
        
        if stream:
            def generate():
                # Set up model thread
                if DEBUG_MODE:
                    logger.debug("Starting model inference thread")
                thread_modele = threading.Thread(target=modele_rkllm.run, args=(prompt_tokens,))
                thread_modele.start()
                
                thread_finished = False
                count = 0
                start = time.time()
                full_response = ""
                last_check_time = time.time()
                check_interval = 0.1  # Check thread status every 100ms
                final_sent = False
                
                while not thread_finished or len(variables.global_text) > 0:
                    # Process any new tokens from the model
                    tokens_processed = False
                    while len(variables.global_text) > 0:
                        count += 1
                        output_text = variables.global_text.pop(0)
                        full_response += output_text
                        tokens_processed = True
                        
                        # Format response in Ollama style
                        ollama_chunk = {
                            "model": model_name,
                            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            "message": {
                                "role": "assistant",
                                "content": output_text
                            },
                            "done": False  # Always false until final message
                        }
                        
                        if DEBUG_MODE:
                            logger.debug(f"Streaming chunk: {len(output_text)} chars, done=False")
                        yield f"{json.dumps(ollama_chunk)}\n"
                        last_check_time = time.time()  # Reset the check timer
                    
                    # Check if thread is still running periodically
                    current_time = time.time()
                    if current_time - last_check_time >= check_interval:
                        thread_modele.join(timeout=0.005)
                        thread_finished = not thread_modele.is_alive()
                        last_check_time = current_time
                        
                        # If no tokens were processed this loop and model is done, break
                        if thread_finished and not tokens_processed and len(variables.global_text) == 0:
                            break
                
                # Always send the final message with done=true
                if not final_sent:
                    total_time = time.time() - start
                    # Send final message with done=true and complete response
                    final_chunk = {
                        "model": model_name,
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "message": {
                            "role": "assistant",
                            "content": full_response
                        },
                        "done": True,
                        "total_duration": total_time * 1000000000,
                        "prompt_eval_count": prompt_token_count,
                        "prompt_eval_duration": 0,
                        "eval_count": count,
                        "eval_duration": total_time * 1000000000
                    }
                    if DEBUG_MODE:
                        logger.debug(f"Sending final message: {len(full_response)} chars, done=True")
                    yield f"{json.dumps(final_chunk)}\n"
                    final_sent = True
                    
                variables.generation_complete = True
                if DEBUG_MODE:
                    logger.debug(f"Generation complete. Total tokens: {count}, Total time: {total_time:.2f}s")
                    
            return Response(generate(), content_type='application/json')
        else:
            # Non-streaming response
            if DEBUG_MODE:
                logger.debug("Processing non-streaming request")
            thread_modele = threading.Thread(target=modele_rkllm.run, args=(prompt_tokens,))
            thread_modele.start()
            
            output_text = ""
            count = 0
            start = time.time()
            
            # Wait for completion
            while thread_modele.is_alive() or len(variables.global_text) > 0:
                while len(variables.global_text) > 0:
                    count += 1
                    output_text += variables.global_text.pop(0)
                    
                thread_modele.join(timeout=0.005)
                
            total_time = time.time() - start
            variables.generation_complete = True
            if DEBUG_MODE:
                logger.debug(f"Non-streaming generation complete. Total tokens: {count}, Total time: {total_time:.2f}s")
                
            # Return complete response
            ollama_response = {
                "model": model_name,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "message": {
                    "role": "assistant",
                    "content": output_text
                },
                "done": True,
                "total_duration": total_time * 1000000000,
                "load_duration": 0,
                "prompt_eval_count": prompt_token_count,
                "prompt_eval_duration": 0,
                "eval_count": count,
                "eval_duration": total_time * 1000000000
            }
            
            return jsonify(ollama_response), 200
    except Exception as e:
        if DEBUG_MODE:
            logger.exception(f"Error processing Ollama chat request: {str(e)}")
        raise e
    finally:
        # Restore original system prompt
        variables.system = original_system
