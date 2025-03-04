import threading, time, json
from transformers import AutoTokenizer
from flask import Flask, request, jsonify, Response
import src.variables as variables
import datetime
import logging
import os  # For accessing environment variables
from .format_utils import create_format_instruction, validate_format_response
from src.model_utils import get_simplified_model_name  # Import at the top level

logger = logging.getLogger("rkllama.process")

# Import DEBUG_MODE setting from environment variables
DEBUG_MODE = os.environ.get("RKLLAMA_DEBUG", "0").lower() in ["1", "true", "yes", "on"]

def Request(modele_rkllm, custom_request=None):
    """
    Process a request to the language model
    
    Args:
        modele_rkllm: The language model instance
        custom_request: Optional custom request object that mimics Flask request
    
    Returns:
        Flask response with generated text
    """
    try:
        # Mettre le serveur en état de blocage.
        isLocked = True

        # Use custom_request if provided, otherwise use Flask's request
        req = custom_request if custom_request is not None else request
        data = req.json
        
        if data and 'messages' in data:
            # Extract format parameters
            format_spec = data.get('format')
            format_options = data.get('options', {})
            
            # Store format settings in model instance for reference
            if modele_rkllm:
                modele_rkllm.format_schema = format_spec
                modele_rkllm.format_type = (
                    format_spec.get("type", "") if isinstance(format_spec, dict) 
                    else format_spec
                )
                modele_rkllm.format_options = format_options
            
            # Réinitialiser les variables globales.
            variables.global_status = -1

            # Définir la structure de la réponse renvoyée.
            llmResponse = {
                "id": "rkllm_chat",
                "object": "rkllm_chat",
                "created": int(time.time()),
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens_per_second": 0,
                    "total_tokens": 0
                }
            }

            # Check if this is an Ollama-style request
            is_ollama_request = req.path.startswith('/api/')
            
            # Récupérer l'historique du chat depuis la requête JSON
            messages = data["messages"]

            # Create format instructions
            if format_spec:
                format_instruction = create_format_instruction(format_spec)
                if format_instruction:
                    # Find the last user message and append format instructions
                    last_user_msg_idx = -1
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]["role"] == "user":
                            last_user_msg_idx = i
                            break
                    
                    if last_user_msg_idx >= 0:
                        original_content = messages[last_user_msg_idx]["content"]
                        messages[last_user_msg_idx]["content"] = original_content + format_instruction
                        logger.debug(f"Added format instruction: {format_instruction}")

            # Mise en place du tokenizer
            tokenizer = AutoTokenizer.from_pretrained(variables.model_id, trust_remote_code=True)
            supports_system_role = "raise_exception('System role not supported')" not in tokenizer.chat_template

            if variables.system and supports_system_role:
                prompt = [{"role": "system", "content": variables.system}] + messages
            else:
                prompt = messages

            for i in range(1, len(prompt)):
                if prompt[i]["role"] == prompt[i - 1]["role"]:
                    raise ValueError("Les rôles doivent alterner entre 'user' et 'assistant'.")

            # Mise en place du chat Template
            prompt = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            llmResponse["usage"]["prompt_tokens"] = llmResponse["usage"]["total_tokens"] = len(prompt)

            sortie_rkllm = ""

            if "stream" in data.keys() and data["stream"] == True:
                def generate():
                    thread_modele = threading.Thread(target=modele_rkllm.run, args=(prompt,))
                    thread_modele.start()

                    thread_modele_terminé = False
                    count = 0
                    start = time.time()
                    prompt_eval_end_time = None
                    final_message_sent = False  # Track if we've sent the final message
                    
                    # Initialize accumulated text for JSON format validation
                    complete_text = ""

                    while not thread_modele_terminé or not final_message_sent:
                        processed_tokens = False
                        
                        while len(variables.global_text) > 0:
                            processed_tokens = True
                            count += 1
                            current_token = variables.global_text.pop(0)
                            
                            # Mark time when first token is generated
                            if count == 1:
                                prompt_eval_end_time = time.time()
                            
                            # Accumulate text for format validation
                            complete_text += current_token

                            llmResponse["choices"] = [
                                {
                                "role": "assistant",
                                "content": current_token,
                                "logprobs": None,
                                "finish_reason": "stop" if variables.global_status == 1 else None,
                                }
                            ]
                            llmResponse["usage"]["completion_tokens"] = count
                            llmResponse["usage"]["total_tokens"] += 1
                            
                            if is_ollama_request:
                                # Get simplified model name for consistency
                                simplified_model_name = get_simplified_model_name(variables.model_id)
                                
                                if variables.global_status != 1:
                                    # Intermediate chunks - minimal fields only
                                    ollama_chunk = {
                                        "model": simplified_model_name,
                                        "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                        "message": {
                                            "role": "assistant",
                                            "content": current_token
                                        },
                                        "done": False
                                    }
                                    yield f"{json.dumps(ollama_chunk)}\n"
                                else:
                                    # This is the final token from the model, mark that we're ready to send final message
                                    final_message_sent = True
                                    
                                    # Final chunk - include all metrics
                                    current_time = time.time()
                                    total_duration = current_time - start
                                    
                                    # Calculate durations
                                    if prompt_eval_end_time is None:
                                        prompt_eval_end_time = start + (total_duration * 0.1)
                                    
                                    prompt_eval_duration = prompt_eval_end_time - start
                                    eval_duration = current_time - prompt_eval_end_time
                                    load_duration = 0.1  # Fixed 100ms in seconds
                                    
                                    # Process format validation if requested
                                    cleaned_content = None
                                    parsed_data = None
                                    if format_spec:
                                        success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
                                        if success and cleaned_json:
                                            cleaned_content = cleaned_json
                                    
                                    # Final message with metrics
                                    ollama_chunk = {
                                        "model": simplified_model_name,
                                        "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                        "message": {
                                            "role": "assistant",
                                            # Use cleaned content if available, otherwise use current token
                                            "content": cleaned_content if cleaned_content else current_token
                                        },
                                        "done": True,
                                        "done_reason": "stop",
                                        "total_duration": int(total_duration * 1_000_000_000),
                                        "load_duration": int(load_duration * 1_000_000_000),
                                        "prompt_eval_count": llmResponse["usage"]["prompt_tokens"],
                                        "prompt_eval_duration": int(prompt_eval_duration * 1_000_000_000),
                                        "eval_count": count,
                                        "eval_duration": int(eval_duration * 1_000_000_000)
                                    }
                                    
                                    # Add format information if available
                                    if format_spec and parsed_data:
                                        format_type = (
                                            format_spec.get("type", "") if isinstance(format_spec, dict)
                                            else "json"
                                        )
                                        # Remove format and parsed from response - they are not part of Ollama API
                                        # ollama_chunk["format"] = format_type
                                        # ollama_chunk["parsed"] = parsed_data
                                    
                                    yield f"{json.dumps(ollama_chunk)}\n"
                            else:
                                # For non-Ollama streaming
                                # Process format in the final chunk
                                if variables.global_status == 1 and format_spec:
                                    success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
                                    if success and parsed_data:
                                        llmResponse["choices"][0]["format"] = format_spec
                                        llmResponse["choices"][0]["parsed"] = parsed_data
                                
                                yield f"{json.dumps(llmResponse)}\n\n"

                        # Check if thread is done but we haven't sent final message yet
                        thread_modele.join(timeout=0.005)
                        thread_modele_terminé = not thread_modele.is_alive()
                        
                        # If model is done and we haven't sent the final message yet, do it now
                        if thread_modele_terminé and not final_message_sent and is_ollama_request:
                            final_message_sent = True
                            
                            # Calculate final metrics
                            current_time = time.time()
                            total_duration = current_time - start
                            
                            if prompt_eval_end_time is None:
                                prompt_eval_end_time = start + (total_duration * 0.1)
                                
                            prompt_eval_duration = prompt_eval_end_time - start
                            eval_duration = current_time - prompt_eval_end_time
                            load_duration = 0.1
                            
                            # Process format validation if requested
                            parsed_data = None
                            if format_spec and complete_text:
                                success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
                            
                            # Create final message with empty content to avoid duplication
                            ollama_final = {
                                "model": get_simplified_model_name(variables.model_id),
                                "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                "message": {
                                    "role": "assistant",
                                    "content": ""  # Empty content to avoid duplicating text
                                },
                                "done": True,
                                "done_reason": "stop",
                                "total_duration": int(total_duration * 1_000_000_000),
                                "load_duration": int(load_duration * 1_000_000_000),
                                "prompt_eval_count": llmResponse["usage"]["prompt_tokens"],
                                "prompt_eval_duration": int(prompt_eval_duration * 1_000_000_000),
                                "eval_count": count,
                                "eval_duration": int(eval_duration * 1_000_000_000)
                            }
                            
                            # Add format information if available
                            if format_spec and parsed_data:
                                format_type = (
                                    format_spec.get("type", "") if isinstance(format_spec, dict)
                                    else "json"
                                )
                                # Remove format and parsed fields
                                # ollama_final["format"] = format_type
                                # ollama_final["parsed"] = parsed_data
                            
                            yield f"{json.dumps(ollama_final)}\n"
                            
                        # If we didn't process any tokens in this loop iteration, add a small sleep to avoid CPU spin
                        if not processed_tokens:
                            time.sleep(0.01)
                    
                # Return appropriate streaming response
                return Response(generate(), content_type='application/x-ndjson' if is_ollama_request else 'text/plain')
            
            # For non-streaming responses
            else:
                # Create inference thread
                thread_modele = threading.Thread(target=modele_rkllm.run, args=(prompt,))
                try:
                    thread_modele.start()
                    print("Inference thread started")
                except Exception as e:
                    print("Error starting thread:", e)

                # Wait for model to finish
                threadFinish = False
                count = 0
                start = time.time()
                prompt_eval_end_time = None  # Will store time when first token is generated
                complete_text = ""
                first_token_generated = False

                while not threadFinish:
                    while len(variables.global_text) > 0:
                        count += 1
                        token = variables.global_text.pop(0)
                        
                        # Mark the time when first token is generated (end of prompt evaluation)
                        if not first_token_generated:
                            first_token_generated = True
                            prompt_eval_end_time = time.time()
                        
                        complete_text += token
                        time.sleep(0.005)

                        thread_modele.join(timeout=0.005)
                    threadFinish = not thread_modele.is_alive()

                end_time = time.time()
                total_duration = end_time - start
                
                # Calculate the various duration metrics
                if prompt_eval_end_time is None:
                    # If no tokens were generated, use 10% of total time as estimate
                    prompt_eval_end_time = start + (total_duration * 0.1)
                
                prompt_eval_duration = prompt_eval_end_time - start  # Time spent evaluating prompt
                eval_duration = end_time - prompt_eval_end_time  # Time spent generating tokens
                load_duration = 0.1  # Fixed 100ms in seconds
                
                # Handle format validation for completed response
                if format_spec and complete_text:
                    # Updated to unpack the additional cleaned_json return value
                    success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
                    logger.debug(f"Format validation: success={success}, error={error}")
                    
                    # Structure Ollama-compatible response
                    if is_ollama_request:
                        # Get simplified model name for consistency
                        simplified_model_name = get_simplified_model_name(variables.model_id)
                        
                        ollama_response = {
                            "model": simplified_model_name,
                            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            "message": {
                                "role": "assistant", 
                                # Use only the clean JSON text if available, otherwise use complete response
                                "content": cleaned_json if success and cleaned_json else complete_text
                            },
                            "done_reason": "stop",  # Always add done_reason for completed responses
                            "done": True,
                            # Add all required duration fields in nanoseconds
                            "total_duration": int(total_duration * 1_000_000_000),
                            "load_duration": int(load_duration * 1_000_000_000),  # Fixed 100ms
                            "prompt_eval_count": llmResponse["usage"]["prompt_tokens"],
                            "prompt_eval_duration": int(prompt_eval_duration * 1_000_000_000),
                            "eval_count": count,
                            "eval_duration": int(eval_duration * 1_000_000_000)
                        }
                        
                        # Remove format and parsed fields
                        # if success and parsed_data:
                        #     ollama_response["format"] = format_type
                        #     ollama_response["parsed"] = parsed_data
                        
                        return jsonify(ollama_response), 200
                    else:
                        # Standard API response
                        llmResponse["choices"] = [{
                            "role": "assistant",
                            # Use only the clean JSON text if available
                            "content": cleaned_json if success and cleaned_json else complete_text,
                            "logprobs": None,
                            "finish_reason": "stop"
                        }]
                        
                        # Remove format and parsed fields
                        # if success and parsed_data:
                        #     llmResponse["choices"][0]["format"] = format_spec
                        #     llmResponse["choices"][0]["parsed"] = parsed_data
                        
                        return jsonify(llmResponse), 200

                # Default response format if no format specification or validation failed
                if is_ollama_request:
                    # Get simplified model name for consistency
                    simplified_model_name = get_simplified_model_name(variables.model_id)
                    
                    ollama_response = {
                        "model": simplified_model_name,
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "message": {
                            "role": "assistant",
                            "content": complete_text
                        },
                        "done_reason": "stop",
                        "done": True,
                        # Add all required duration fields in nanoseconds
                        "total_duration": int(total_duration * 1_000_000_000),
                        "load_duration": int(load_duration * 1_000_000_000),
                        "prompt_eval_count": llmResponse["usage"]["prompt_tokens"],
                        "prompt_eval_duration": int(prompt_eval_duration * 1_000_000_000),
                        "eval_count": count,
                        "eval_duration": int(eval_duration * 1_000_000_000)
                    }
                    return jsonify(ollama_response), 200
                else:
                    llmResponse["choices"] = [{
                        "role": "assistant",
                        "content": complete_text,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }]
                    return jsonify(llmResponse), 200
                    
        else:
            return jsonify({'status': 'error', 'message': 'Données JSON invalides !'}), 400
    finally:
        # No need to release the lock here as it should be handled by the calling function
        if custom_request is None:
            variables.verrou.release()
        est_bloqué = False
