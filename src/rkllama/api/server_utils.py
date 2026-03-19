import json
import time
import datetime
import logging
import hashlib
import base64
import os
import re  # Add import for regex used in JSON extraction
import rkllama.api.variables as variables
from transformers import AutoTokenizer
from flask import jsonify, Response, stream_with_context
from .format_utils import create_format_instruction, validate_format_response, get_tool_calls, handle_ollama_response, handle_ollama_embedding_response, get_base64_image_from_pil, get_url_image_from_pil
from .model_utils import get_property_modelfile
import rkllama.config

# Check for debug mode using the improved method from config
DEBUG_MODE = rkllama.config.is_debug_mode()

# Set up logging based on debug mode
logging_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(rkllama.config.get_path("logs"),'rkllama_debug.log')) if DEBUG_MODE else logging.NullHandler()
    ]
)
logger = logging.getLogger("rkllama.server_utils")


class RequestWrapper:
    """A class that mimics Flask's request object for custom request handling"""
    def __init__(self, json_data, path="/"):
        self.json = json_data
        self.path = path


class EndpointHandler:
    """Base class for endpoint handlers with common functionality"""
    
    
    @staticmethod
    def prepare_prompt(model_name, messages, system="", tools=None, enable_thinking=False, images= None):
        """Prepare prompt with proper system handling"""

        # Get the tokenizer configured for the model (locally or remote)
        tokenizer = EndpointHandler.get_tokenizer(model_name)
        supports_system_role = "raise_exception('System role not supported')" not in tokenizer.chat_template
        
        if system and supports_system_role:
            prompt_messages = [{"role": "system", "content": system}] + messages
        else:
            prompt_messages = messages
        
        # Tokenize the <image> required token by rkllm
        if images:
            prompt_messages = messages + [{"role": "user", "content": "<image>"}]

        # Apply the template to the message without tokenize for debuging
        final_prompt = tokenizer.apply_chat_template(prompt_messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        logger.debug(f"Chat Template generated:\n{final_prompt}")

        # Tokenize the prompt
        tokenized = tokenizer(
            final_prompt,
            add_special_tokens=False,
        )["input_ids"]

        # Get the prompt file for the chat session
        prompt_cache_file = EndpointHandler.build_prompt_chat_session_file_id(prompt_messages)
        logger.debug(f"Prompt Cache File expected for the model {model_name} in this session:\n{prompt_cache_file}")

        # Return the tokens
        return tokenized, prompt_cache_file, final_prompt
    

    @staticmethod
    def get_tokenizer(model_name):
        """Get the tokenizer for the model. First try to get from local filesystem and then from HF"""

        # Construct the path for the local tokenizer
        local_tokenizer_path = os.path.join(rkllama.config.get_path("models"),model_name, "tokenizer")
        
        if not os.path.isdir(local_tokenizer_path):
            logger.debug("Local Tokenizer doesn't exists!")

            # Get model specific tokenizer from Huggin Face specified in Modelfile
            model_in_hf = get_property_modelfile(model_name, "HUGGINGFACE_PATH", rkllama.config.get_path("models")).replace('"', '').replace("'", "")
            logger.info(f"Download the tokenizer only one time from Hugging face repo: {model_in_hf}")
            
            # Get the tokenizer configured for the model
            tokenizer = AutoTokenizer.from_pretrained(model_in_hf, trust_remote_code=True)

            # Save to the disk the local tokenizer for future use
            tokenizer.save_pretrained(local_tokenizer_path)

        else:     
            logger.debug("Local Tokenizer found! Using it...")
            # Get the local tokenizer for the model
            tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, trust_remote_code=True)    

        # Return the tokenizer
        return tokenizer


    @staticmethod
    def calculate_durations(start_time, prompt_eval_time, current_time=None):
        """Calculate duration metrics for responses"""
        if not current_time:
            current_time = time.time()
            
        total_duration = current_time - start_time
        
        if prompt_eval_time is None:
            prompt_eval_time = start_time + (total_duration * 0.1)
            
        prompt_eval_duration = prompt_eval_time - start_time
        eval_duration = current_time - prompt_eval_time
        
        return {
            "total": int(total_duration * 1_000_000_000),
            "prompt_eval": int(prompt_eval_duration * 1_000_000_000),
            "eval": int(eval_duration * 1_000_000_000),
            "load": int(0.1 * 1_000_000_000)
        }


    @staticmethod
    def build_prompt_chat_session_file_id(messages, hash_len=50):
        """
        Generate a deterministic file ID for a prompt chat session.

        The ID is based on the first two messages of the chat (or the first one
        if only one exists) and includes a 3-digit padded message count.

        Example output:
            ksdfJld38dh4k887fJgfKnjsd38j4ss99djP8sd9LmQa21_005
        """

        if not messages:
            raise ValueError("messages list cannot be empty")

        # Build deterministic base string using up to the first two messages
        parts = []
        for msg in messages[:2]:
            role = msg.get("role", "")
            if role.lower() in ("system", "user"):
                content = msg.get("content", "")
                parts.append(f"{role}:{content}")

        base_string = "|".join(parts)

        # Compute SHA256 hash
        digest = hashlib.sha256(base_string.encode("utf-8")).digest()

        # Convert to URL-safe base64 (safe for filenames)
        short_hash = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

        # Truncate to desired length
        short_hash = short_hash[:hash_len]

        # Message count padded to 3 digits
        message_count = f"{len(messages):03d}"

        # Return the generated prompt cache file name
        return f"{short_hash}_{message_count}"
    

class ChatEndpointHandler(EndpointHandler):
    """Handler for /api/chat endpoint requests"""
    
    @staticmethod
    def format_streaming_chunk(model_name, token, is_final=False, metrics=None, format_data=None, tool_calls=None):
        """Format a streaming chunk for chat endpoint"""
        chunk = {
            "model": model_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "message": {
                "role": "assistant",
                "content": token if not is_final else ""
            },
            "done": is_final
        }

        if tool_calls:
            chunk["message"]["content"] = ""
            if not is_final:
               chunk["message"]["tool_calls"] = token
            
        
        if is_final:
            chunk["done_reason"] = "stop" if not tool_calls else "tool_calls"
            if metrics:
                chunk.update({
                    "total_duration": metrics["total"],
                    "load_duration": metrics["load"],
                    "prompt_eval_count": metrics.get("prompt_tokens", 0),
                    "prompt_eval_duration": metrics["prompt_eval"],
                    "eval_count": metrics.get("token_count", 0),
                    "eval_duration": metrics["eval"]
                })
                
        return chunk
    
    @staticmethod
    def format_complete_response(model_name, complete_text, metrics, format_data=None):
        """Format a complete non-streaming response for chat endpoint"""
        response = {
            "model": model_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "message": {
                "role": "assistant",
                "content": complete_text if not (format_data and "cleaned_json" in format_data) 
                          else format_data["cleaned_json"]
            },
            "done_reason": "stop" if not (format_data and "tool_call" in format_data) else "tool_calls",
            "done": True,
            "total_duration": metrics["total"],
            "load_duration": metrics["load"],
            "prompt_eval_count": metrics.get("prompt_tokens", 0),
            "prompt_eval_duration": metrics["prompt_eval"],
            "eval_count": metrics.get("token_count", 0),
            "eval_duration": metrics["eval"]
        }

        if format_data and "tool_call" in format_data:
            response["message"]["tool_calls"] = format_data["tool_call"]
        
        return response
        
    @classmethod
    def handle_request(cls, model_name, messages, system="", stream=True, format_spec=None, options=None, tools=None, enable_thinking=False, is_openai_request=False, images=None):
        """Process a chat request with proper format handling"""
        
        original_system = variables.system
        if system:
            variables.system = system
            
        try:
            variables.global_status = -1
            
            if format_spec:
                format_instruction = create_format_instruction(format_spec)
                if format_instruction:
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]["role"] == "user":
                            messages[i]["content"] += format_instruction
                            break
            
            
            # If Multimodal request, do not use tokenizer
            final_prompt = None
            prompt_cache_file = None
            if not images:    
                # Create the final prompt for token input requests
                final_prompt, prompt_cache_file, _ = cls.prepare_prompt(model_name, messages, system, tools, enable_thinking, images)
            
            else:
                if DEBUG_MODE:
                    logger.debug(f"Multimodal request detected. Skipping tokenization.")
                
                # Create the final prompt for text only requests
                _ , prompt_cache_file , final_prompt = cls.prepare_prompt(model_name, messages, system, tools, enable_thinking, images)
            
            # Ollama request handling 
            if stream:
                ollama_chunk = cls.handle_streaming(model_name, final_prompt, 
                                          format_spec, tools, enable_thinking, images, prompt_cache_file)
                if is_openai_request:

                    # Use unified handler
                    result = handle_ollama_response(ollama_chunk, stream=stream, is_chat=True)

                    # Convert Ollama streaming response to OpenAI format
                    ollama_chunk = Response(stream_with_context(result), mimetype="text/event-stream")
                
                # Return Ollama streaming response
                return ollama_chunk
            else:
                ollama_response, code =  cls.handle_complete(model_name, final_prompt, 
                                         format_spec, tools, enable_thinking,images, prompt_cache_file)
                
                if is_openai_request:
                    # Convert Ollama response to OpenAI format
                    ollama_response = handle_ollama_response(ollama_response, stream=stream, is_chat=True)

                # Return Ollama response
                return ollama_response, code

        finally:
            variables.system = original_system
            
    @classmethod
    def handle_streaming(cls, model_name, final_prompt, format_spec, tools, enable_thinking, images=None, prompt_cache_file = None):
        """Handle streaming chat response"""

        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model
            variables.worker_manager_rkllm.inference(model_name, final_prompt, prompt_cache_file)
        else:
            # Send the task of multimodal inference to the model
            variables.worker_manager_rkllm.multimodal(model_name, final_prompt, images, prompt_cache_file)
        
        # Wait for result pipe
        manager_pipe = variables.worker_manager_rkllm.get_result(model_name)

        def generate():
            
            count = 0
            start_time = time.time()
            prompt_eval_time = None
            complete_text = ""
            final_sent = False
            prompt_token_count = 0 
            token_count = 0
            prompt_eval = None
            eval = None
            
            thread_finished = False

            # Tool calls detection
            max_token_to_wait_for_tool_call = 100 if tools else 1 # Max tokens to wait for tool call definition
            tool_calls = False
            
            # Thinking variables
            thinking = enable_thinking
            response_tokens = [] # All tokens from response
            thinking_response_tokens = [] # Thinking tokens from response
            final_response_tokens = [] # Final answer tokens from response
            

            while not thread_finished or not final_sent:
                if manager_pipe.poll(int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))):  # Timeout in seconds
                    token = manager_pipe.recv()
                else:
                    # Abort the current inference
                    variables.worker_manager_rkllm.workers[model_name].abort_flag.value = True
                    
                    # Raise Exception
                    logger.error(f"No response received by the Worker of the model {model_name} in {int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))} seconds.")
                    
                    # Send message to the user
                    token=f"Aborted inference by Timeout ({int(rkllama.config.get("model","max_seconds_waiting_worker_response"))} seconds). Try again."

                    # Set finished state of the thread inference
                    thread_finished = True

                # Checking if finished inference
                if isinstance(token, tuple):      
                    thread_finished = True
                    # Get the stats from the inference
                    _, prompt_token_count, token_count, prompt_eval, eval = token
            
                if not thread_finished:
                    count += 1
                    
                    if count == 1:
                        prompt_eval_time = time.time()
                        
                        if thinking and "<think>" not in token.lower():
                            thinking_response_tokens.append(token)
                            token = "<think>" + token # Ensure correct initial format token <think>
                    else:
                        if thinking:
                            if "</think>" in token.lower():
                                thinking = False
                            else:
                                thinking_response_tokens.append(token)        
                    
                    complete_text += token
                    response_tokens.append(token)

                    if not thinking and token != "</think>": 
                        final_response_tokens.append(token)
                    
                    if not tool_calls:
                        if len(final_response_tokens) > max_token_to_wait_for_tool_call or not tools:
                            if variables.global_status != 1:
                                chunk = cls.format_streaming_chunk(model_name=model_name, token=token)
                                yield f"{json.dumps(chunk)}\n"
                            else:
                                pass
                        elif len(final_response_tokens) == max_token_to_wait_for_tool_call:
                            if variables.global_status != 1:
                                
                                for temp_token in response_tokens:
                                    time.sleep(0.1) # Simulate delay to stream previos tokens
                                    chunk = cls.format_streaming_chunk(model_name=model_name, token=temp_token)
                                    yield f"{json.dumps(chunk)}\n"
                            else:
                                pass 
                        elif len(final_response_tokens)  < max_token_to_wait_for_tool_call:
                            if variables.global_status != 1:
                                # Check if tool call founded in th first tokens in the response
                                tool_calls = "<tool_call>" in token
                                
                            else:
                                pass 
            
                if thread_finished and not final_sent:
                    final_sent = True

                    # Final check for tool calls in the complete response
                    if tools:
                        json_tool_calls = get_tool_calls("".join(final_response_tokens))
                        
                        # Last check for non standard <tool_call> token and tools calls only when finished before the wait token time
                        if len(final_response_tokens) < max_token_to_wait_for_tool_call:
                            if not tool_calls and json_tool_calls:
                                tool_calls = True

                    # If tool calls detected, send them as final response
                    if tools and tool_calls:
                        chunk_tool_call = cls.format_streaming_chunk(model_name=model_name, token=json_tool_calls, tool_calls=tool_calls)
                        yield f"{json.dumps(chunk_tool_call)}\n"
                    elif len(final_response_tokens)  < max_token_to_wait_for_tool_call: 
                        for temp_token in response_tokens:
                              time.sleep(0.1) # Simulate delay to stream previos tokens
                              chunk = cls.format_streaming_chunk(model_name=model_name, token=temp_token,tool_calls=tool_calls)
                              yield f"{json.dumps(chunk)}\n"

                    metrics = cls.calculate_durations(start_time, prompt_eval_time)
                    metrics["prompt_tokens"] = prompt_token_count
                    metrics["token_count"] = token_count
                    metrics["prompt_eval"] = prompt_eval 
                    metrics["eval"] = eval
                    
                    format_data = None
                    if format_spec and complete_text:
                        success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
                        if success and parsed_data:
                            format_type = (
                                format_spec.get("type", "") if isinstance(format_spec, dict) 
                                else "json"
                            )
                            format_data = {
                                "format_type": format_type,
                                "parsed": parsed_data,
                                "cleaned_json": cleaned_json
                            }
                    final_chunk = cls.format_streaming_chunk(model_name=model_name, token="", is_final=True, metrics=metrics, format_data=format_data,tool_calls=tool_calls)
                    yield f"{json.dumps(final_chunk)}\n"
                    
        return Response(generate(), content_type='application/x-ndjson')
    

    @classmethod
    def handle_complete(cls, model_name, final_prompt, format_spec, tools, enable_thinking, images=None, prompt_cache_file = None):
        """Handle complete non-streaming chat response"""
        
        start_time = time.time()
        prompt_eval_time = None
        thread_finished = False
        prompt_token_count = 0 
        token_count = 0
        prompt_eval = None 
        eval = None
        
        count = 0
        complete_text = ""
    
        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model
            variables.worker_manager_rkllm.inference(model_name, final_prompt, prompt_cache_file)
            
        else:
            # Send the task of multimodal inference to the model
            variables.worker_manager_rkllm.multimodal(model_name, final_prompt, images, prompt_cache_file)

        # Wait for result pipe
        manager_pipe = variables.worker_manager_rkllm.get_result(model_name)

        while not thread_finished:
            if manager_pipe.poll(int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))):  # Timeout in seconds
                token = manager_pipe.recv()
            else:
                # Abort the current inference
                variables.worker_manager_rkllm.workers[model_name].abort_flag.value = True

                # Raise Exception
                logger.error(f"No response received by the Worker of the model {model_name} in {int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))} seconds.")
                
                # Send message to the user
                token=f"Aborted inference by Timeout ({int(rkllama.config.get("model","max_seconds_waiting_worker_response"))} seconds). Try again."

                # Set finished state of the thread inference
                thread_finished = True
 
            # Checking if finished inference
            if isinstance(token, tuple):    
                thread_finished = True
                # Get the stats from the inference
                _, prompt_token_count, token_count, prompt_eval, eval = token

                # Exit the loop    
                continue
            
            count += 1
            if count == 1:
                prompt_eval_time = time.time()

                if enable_thinking and "<think>" not in token.lower():
                    token = "<think>" + token # Ensure correct initial format
            
            complete_text += token
        
        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        metrics["prompt_tokens"] = prompt_token_count
        metrics["token_count"] = token_count
        metrics["prompt_eval"] = prompt_eval 
        metrics["eval"] = eval
        
        format_data = None
        tool_calls = get_tool_calls(complete_text) if tools else None
        if format_spec and complete_text and not tool_calls:
            success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
            if success and parsed_data:
                format_type = (
                    format_spec.get("type", "") if isinstance(format_spec, dict) 
                    else "json"
                )
                format_data = {
                    "format_type": format_type,
                    "parsed": parsed_data,
                    "cleaned_json": cleaned_json
                }
        
        if tool_calls:
           format_data = {
                   "format_type" : "json",
                   "parsed": "",
                   "cleaned_json": "",
                   "tool_call": tool_calls
           }

        response = cls.format_complete_response(model_name, complete_text, metrics, format_data)
        return jsonify(response), 200


class GenerateEndpointHandler(EndpointHandler):
    """Handler for /api/generate endpoint requests"""
    
    @staticmethod
    def format_streaming_chunk(model_name, token, is_final=False, metrics=None, format_data=None, tool_calls=None):
        """Format a streaming chunk for generate endpoint"""
        chunk = {
            "model": model_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "response": token if not is_final else "",
            "done": is_final
        }
        
        if tool_calls:
            chunk["message"]["content"] = ""
            if not is_final:
               chunk["message"]["tool_calls"] = token

        if is_final:
            chunk["done_reason"] = "stop" if not tool_calls else "tool_calls"
            if metrics:
                chunk.update({
                    "total_duration": metrics["total"],
                    "load_duration": metrics["load"],
                    "prompt_eval_count": metrics.get("prompt_tokens", 0),
                    "prompt_eval_duration": metrics["prompt_eval"],
                    "eval_count": metrics.get("token_count", 0),
                    "eval_duration": metrics["eval"]
                })

        return chunk
    
    @staticmethod
    def format_complete_response(model_name, complete_text, metrics, format_data=None):
        """Format a complete non-streaming response for generate endpoint"""
        response = {
            "model": model_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "response": complete_text if not (format_data and "cleaned_json" in format_data) 
                       else format_data["cleaned_json"],
            "done_reason": "stop",
            "done": True,
            "total_duration": metrics["total"],
            "load_duration": metrics["load"],
            "prompt_eval_count": metrics.get("prompt_tokens", 0),
            "prompt_eval_duration": metrics["prompt_eval"],
            "eval_count": metrics.get("token_count", 0),
            "eval_duration": metrics["eval"],
            "context": []
        }
        
        return response
    
    @classmethod
    def handle_request(cls, model_name, prompt, system="", stream=True, format_spec=None, options=None,enable_thinking=False, is_openai_request=False, images=None):
        """Process a generate request with proper format handling"""
        messages = [{"role": "user", "content": prompt}]
        
        original_system = variables.system
        if system:
            variables.system = system
        
        if DEBUG_MODE:
            logger.debug(f"GenerateEndpointHandler: processing request for {model_name}")
            logger.debug(f"Format spec: {format_spec}")
        
        try:
            variables.global_status = -1
            
            if format_spec:
                format_instruction = create_format_instruction(format_spec)
                if format_instruction and messages:
                    if DEBUG_MODE:
                        logger.debug(f"Adding format instruction to prompt: {format_instruction}")
                    messages[0]["content"] += format_instruction

            
            # If Multimodal request, do not use tokenizer
            final_prompt = None
            if not images:    
                # Create the final prompts for token input requests
                final_prompt, _, _ = cls.prepare_prompt(model_name=model_name, messages=messages, system=system,enable_thinking=enable_thinking, images=images)
            else:
                if DEBUG_MODE:
                    logger.debug(f"Multimodal request detected. Skipping tokenization.")

                # Create the final prompt for text only requests
                _ , _, final_prompt = cls.prepare_prompt(model_name=model_name, messages=messages, system=system,enable_thinking=enable_thinking, images=images)
                
            # Ollama request handling 
            if stream:
                ollama_chunk = cls.handle_streaming(model_name, final_prompt, 
                                           format_spec, enable_thinking, images, None) # No cache for generation (it is not a chat conversation)
                if is_openai_request:

                    # Use unified handler
                    result = handle_ollama_response(ollama_chunk, stream=stream, is_chat=False)

                    # Convert Ollama streaming response to OpenAI format
                    ollama_chunk = Response(stream_with_context(result), mimetype="text/event-stream")
                
                # Return Ollama streaming response
                return ollama_chunk
            else:
                ollama_response, code =  cls.handle_complete(model_name, final_prompt, 
                                          format_spec, enable_thinking, images, None) # No cache for generation (it is not a chat conversation)
                
                if is_openai_request:
                    # Convert Ollama response to OpenAI format
                    ollama_response = handle_ollama_response(ollama_response, stream=stream, is_chat=False)

                # Return Ollama response
                return ollama_response, code
            
        finally:
            variables.system = original_system
    
    @classmethod
    def handle_streaming(cls, model_name, final_prompt, format_spec, enable_thinking, images=None, prompt_cache_file = None):
        """Handle streaming generate response"""

        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model
            variables.worker_manager_rkllm.inference(model_name, final_prompt, prompt_cache_file)
        else:
            # Send the task of multimodal inference to the model
            variables.worker_manager_rkllm.multimodal(model_name, final_prompt, images, prompt_cache_file)
        
        # Wait for result pipe
        manager_pipe = variables.worker_manager_rkllm.get_result(model_name)


        def generate():
            
            count = 0
            start_time = time.time()
            prompt_eval_time = None
            complete_text = ""
            final_sent = False
            prompt_token_count = 0
            token_count = 0
            prompt_eval = None 
            eval = None

            thread_finished = False
 
            while not thread_finished or not final_sent:
                if manager_pipe.poll(int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))):  # Timeout in seconds
                    token = manager_pipe.recv()
                else:
                    # Abort the current inference
                    variables.worker_manager_rkllm.workers[model_name].abort_flag.value = True
                    
                    # Raise Exception
                    logger.error(f"No response received by the Worker of the model {model_name} in {int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))} seconds.")
                    
                    # Send message to the user
                    token=f"Aborted inference by Timeout ({int(rkllama.config.get("model","max_seconds_waiting_worker_response"))} seconds). Try again." 

                    # Set finished state of the thread inference
                    thread_finished = True
               
                # Checking if finished inference
                if isinstance(token, tuple):      
                    thread_finished = True
                    # Get the stats from the inference
                    _, prompt_token_count, token_count, prompt_eval, eval = token
                
                if not thread_finished:
                    count += 1


                    if count == 1:
                        prompt_eval_time = time.time()
                        if enable_thinking and "<think>" not in token.lower():
                            token = "<think>" + token # Ensure correct initial format token <think>
                    
                    complete_text += token
                    
                    if variables.global_status != 1:
                        chunk = cls.format_streaming_chunk(model_name, token)
                        yield f"{json.dumps(chunk)}\n"
                    else:
                        pass
                
                if thread_finished and not final_sent:
                    final_sent = True
                    
                    metrics = cls.calculate_durations(start_time, prompt_eval_time)
                    metrics["prompt_tokens"] = prompt_token_count
                    metrics["token_count"] = token_count
                    metrics["prompt_eval"] = prompt_eval 
                    metrics["eval"] = eval
                    
                    format_data = None
                    if format_spec and complete_text:
                        success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
                        if success and parsed_data:
                            format_type = (
                                format_spec.get("type", "") if isinstance(format_spec, dict) 
                                else "json"
                            )
                            format_data = {
                                "format_type": format_type,
                                "parsed": parsed_data,
                                "cleaned_json": cleaned_json
                            }
                    
                    final_chunk = cls.format_streaming_chunk(model_name, "", True, metrics, format_data)
                    yield f"{json.dumps(final_chunk)}\n"
                
                    
        return Response(generate(), content_type='application/x-ndjson')
    
    @classmethod
    def handle_complete(cls, model_name, final_prompt, format_spec, enable_thinking, images=None, prompt_cache_file = None):
        """Handle complete generate response"""

        start_time = time.time()
        prompt_eval_time = None
        thread_finished = False
        prompt_token_count = 0
        token_count = 0
        prompt_eval = None 
        eval = None
        
        count = 0
        complete_text = ""
        
        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model
            variables.worker_manager_rkllm.inference(model_name, final_prompt, prompt_cache_file)
        else:
            # Send the task of multimodal inference to the model
            variables.worker_manager_rkllm.multimodal(model_name, final_prompt, images, prompt_cache_file)
        
        # Wait for result pipe
        manager_pipe = variables.worker_manager_rkllm.get_result(model_name)

        while not thread_finished:
            if manager_pipe.poll(int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))):  # Timeout in seconds
                token = manager_pipe.recv()
            else:
                # Abort the current inference
                variables.worker_manager_rkllm.workers[model_name].abort_flag.value = True
                
                # Raise Exception
                logger.error(f"No response received by the Worker of the model {model_name} in {int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))} seconds.")
                
                # Send message to the user
                token=f"Aborted inference by Timeout ({int(rkllama.config.get("model","max_seconds_waiting_worker_response"))} seconds). Try again." 

                # Set finished state of the thread inference
                thread_finished = True

            # Checking if finished inference
            if isinstance(token, tuple):  
                thread_finished = True
                # Get the stats from the inference
                _, prompt_token_count, token_count, prompt_eval, eval = token
                
                # Exit the loop
                continue
            
            count += 1
            if count == 1:
                prompt_eval_time = time.time()

                if enable_thinking and "<think>" not in token.lower():
                    token = "<think>" + token # Ensure correct initial format
            
            complete_text += token
        
        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        metrics["prompt_tokens"] = prompt_token_count
        metrics["token_count"] = token_count
        metrics["prompt_eval"] = prompt_eval 
        metrics["eval"] = eval
        
        format_data = None
        if format_spec and complete_text:
            if DEBUG_MODE:
                logger.debug(f"Validating format for complete text: {complete_text[:300]}...")
                if isinstance(format_spec, str):
                    logger.debug(f"Format is string type: {format_spec}")
            
            success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
            
            if not success and isinstance(format_spec, str) and format_spec.lower() == 'json':
                if DEBUG_MODE:
                    logger.debug("Simple JSON format validation failed, attempting additional extraction")
                
                json_pattern = r'\{[\s\S]*?\}'
                matches = re.findall(json_pattern, complete_text)
                
                for match in matches:
                    try:
                        fixed = match.replace("'", '"')
                        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
                        test_parsed = json.loads(fixed)
                        success, parsed_data, error, cleaned_json = True, test_parsed, None, fixed
                        if DEBUG_MODE:
                            logger.debug(f"Extracted valid JSON using additional methods: {cleaned_json}")
                        break
                    except:
                        continue
            
            elif not success and isinstance(format_spec, dict) and format_spec.get('type') == 'object':
                if DEBUG_MODE:
                    logger.debug(f"Initial validation failed: {error}. Trying to fix JSON...")
                
                json_pattern = r'\{[\s\S]*?\}'
                matches = re.findall(json_pattern, complete_text)
                
                for match in matches:
                    fixed = match.replace("'", '"')
                    fixed = re.sub(r'(\w+):', r'"\1":', fixed)
                    
                    try:
                        test_parsed = json.loads(fixed)
                        required_fields = format_spec.get('required', [])
                        has_required = all(field in test_parsed for field in required_fields)
                        
                        if has_required:
                            success, parsed_data, error, cleaned_json = validate_format_response(fixed, format_spec)
                            if success:
                                if DEBUG_MODE:
                                    logger.debug(f"Fixed JSON validation succeeded: {cleaned_json}")
                                break
                    except:
                        continue
            
            if DEBUG_MODE:
                logger.debug(f"Format validation result: success={success}, error={error}")
                if cleaned_json and success:
                    logger.debug(f"Cleaned JSON: {cleaned_json}")
                elif not success:
                    logger.debug(f"JSON validation failed, response will not include parsed data")
            
            if success and parsed_data:
                if isinstance(format_spec, str):
                    format_type = format_spec
                else:
                    format_type = format_spec.get("type", "json") if isinstance(format_spec, dict) else "json"
                
                format_data = {
                    "format_type": format_type,
                    "parsed": parsed_data,
                    "cleaned_json": cleaned_json
                }

        response = cls.format_complete_response(model_name, complete_text, metrics, format_data)
        
        if DEBUG_MODE and format_data:
            logger.debug(f"Created formatted response with JSON content")
            
        return jsonify(response), 200



class EmbedEndpointHandler(EndpointHandler):
    """Handler for /api/embed endpoint requests"""
    
    @staticmethod
    def format_complete_response(model_name, complete_embedding, metrics, format_data=None):
        """Format a complete non-streaming response for generate endpoint"""
        response = {
            "model": model_name,
            "embeddings": complete_embedding,
            "total_duration": metrics["total"],
            "load_duration": metrics["load"],
            "prompt_eval_count": metrics.get("prompt_tokens", 0)
        }
        
        return response
    
    @classmethod
    def handle_request(cls, model_name, input_text, truncate=True, keep_alive=None, options=None, is_openai_request=False):
        """Process a generate request with proper format handling"""

        if DEBUG_MODE:
            logger.debug(f"EmbedEndpointHandler: processing request for {model_name}")
        
        variables.global_status = -1

        logger.debug(f"Skipping tokenization for embedding")

        # Ollama request handling 
        ollama_response, code =  cls.handle_complete(model_name, input_text)
        
        if is_openai_request:
            # Convert Ollama response to OpenAI format
            ollama_response = handle_ollama_embedding_response(ollama_response)

        # Return Ollama response
        return ollama_response, code
    
    
    @classmethod
    def handle_complete(cls, model_name, input_text):
        """Handle complete embedding response"""

        start_time = time.time()
        prompt_eval_time = None
        
        # Define a list of inputs to manage request for input text and list of inputs texts to embedd
        all_inputs = []
        all_embeddings = []

        # Check the type of input
        if isinstance(input_text, list):
            all_inputs.extend(input_text)
        else:
            all_inputs.append(input_text)   

        # Initialize metricts
        total_tokens=0

        # Loop over each input
        for input in all_inputs:
            
            # Send the task of embedding to the model
            variables.worker_manager_rkllm.embedding(model_name, input)

            # Get the result from the input
            manager_pipe = variables.worker_manager_rkllm.get_result(model_name)

            # Wait for the last_embedding hidden layer return
            if manager_pipe.poll(int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))):  # Timeout in seconds
                last_embeddings = manager_pipe.recv()
            else:
                # Abort the current inference
                variables.worker_manager_rkllm.workers[model_name].abort_flag.value = True
                # Raise Exception
                logger.error(f"No response received by the Worker of the model {model_name} in {int(rkllama.config.get("model", "max_seconds_waiting_worker_response"))} seconds.")
                # Send empty embedding
                last_embeddings = embeddings = {
                        'embedding': [],
                        'embd_size': 0,
                        'num_tokens': 0
                    }

            # Add the embedding to the list of result
            all_embeddings.append(last_embeddings["embedding"].tolist())
        
            # Increase the number of tokens
            total_tokens= total_tokens + last_embeddings["num_tokens"]
            
        # Calculate metrics
        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        metrics["prompt_tokens"] = total_tokens
            
        # Format response
        response = cls.format_complete_response(model_name, all_embeddings, metrics, None)
        
        # Return response
        return jsonify(response), 200
    

class GenerateImageEndpointHandler(EndpointHandler):
    """Handler for v1/images/generations endpoint requests"""
    
    @staticmethod
    def format_complete_response(image_list, model_name, model_dir, output_format, response_format, metrics):
        """Format a complete non-streaming response for generate endpoint"""

        # Construct the default base64 response format
        data = [{"b64_json": get_base64_image_from_pil(img, output_format)} for img in image_list]

        if response_format == "url":
            # Construct the output dir for images
            output_dir = f"{model_dir}/images"

            # Construct the url response format
            data = [{"url": get_url_image_from_pil(img, model_name, output_dir, output_format)} for img in image_list]

        response = {
            "created": int(time.time()),
            "data": data,
            "usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "input_tokens_details": {
                    "text_tokens": 0,
                    "image_tokens": 0
                }
            }
        }
        
        return response
    
    @classmethod
    def handle_request(cls,  model_name, prompt, stream, size, response_format, output_format, num_images, seed, num_inference_steps, guidance_scale):
        """Process a generate request with proper format handling"""

        if DEBUG_MODE:
            logger.debug(f"GenerateImageEndpointHandler: processing request for {model_name}")
        
        # Check if streaming or not
        if not stream:
            # Ollama request handling 
            ollama_response, code =  cls.handle_complete(model_name, prompt, size, response_format, output_format, num_images, seed, num_inference_steps, guidance_scale)
        
            # Return Ollama response
            return ollama_response, code
        else:
            # Streaming not supported for image generation
            return Response("Streaming not supported yet for image generation", status=400)
        
    
    @classmethod
    def handle_complete(cls, model_name, prompt, size, response_format, output_format, num_images, seed, num_inference_steps, guidance_scale):
        """Handle complete generate image response"""


        start_time = time.time()
        prompt_eval_time = None
        
        # Use config for models path
        model_dir = os.path.join(rkllama.config.get_path("models"), model_name)

        # Send the task of generate image to the model
        image_list = variables.worker_manager_rkllm.generate_image(model_name, model_dir, prompt, size, num_images, seed, num_inference_steps, guidance_scale)
        
        # Calculate metrics
        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        
        # Format response
        response = cls.format_complete_response(image_list, model_name, model_dir, output_format, response_format, metrics)

        # Return response
        return jsonify(response), 200
    


class GenerateSpeechEndpointHandler(EndpointHandler):
    """Handler for v1/audio/speech endpoint requests"""
    
    @staticmethod
    def format_complete_response(audio, model_name, model_dir, output_format, response_format, metrics):
        """Format a complete non-streaming response for generate endpoint"""

        # Construct the default base64 response format
        data = [{"b64_json": get_base64_image_from_pil(img, output_format)} for img in image_list]

        if response_format == "url":
            # Construct the output dir for images
            output_dir = f"{model_dir}/images"

            # Construct the url response format
            data = [{"url": get_url_image_from_pil(img, model_name, output_dir, output_format)} for img in image_list]

        response = {
            "created": int(time.time()),
            "data": data,
            "usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "input_tokens_details": {
                    "text_tokens": 0,
                    "image_tokens": 0
                }
            }
        }
        
        return response
    
    @classmethod
    def handle_request(cls, model_name,input,voice,response_format,stream_format,speed):
        """Process a generate request with proper format handling"""
        
        def stream_bytes(data: bytes, chunk_size: int = 1024): # 1024 CHunk sizes
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]

        if DEBUG_MODE:
            logger.debug(f"GenerateSpeechEndpointHandler: processing request for {model_name}")
        
        # Check if streaming or not
        if stream_format == "sse":

            # Streaming not supported yet for audio generation
            return Response("Streaming not supported yet for audio generation", status=400)
        

        else:
            # Audio output 
            audio_bytes, media_type =  cls.handle_complete(model_name,input,voice,response_format,stream_format,speed)
        
            # COnstruct the response
            response = Response(
                response=stream_bytes(audio_bytes),
                mimetype=media_type
            )

            # Set the headers
            response.headers["Content-Length"] = str(len(audio_bytes))
            response.headers["Accept-Ranges"] = "bytes"
            
            # Return response
            return response
    
    @classmethod
    def handle_complete(cls, model_name,input,voice,response_format,stream_format,speed):
        """Handle complete generate speech response"""

        # Use config for models path
        model_dir = os.path.join(rkllama.config.get_path("models"), model_name)

        # Send the task of generate speech to the model
        audio = variables.worker_manager_rkllm.generate_speech(model_name, model_dir, input,voice,response_format,stream_format,speed)
        
        # Return the audio
        return audio
    


class GenerateTranscriptionsEndpointHandler(EndpointHandler):
    """Handler for v1/audio/transcriptions endpoint requests"""
    
    @staticmethod
    def format_complete_response(text, response_format):
        """Format a complete non-streaming response for generate endpoint"""

        response ={
            "text": text,
            "usage": {
                "type": "tokens",
                "input_tokens": 0,
                "input_token_details": {
                "text_tokens": 0,
                "audio_tokens": 0
                },
                "output_tokens": 0,
                "total_tokens": 0
            }
        }
        
        return response
    
    @classmethod
    def handle_request(cls, model_name,file, language, response_format, stream):
        """Process a generate request with proper format handling"""
        
        if DEBUG_MODE:
            logger.debug(f"GenerateTranscriptionsEndpointHandler: processing request for {model_name}")
        
        # Check if streaming or not
        if stream:

            # Streaming not supported yet for audio generation
            return Response("Streaming not supported yet for audio transcription", status=400)
        

        else:
            # Transcription output 
            transcription_text =  cls.handle_complete(model_name,file, language, response_format)
        
            # Return response
            return cls.format_complete_response(transcription_text, response_format)
    
    @classmethod
    def handle_complete(cls, model_name,file, language, response_format):
        """Handle complete generate transcription response"""

        # Use config for models path
        model_dir = os.path.join(rkllama.config.get_path("models"), model_name)

        # Send the task of generate transcription to the model
        transcription_text = variables.worker_manager_rkllm.generate_transcription(model_name, model_dir, file, language, response_format)

        # Return the transcription text
        return transcription_text
    

class GenerateTranslationsEndpointHandler(EndpointHandler):
    """Handler for v1/audio/translations endpoint requests"""
    
    @staticmethod
    def format_complete_response(text, response_format):
        """Format a complete non-streaming response for generate endpoint"""

        response ={
            "text": text,
        }
        
        return response
    
    @classmethod
    def handle_request(cls, model_name,file, language, response_format):
        """Process a generate request with proper format handling"""
        
        if DEBUG_MODE:
            logger.debug(f"GenerateTranslationsEndpointHandler: processing request for {model_name}")
        
        # Translation output 
        translation_text =  cls.handle_complete(model_name,file, language, response_format)
    
        # Return response
        return cls.format_complete_response(translation_text, response_format)
    
    @classmethod
    def handle_complete(cls, model_name,file, language, response_format):
        """Handle complete generate translation response"""

        # Use config for models path
        model_dir = os.path.join(rkllama.config.get_path("models"), model_name)

        # Send the task of generate translation to the model
        translation_text = variables.worker_manager_rkllm.generate_translation(model_name, model_dir, file, language, response_format)
        
        # Return the translation text
        return translation_text
