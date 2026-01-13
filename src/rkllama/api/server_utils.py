import datetime
import json
import os
import re
import time

from fastapi.responses import Response, StreamingResponse
from transformers import AutoTokenizer

import rkllama.api.variables as variables
import rkllama.config
from rkllama.logging import get_logger

from .format_utils import (
    create_format_instruction,
    get_base64_image_from_pil,
    get_tool_calls,
    get_url_image_from_pil,
    handle_ollama_embedding_response,
    handle_ollama_response,
    validate_format_response,
)
from .model_utils import get_property_modelfile

# Check for debug mode using the improved method from config
DEBUG_MODE = rkllama.config.is_debug_mode()

logger = get_logger("rkllama.server_utils")


class RequestWrapper:
    """A class that mimics Flask's request object for custom request handling"""
    def __init__(self, json_data, path="/"):
        self.json = json_data
        self.path = path


class EndpointHandler:
    """Base class for endpoint handlers with common functionality"""


    @staticmethod
    def _tokenizer_supports_tools(tokenizer) -> bool:
        """Check if the tokenizer's chat template supports tools."""
        if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
            return False
        # Check for common tool-related Jinja2 patterns
        tool_patterns = ['tools', 'tool_call', 'function', '{% for tool']
        return any(pattern in tokenizer.chat_template for pattern in tool_patterns)

    @staticmethod
    def _load_tokenizer(model_name: str):
        """Load tokenizer from local path or HuggingFace.

        Tries local tokenizer first (from TOKENIZER property in Modelfile),
        then falls back to HuggingFace (from HUGGINGFACE_PATH).

        Returns:
            AutoTokenizer: Loaded tokenizer
        """
        models_path = rkllama.config.get_path("models")
        model_dir = os.path.join(models_path, model_name)

        tokenizer = None

        # Try local tokenizer first
        tokenizer_path = get_property_modelfile(model_name, "TOKENIZER", models_path)
        if tokenizer_path:
            tokenizer_path = tokenizer_path.replace('"', '').replace("'", "")
            # Handle relative paths (e.g., "./tokenizer")
            if tokenizer_path.startswith("./"):
                tokenizer_path = os.path.join(model_dir, tokenizer_path[2:])
            elif not os.path.isabs(tokenizer_path):
                tokenizer_path = os.path.join(model_dir, tokenizer_path)

            if os.path.isdir(tokenizer_path):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                    logger.debug("Loaded tokenizer from local path", path=tokenizer_path)
                except Exception as e:
                    logger.warning("Failed to load local tokenizer, falling back to HuggingFace", error=str(e))

        # Fallback to HuggingFace
        if tokenizer is None:
            model_in_hf = get_property_modelfile(model_name, "HUGGINGFACE_PATH", models_path)
            if model_in_hf:
                model_in_hf = model_in_hf.replace('"', '').replace("'", "")
                tokenizer = AutoTokenizer.from_pretrained(model_in_hf, trust_remote_code=True)
                logger.debug("Loaded tokenizer from HuggingFace", model=model_in_hf)
            else:
                raise ValueError(f"No tokenizer path or HUGGINGFACE_PATH found for model {model_name}")

        return tokenizer

    @staticmethod
    def prepare_prompt(model_name, messages, system="", tools=None, tool_choice=None, enable_thinking=False, tokenize=True):
        """Prepare prompt with proper system handling

        Args:
            model_name: Name of the model
            messages: List of chat messages
            system: System prompt
            tools: Optional tools for function calling
            tool_choice: Optional tool selection mode (e.g., "required", "auto")
            enable_thinking: Enable thinking/reasoning mode
            tokenize: If True, return token IDs; if False, return formatted string

        Returns:
            tuple: (tokenizer, prompt_tokens_or_string, token_count)
        """
        # Load tokenizer (local first, then HuggingFace)
        tokenizer = EndpointHandler._load_tokenizer(model_name)

        # Warn if tools provided but tokenizer doesn't support them
        if tools and not EndpointHandler._tokenizer_supports_tools(tokenizer):
            logger.warning(
                "Tools provided but tokenizer chat template does not appear to support tools",
                model=model_name,
                tools_count=len(tools)
            )

        supports_system_role = "raise_exception('System role not supported')" not in tokenizer.chat_template

        if system and supports_system_role:
            prompt_messages = [{"role": "system", "content": system}] + messages
        else:
            prompt_messages = messages

        # Build template parameters
        template_params = {
            "tokenize": tokenize,
            "add_generation_prompt": True,
        }

        # Add tools if provided
        if tools:
            template_params["tools"] = tools

        # Add tool_choice if provided and template supports it
        if tool_choice and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template and "tool_choice" in tokenizer.chat_template:
            template_params["tool_choice"] = tool_choice

        # Add enable_thinking if supported
        if enable_thinking:
            template_params["enable_thinking"] = enable_thinking

        prompt_result = tokenizer.apply_chat_template(prompt_messages, **template_params)

        # Calculate token count - if we got a string, tokenize it to count
        if tokenize:
            token_count = len(prompt_result)
        else:
            # Get token count by tokenizing the string
            token_count = len(tokenizer.encode(prompt_result))

        return tokenizer, prompt_result, token_count


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
    def handle_request(cls, model_name, messages, system="", stream=True, format_spec=None, options=None, tools=None, tool_choice=None, enable_thinking=False, is_openai_request=False, images=None):
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
            prompt_input = None
            prompt_token_count = None
            if not images:
                # Create the prompt string for text only requests (using PROMPT mode)
                tokenizer, prompt_input, prompt_token_count = cls.prepare_prompt(model_name, messages, system, tools, tool_choice, enable_thinking, tokenize=False)

            else:
                if DEBUG_MODE:
                    logger.debug("Multimodal request detected, skipping tokenization")

                for message in messages:
                    if "images" in message:
                        message.pop("images")  # Remove images from messages to avoid context length reach with base64 images
                prompt_input = f"<image>{str(messages)}"
                prompt_token_count = 0

            # Ollama request handling
            if stream:
                ollama_chunk = cls.handle_streaming(model_name, prompt_input,
                                          prompt_token_count, format_spec, tools, enable_thinking, images)
                if is_openai_request:

                    # Use unified handler
                    result = handle_ollama_response(ollama_chunk, stream=stream, is_chat=True)

                    # Convert Ollama streaming response to OpenAI format
                    ollama_chunk = StreamingResponse(result, media_type="text/event-stream")

                # Return Ollama streaming response
                return ollama_chunk
            else:
                ollama_response = cls.handle_complete(model_name, prompt_input,
                                         prompt_token_count, format_spec, tools, enable_thinking,images)

                if is_openai_request:
                    # Convert Ollama response to OpenAI format
                    ollama_response = handle_ollama_response(ollama_response, stream=stream, is_chat=True)

                # Return Ollama response
                return ollama_response

        finally:
            variables.system = original_system

    @classmethod
    def handle_streaming(cls, model_name, prompt_input, prompt_token_count, format_spec, tools, enable_thinking, images=None):
        """Handle streaming chat response"""

        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model (using PROMPT mode - raw string)
            variables.worker_manager_rkllm.inference(model_name, prompt_input, role="user", enable_thinking=enable_thinking, use_prompt_mode=True)
        else:
            # Send the task of multimodal inference to the model
            variables.worker_manager_rkllm.multimodal(model_name, prompt_input, images, role="user", enable_thinking=enable_thinking)
            # Clear the cache to prevent image embedding problems
            variables.worker_manager_rkllm.clear_cache_worker(model_name)


        # Wait for result queue
        result_q = variables.worker_manager_rkllm.get_result(model_name)
        if result_q is None:
            raise RuntimeError(f"Model '{model_name}' is not loaded. Please load the model first.")
        finished_inference_token = variables.worker_manager_rkllm.get_finished_inference_token()


        def generate():

            count = 0
            start_time = time.time()
            prompt_eval_time = None
            complete_text = ""
            final_sent = False

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
                token = result_q.get(timeout=300)  # Block until receive any token
                if token == finished_inference_token:
                    thread_finished = True

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
                    metrics["token_count"] = count

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

        return StreamingResponse(generate(), media_type='application/x-ndjson')


    @classmethod
    def handle_complete(cls, model_name, prompt_input, prompt_token_count, format_spec, tools, enable_thinking, images=None):
        """Handle complete non-streaming chat response"""

        start_time = time.time()
        prompt_eval_time = None
        thread_finished = False

        count = 0
        complete_text = ""

        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model (using PROMPT mode - raw string)
            variables.worker_manager_rkllm.inference(model_name, prompt_input, role="user", enable_thinking=enable_thinking, use_prompt_mode=True)
        else:
            # Send the task of multimodal inference to the model
            variables.worker_manager_rkllm.multimodal(model_name, prompt_input, images, role="user", enable_thinking=enable_thinking)
            # Clear the cache to prevent image embedding problems
            variables.worker_manager_rkllm.clear_cache_worker(model_name)

        # Wait for result queue
        result_q = variables.worker_manager_rkllm.get_result(model_name)
        if result_q is None:
            raise RuntimeError(f"Model '{model_name}' is not loaded. Please load the model first.")
        finished_inference_token = variables.worker_manager_rkllm.get_finished_inference_token()


        while not thread_finished:
            token = result_q.get(timeout=300)  # Block until receive any token
            if token == finished_inference_token:
                thread_finished = True
                continue

            count += 1
            if count == 1:
                prompt_eval_time = time.time()

                if enable_thinking and "<think>" not in token.lower():
                    token = "<think>" + token # Ensure correct initial format

            complete_text += token

        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        metrics["prompt_tokens"] = prompt_token_count
        metrics["token_count"] = count

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
        return response


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
            logger.debug("GenerateEndpointHandler processing request", model=model_name)
            logger.debug("Format spec", format_spec=format_spec)

        try:
            variables.global_status = -1

            if format_spec:
                format_instruction = create_format_instruction(format_spec)
                if format_instruction and messages:
                    if DEBUG_MODE:
                        logger.debug("Adding format instruction to prompt", instruction=format_instruction)
                    messages[0]["content"] += format_instruction


            # If Multimodal request, do not use tokenizer
            prompt_input = None
            prompt_token_count = None
            if not images:
                # Create the prompt string for text only requests (using PROMPT mode)
                tokenizer, prompt_input, prompt_token_count = cls.prepare_prompt(model_name=model_name, messages=messages, system=system, enable_thinking=enable_thinking, tokenize=False)
            else:
                if DEBUG_MODE:
                    logger.debug("Multimodal request detected, skipping tokenization")
                prompt_input = f"<image>{prompt}"
                prompt_token_count = 0

            # Ollama request handling
            if stream:
                ollama_chunk = cls.handle_streaming(model_name, prompt_input,
                                          prompt_token_count, format_spec, enable_thinking, images)
                if is_openai_request:

                    # Use unified handler
                    result = handle_ollama_response(ollama_chunk, stream=stream, is_chat=False)

                    # Convert Ollama streaming response to OpenAI format
                    ollama_chunk = StreamingResponse(result, media_type="text/event-stream")

                # Return Ollama streaming response
                return ollama_chunk
            else:
                ollama_response = cls.handle_complete(model_name, prompt_input,
                                         prompt_token_count, format_spec, enable_thinking, images)

                if is_openai_request:
                    # Convert Ollama response to OpenAI format
                    ollama_response = handle_ollama_response(ollama_response, stream=stream, is_chat=False)

                # Return Ollama response
                return ollama_response

        finally:
            variables.system = original_system

    @classmethod
    def handle_streaming(cls, model_name, prompt_input, prompt_token_count, format_spec, enable_thinking, images=None):
        """Handle streaming generate response"""

        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model (using PROMPT mode - raw string)
            variables.worker_manager_rkllm.inference(model_name, prompt_input, role="user", enable_thinking=enable_thinking, use_prompt_mode=True)
        else:
            # Send the task of multimodal inference to the model
            variables.worker_manager_rkllm.multimodal(model_name, prompt_input, images, role="user", enable_thinking=enable_thinking)
            # Clear the cache to prevent image embedding problems
            variables.worker_manager_rkllm.clear_cache_worker(model_name)

        # Wait for result queue
        result_q = variables.worker_manager_rkllm.get_result(model_name)
        if result_q is None:
            raise RuntimeError(f"Model '{model_name}' is not loaded. Please load the model first.")
        finished_inference_token = variables.worker_manager_rkllm.get_finished_inference_token()


        def generate():

            count = 0
            start_time = time.time()
            prompt_eval_time = None
            complete_text = ""
            final_sent = False

            thread_finished = False

            while not thread_finished or not final_sent:
                token = result_q.get(timeout=300)  # Block until receive any token
                if token == finished_inference_token:
                    thread_finished = True

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
                    metrics["token_count"] = count

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


        return StreamingResponse(generate(), media_type='application/x-ndjson')

    @classmethod
    def handle_complete(cls, model_name, prompt_input, prompt_token_count, format_spec, enable_thinking, images=None):
        """Handle complete generate response"""

        start_time = time.time()
        prompt_eval_time = None
        thread_finished = False

        count = 0
        complete_text = ""

        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model (using PROMPT mode - raw string)
            variables.worker_manager_rkllm.inference(model_name, prompt_input, role="user", enable_thinking=enable_thinking, use_prompt_mode=True)
        else:
            # Send the task of multimodal inference to the model
            variables.worker_manager_rkllm.multimodal(model_name, prompt_input, images, role="user", enable_thinking=enable_thinking)
            # Clear the cache to prevent image embedding problems
            variables.worker_manager_rkllm.clear_cache_worker(model_name)

        # Wait for result queue
        result_q = variables.worker_manager_rkllm.get_result(model_name)
        if result_q is None:
            raise RuntimeError(f"Model '{model_name}' is not loaded. Please load the model first.")
        finished_inference_token = variables.worker_manager_rkllm.get_finished_inference_token()

        while not thread_finished:
            token = result_q.get(timeout=300)  # Block until receive any token
            if token == finished_inference_token:
                thread_finished = True
                continue

            count += 1
            if count == 1:
                prompt_eval_time = time.time()

                if enable_thinking and "<think>" not in token.lower():
                    token = "<think>" + token # Ensure correct initial format

            complete_text += token

        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        metrics["prompt_tokens"] = prompt_token_count
        metrics["token_count"] = count

        format_data = None
        if format_spec and complete_text:
            if DEBUG_MODE:
                logger.debug("Validating format for complete text", preview=complete_text[:300])
                if isinstance(format_spec, str):
                    logger.debug("Format is string type", format_spec=format_spec)

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
                            logger.debug("Extracted valid JSON using additional methods", cleaned_json=cleaned_json)
                        break
                    except:
                        continue

            elif not success and isinstance(format_spec, dict) and format_spec.get('type') == 'object':
                if DEBUG_MODE:
                    logger.debug("Initial validation failed, trying to fix JSON", error=error)

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
                                    logger.debug("Fixed JSON validation succeeded", cleaned_json=cleaned_json)
                                break
                    except:
                        continue

            if DEBUG_MODE:
                logger.debug("Format validation result", success=success, error=error)
                if cleaned_json and success:
                    logger.debug("Cleaned JSON", cleaned_json=cleaned_json)
                elif not success:
                    logger.debug("JSON validation failed, response will not include parsed data")

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
            logger.debug("Created formatted response with JSON content")

        return response


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
            logger.debug("EmbedEndpointHandler processing request", model=model_name)

        variables.global_status = -1

        # Create the prompts
        _, prompt_tokens, prompt_token_count = cls.prepare_prompt(model_name=model_name, messages=input_text)

        # Ollama request handling
        ollama_response = cls.handle_complete(model_name, prompt_tokens, prompt_token_count)

        if is_openai_request:
            # Convert Ollama response to OpenAI format
            ollama_response = handle_ollama_embedding_response(ollama_response)

        # Return Ollama response
        return ollama_response


    @classmethod
    def handle_complete(cls, model_name, input_tokens, prompt_token_count):
        """Handle complete embedding response"""

        start_time = time.time()
        prompt_eval_time = None

        # Send the task of embedding to the model
        variables.worker_manager_rkllm.embedding(model_name, input_tokens)
        result_q = variables.worker_manager_rkllm.get_result(model_name)
        if result_q is None:
            raise RuntimeError(f"Model '{model_name}' is not loaded. Please load the model first.")

        # Wait for the last_embedding hidden layer return
        embeddings = result_q.get(timeout=300)

        # Calculate metrics
        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        metrics["prompt_tokens"] = prompt_token_count

        # Format response
        response = cls.format_complete_response(model_name, embeddings.tolist(), metrics, None)

        # Return response
        return response


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
            logger.debug("GenerateImageEndpointHandler processing request", model=model_name)

        # Check if streaming or not
        if not stream:
            # Ollama request handling
            ollama_response = cls.handle_complete(model_name, prompt, size, response_format, output_format, num_images, seed, num_inference_steps, guidance_scale)

            # Return Ollama response
            return ollama_response
        else:
            # Streaming not supported for image generation
            return Response(content="Streaming not supported yet for image generation", status_code=400)


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
        return response



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
    def handle_request(cls, model_name,input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio):
        """Process a generate request with proper format handling"""

        def stream_bytes(data: bytes, chunk_size: int = 1024): # 1024 CHunk sizes
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]

        if DEBUG_MODE:
            logger.debug("GenerateSpeechEndpointHandler processing request", model=model_name)

        # Check if streaming or not
        if stream_format == "sse":

            # Streaming not supported yet for audio generation
            return Response(content="Streaming not supported yet for audio generation", status_code=400)


        else:
            # Audio output
            audio_bytes, media_type =  cls.handle_complete(model_name,input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio)

            # Construct the response with headers
            headers = {
                "Content-Length": str(len(audio_bytes)),
                "Accept-Ranges": "bytes"
            }

            # Return streaming response
            return StreamingResponse(
                content=stream_bytes(audio_bytes),
                media_type=media_type,
                headers=headers
            )

    @classmethod
    def handle_complete(cls, model_name,input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio):
        """Handle complete generate speech response"""

        # Use config for models path
        model_dir = os.path.join(rkllama.config.get_path("models"), model_name)

        # Send the task of generate speech to the model
        audio = variables.worker_manager_rkllm.generate_speech(model_name, model_dir, input,voice,response_format,stream_format,volume,length_scale,noise_scale,noise_w_scale,normalize_audio)

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
            logger.debug("GenerateTranscriptionsEndpointHandler processing request", model=model_name)

        # Check if streaming or not
        if stream:

            # Streaming not supported yet for audio generation
            return Response(content="Streaming not supported yet for audio transcription", status_code=400)


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
