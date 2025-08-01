import threading
import json
import time
import datetime
import logging
import os
import re  # Add import for regex used in JSON extraction
import src.variables as variables
from transformers import AutoTokenizer
from flask import jsonify, Response, stream_with_context
from .format_utils import create_format_instruction, validate_format_response, get_tool_calls, handle_ollama_response

import config

# Check for debug mode using the improved method from config
DEBUG_MODE = config.is_debug_mode()

# Set up logging based on debug mode
logging_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.get_path("logs"),'rkllama_debug.log')) if DEBUG_MODE else logging.NullHandler()
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
    def prepare_prompt(messages, system="", tools=None, enable_thinking=False):
        """Prepare prompt with proper system handling"""
        tokenizer = AutoTokenizer.from_pretrained(variables.model_id, trust_remote_code=True)
        supports_system_role = "raise_exception('System role not supported')" not in tokenizer.chat_template
        
        
        #print(messages)
        #print("-" * 50)
        #prepared_messages = []
        #for message in messages:
        #    all_contents = ""
        #    for content in message.get("content", []):
        #        if isinstance(content, str):
        #            content = content.strip()
        #            if content:
        #                all_contents = all_contents + "\n" +message["content"]
        #        elif isinstance(content, dict):
        #                if "text" in content:
        #                    content["text"] = content["text"].strip()
        #                    if content["text"]:
        #                        all_contents = all_contents + "\n" + content["text"]
        #    prepared_messages.append({"role": message["role"], "content": all_contents})
        
        if system and supports_system_role:
            #prompt_messages = [{"role": "system", "content": system}] + prepared_messages #messages
            prompt_messages = [{"role": "system", "content": system}] + messages
        else:
            prompt_messages = messages
        
        prompt_tokens = tokenizer.apply_chat_template(prompt_messages, tools=tools, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking)
        
        return tokenizer, prompt_tokens, len(prompt_tokens)
    
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
    def handle_request(cls, modele_rkllm, model_name, messages, system="", stream=True, format_spec=None, options=None, tools=None, enable_thinking=False, is_openai_request=False):
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
            
            tokenizer, prompt_tokens, prompt_token_count = cls.prepare_prompt(messages, system, tools, enable_thinking)
            
            # Ollama request handling 
            if stream:
                ollama_chunk = cls.handle_streaming(modele_rkllm, model_name, prompt_tokens, 
                                          prompt_token_count, format_spec, tools, enable_thinking)
                if is_openai_request:

                    # Use unified handler
                    result = handle_ollama_response(ollama_chunk, stream=stream)

                    # Convert Ollama streaming response to OpenAI format
                    ollama_chunk = Response(stream_with_context(result), mimetype="text/event-stream")
                
                # Return Ollama streaming response
                return ollama_chunk
            else:
                ollama_response, code =  cls.handle_complete(modele_rkllm, model_name, prompt_tokens, 
                                         prompt_token_count, format_spec, tools, enable_thinking)
                
                if is_openai_request:
                    # Convert Ollama streaming response to OpenAI format
                    ollama_response = handle_ollama_response(ollama_response, stream=stream)

                # Return Ollama streaming response
                return ollama_response, code

        finally:
            variables.system = original_system
            
    @classmethod
    def handle_streaming(cls, modele_rkllm, model_name, prompt_tokens, prompt_token_count, format_spec, tools, enable_thinking):
        """Handle streaming chat response"""
        def generate():
            thread_model = threading.Thread(target=modele_rkllm.run, args=(prompt_tokens,))
            thread_model.start()
            
            count = 0
            start_time = time.time()
            prompt_eval_time = None
            complete_text = ""
            final_sent = False
            
            thread_finished = False

            # Tool calls detection
            max_token_to_wait_for_tool_call = 100 if tools else 1 # Max tokens to wait for tool call definition
            tool_calls = False
            first_tokens = []
            thinking = enable_thinking
            final_response_tokens = []
            
            while not thread_finished or not final_sent:
                tokens_processed = False
                
                while len(variables.global_text) > 0:
                    tokens_processed = True
                    count += 1
                    token = variables.global_text.pop(0)
                    
                    if count == 1:
                        prompt_eval_time = time.time()
                        
                        if thinking and "<think>" not in token.lower():
                            token = "<think>" + token # Ensure correct initial format token <think>
                    else:
                        if thinking and "</think>" in token.lower():
                            thinking = False
                    
                    complete_text += token
                    first_tokens.append(token)

                    if not thinking and token != "</think>": 
                        final_response_tokens.append(token)
                    
                    if not tool_calls:
                        if len(final_response_tokens) > max_token_to_wait_for_tool_call:
                            if variables.global_status != 1:
                                chunk = cls.format_streaming_chunk(model_name=model_name, token=token)
                                yield f"{json.dumps(chunk)}\n"
                            else:
                                pass
                        elif len(final_response_tokens) == max_token_to_wait_for_tool_call:
                            if variables.global_status != 1:
                                for temp_token in first_tokens:
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
                
                thread_model.join(timeout=0.005)
                thread_finished = not thread_model.is_alive()
                
                if thread_finished and not final_sent:
                    final_sent = True

                    if tool_calls:
                        chunk_tool_call = cls.format_streaming_chunk(model_name=model_name, token=get_tool_calls(complete_text), tool_calls=tool_calls)
                        yield f"{json.dumps(chunk_tool_call)}\n"
                    elif count < max_token_to_wait_for_tool_call: 
                        for temp_token in first_tokens:
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
                
                if not tokens_processed:
                    time.sleep(0.01)
                    
        return Response(generate(), content_type='application/x-ndjson')
    

    @classmethod
    def handle_complete(cls, modele_rkllm, model_name, prompt_tokens, prompt_token_count, format_spec, tools, enable_thinking):
        """Handle complete non-streaming chat response"""
        start_time = time.time()
        prompt_eval_time = None
        
        thread_model = threading.Thread(target=modele_rkllm.run, args=(prompt_tokens,))
        thread_model.start()
        
        count = 0
        complete_text = ""
    
        
        while thread_model.is_alive() or len(variables.global_text) > 0:
            while len(variables.global_text) > 0:
                count += 1
                token = variables.global_text.pop(0)
                time.sleep(0.005)
                
                if count == 1:
                    prompt_eval_time = time.time()

                    if enable_thinking and "<think>" not in token.lower():
                        token = "<think>" + token # Ensure correct initial format
                
                complete_text += token
            
            thread_model.join(timeout=0.005)
        
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
        return jsonify(response), 200


class GenerateEndpointHandler(EndpointHandler):
    """Handler for /api/generate endpoint requests"""
    
    @staticmethod
    def format_streaming_chunk(model_name, token, is_final=False, metrics=None, format_data=None):
        """Format a streaming chunk for generate endpoint"""
        chunk = {
            "model": model_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "response": token if not is_final else "",
            "done": is_final
        }
        
        if is_final:
            chunk["done_reason"] = "stop"
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
    def handle_request(cls, modele_rkllm, model_name, prompt, system="", stream=True, format_spec=None, options=None, enable_thinking=False):
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
            
            tokenizer, prompt_tokens, prompt_token_count = cls.prepare_prompt(messages=messages, system=system, enable_thinking=enable_thinking)
            
            if stream:
                return cls.handle_streaming(modele_rkllm, model_name, prompt_tokens, 
                                          prompt_token_count, format_spec)
            else:
                return cls.handle_complete(modele_rkllm, model_name, prompt_tokens, 
                                         prompt_token_count, format_spec)
        finally:
            variables.system = original_system
    
    @classmethod
    def handle_streaming(cls, modele_rkllm, model_name, prompt_tokens, prompt_token_count, format_spec):
        """Handle streaming generate response"""
        def generate():
            thread_model = threading.Thread(target=modele_rkllm.run, args=(prompt_tokens,))
            thread_model.start()
            
            count = 0
            start_time = time.time()
            prompt_eval_time = None
            complete_text = ""
            final_sent = False
            
            thread_finished = False
            
            while not thread_finished or not final_sent:
                tokens_processed = False
                
                while len(variables.global_text) > 0:
                    tokens_processed = True
                    count += 1
                    token = variables.global_text.pop(0)
                    
                    if count == 1:
                        prompt_eval_time = time.time()
                    
                    complete_text += token
                    
                    if variables.global_status != 1:
                        chunk = cls.format_streaming_chunk(model_name, token)
                        yield f"{json.dumps(chunk)}\n"
                    else:
                        pass
                
                thread_model.join(timeout=0.005)
                thread_finished = not thread_model.is_alive()
                
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
                
                if not tokens_processed:
                    time.sleep(0.01)
                    
        return Response(generate(), content_type='application/x-ndjson')
    
    @classmethod
    def handle_complete(cls, modele_rkllm, model_name, prompt_tokens, prompt_token_count, format_spec):
        """Handle complete generate response"""
        start_time = time.time()
        prompt_eval_time = None
        
        thread_model = threading.Thread(target=modele_rkllm.run, args=(prompt_tokens,))
        thread_model.start()
        
        count = 0
        complete_text = ""
        
        while thread_model.is_alive() or len(variables.global_text) > 0:
            while len(variables.global_text) > 0:
                count += 1
                token = variables.global_text.pop(0)
                
                if count == 1:
                    prompt_eval_time = time.time()
                
                complete_text += token
            
            thread_model.join(timeout=0.005)
        
        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        metrics["prompt_tokens"] = prompt_token_count
        metrics["token_count"] = count
        
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


def process_ollama_chat_request(modele_rkllm, model_name, messages, system="", stream=True, format_spec=None, options=None):
    """Process /api/chat request with correct format"""
    return ChatEndpointHandler.handle_request(
        modele_rkllm=modele_rkllm,
        model_name=model_name,
        messages=messages,
        system=system,
        stream=stream,
        format_spec=format_spec,
        options=options
    )

def process_ollama_generate_request(modele_rkllm, model_name, prompt, system="", stream=True, format_spec=None, options=None):
    """Process /api/generate request with correct format"""
    return GenerateEndpointHandler.handle_request(
        modele_rkllm=modele_rkllm,
        model_name=model_name,
        prompt=prompt,
        system=system,
        stream=stream,
        format_spec=format_spec,
        options=options
    )
