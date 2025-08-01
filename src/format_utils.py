import json
import logging
from typing import Any, Dict, Optional, Tuple, Union, List
import re
import uuid
import time
from flask import jsonify

try:
    from pydantic import BaseModel, ValidationError, create_model
    from pydantic.fields import FieldInfo
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Use a simple fallback if Pydantic is not available
    class BaseModel:
        pass
    
    class ValidationError(Exception):
        pass
        
    def create_model(*args, **kwargs):
        return None

logger = logging.getLogger("rkllama.format_utils")

def get_pydantic_type(json_type_name: str):
    """Convert JSON schema type to Python/Pydantic type"""
    if not PYDANTIC_AVAILABLE:
        return Any
        
    if json_type_name == "string":
        return str
    elif json_type_name == "integer":
        return int
    elif json_type_name == "number":
        return float
    elif json_type_name == "boolean":
        return bool
    elif json_type_name == "array":
        return List[Any]
    elif json_type_name == "object":
        return Dict[str, Any]
    return Any

def create_pydantic_model(format_spec: Dict) -> Optional[type]:
    """Create a Pydantic model from a JSON schema"""
    if not PYDANTIC_AVAILABLE:
        logger.warning("Pydantic not available, format validation disabled")
        return None
        
    if not format_spec or not isinstance(format_spec, dict):
        return None
        
    try:
        # Get schema properties and required fields
        properties = format_spec.get("properties", {})
        required = format_spec.get("required", [])
        
        # Create field definitions for the Pydantic model
        fields = {}
        for prop_name, prop_spec in properties.items():
            prop_type = prop_spec.get("type", "string")
            python_type = get_pydantic_type(prop_type)
            
            # Make field optional if not required
            if prop_name not in required:
                fields[prop_name] = (Optional[python_type], None)
            else:
                fields[prop_name] = (python_type, ...)
        
        # Create dynamic model based on the schema
        model_name = format_spec.get("title", "DynamicResponseModel")
        model = create_model(model_name, **fields)
        return model
    except Exception as e:
        logger.error(f"Error creating Pydantic model from schema: {str(e)}")
        return None

def create_format_instruction(format_spec):
    """Create a format instruction based on the format specification"""
    if not format_spec:
        return ""
    
    instruction = "\n\n"
    
    # Handle different format types
    if isinstance(format_spec, dict):
        format_type = format_spec.get('type', '')
        
        if format_type == 'json':
            instruction += "You must respond with a valid JSON. Return only the JSON with no explanation text before or after it."
        
        elif format_type == 'object':
            # For object type, create a template based on properties
            properties = format_spec.get('properties', {})
            example = {}
            
            # Create example values for each property
            for prop, details in properties.items():
                prop_type = details.get('type', 'string')
                if prop_type == 'string':
                    example[prop] = ""
                elif prop_type == 'integer':
                    example[prop] = 0
                elif prop_type == 'number':
                    example[prop] = 0.0
                elif prop_type == 'boolean':
                    example[prop] = False
                elif prop_type == 'array':
                    example[prop] = []
                elif prop_type == 'object':
                    example[prop] = {}
            
            required = format_spec.get('required', [])
            if required:
                required_str = ", ".join(required)
                instruction += f"You must respond with a valid JSON object with exactly these required fields: {required_str}.\n\n"
            
            # Add example JSON structure
            instruction += "Format your entire response as a JSON object with ONLY these fields:\n"
            instruction += "```json\n"
            instruction += json.dumps(example, indent=2)
            instruction += "\n```\n\n"
            instruction += "Return ONLY the JSON object, with no explanations, comments or text before or after the JSON.\n"
            instruction += "Never use '_' prefix in your field names."
    
    # Handle simple string format specification like format="json"
    elif isinstance(format_spec, str):
        if format_spec.lower() == 'json':
            instruction += "You must respond with valid JSON. Return ONLY the JSON with no explanation or text before or after it.\n"
            instruction += "Format your entire response as a JSON object containing all the relevant information from your answer.\n"
            instruction += "Ensure the JSON is properly formatted and valid."
    
    return instruction

def get_example_value(type_name: str) -> str:
    """Return an example value for a given JSON schema type"""
    if type_name == "string":
        return '""'
    elif type_name == "integer":
        return "0"
    elif type_name == "number":
        return "0.0"
    elif type_name == "boolean":
        return "false"
    elif type_name == "array":
        return "[]"
    elif type_name == "object":
        return "{}"
    elif type_name == "null":
        return "null"
    return '""'  # default to string

def extract_json(text):
    """Extract JSON from text that might contain non-JSON content"""
    
    # First look for JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    code_matches = re.findall(code_block_pattern, text)
    
    for potential_json in code_matches:
        try:
            parsed = json.loads(potential_json)
            return potential_json.strip(), parsed
        except json.JSONDecodeError:
            continue
    
    # If no valid JSON in code blocks, try to find JSON-like content directly
    json_pattern = r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
    json_matches = re.findall(json_pattern, text)
    
    for potential_json in json_matches:
        try:
            parsed = json.loads(potential_json)
            return potential_json.strip(), parsed
        except json.JSONDecodeError:
            continue
    
    # Try with more lenient pattern
    more_lenient_pattern = r'\{[\s\S]*?\}'
    lenient_matches = re.findall(more_lenient_pattern, text)
    
    for potential_json in lenient_matches:
        # Clean up the text
        cleaned = re.sub(r'[^\{\}\[\],:."\'0-9a-zA-Z_\s-]', '', potential_json)
        cleaned = cleaned.replace("'", '"')  # Replace single quotes with double quotes
        
        try:
            parsed = json.loads(cleaned)
            return cleaned.strip(), parsed
        except json.JSONDecodeError:
            continue
    
    # No valid JSON found
    return None, None

def validate_format_response(text, format_spec):
    """
    Validate that the model's response matches the requested format
    
    Args:
        text: The model's response text
        format_spec: The format specification (dict or string)
    
    Returns:
        tuple: (success, parsed_data, error_message, cleaned_json)
    """
    if not format_spec:
        return False, None, "No format specification provided", None
    
    # Extract JSON from the response text
    json_text, parsed_data = extract_json(text)
    
    if not json_text or not parsed_data:
        return False, None, "Could not extract valid JSON from response", None
    
    # For simple 'json' format, we just need valid JSON
    if format_spec == 'json' or (isinstance(format_spec, str) and format_spec.lower() == 'json') or \
       (isinstance(format_spec, dict) and format_spec.get('type') == 'json'):
        return True, parsed_data, None, json_text
    
    # For 'object' format with schema validation
    if isinstance(format_spec, dict) and format_spec.get('type') == 'object':
        properties = format_spec.get('properties', {})
        required = format_spec.get('required', [])
        
        # Verify all required fields are present
        missing_fields = []
        for field in required:
            if field not in parsed_data:
                missing_fields.append(field)
        
        if missing_fields:
            return False, None, f"Missing required field{'s' if len(missing_fields) > 1 else ''}: {', '.join(missing_fields)}", None
        
        # Check field types
        for field, value in parsed_data.items():
            if field in properties:
                expected_type = properties[field].get('type')
                
                # Validate type
                if expected_type == 'string' and not isinstance(value, str):
                    return False, None, f"Field '{field}' should be a string", None
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    return False, None, f"Field '{field}' should be a number", None
                elif expected_type == 'integer':
                    # Convert floats to ints if they are whole numbers
                    if isinstance(value, float) and value.is_integer():
                        parsed_data[field] = int(value)
                    elif not isinstance(value, int):
                        return False, None, f"Field '{field}' should be an integer", None
                elif expected_type == 'boolean' and not isinstance(value, bool):
                    return False, None, f"Field '{field}' should be a boolean", None
                elif expected_type == 'array' and not isinstance(value, list):
                    return False, None, f"Field '{field}' should be an array", None
                elif expected_type == 'object' and not isinstance(value, dict):
                    return False, None, f"Field '{field}' should be an object", None
        
        # Create a clean JSON with only the expected fields
        if properties:
            clean_data = {}
            for field in properties.keys():
                if field in parsed_data:
                    clean_data[field] = parsed_data[field]
            
            # Include any required fields not in properties
            for field in required:
                if field not in clean_data and field in parsed_data:
                    clean_data[field] = parsed_data[field]
            
            cleaned_json = json.dumps(clean_data, indent=2)
            return True, clean_data, None, cleaned_json
    
    return True, parsed_data, None, json_text


def openai_to_ollama_request(openai_payload: dict) -> dict:
    """
    Translate an OpenAI /v1/chat/completions request payload to Ollama /api/chat format.

    Args:
        openai_payload (dict): OpenAI request payload.

    Returns:
        dict: Ollama-compatible request payload.
    """
    messages = openai_payload.get("messages", [])
    model = openai_payload.get("model", "llama3")
    stream = openai_payload.get("stream", False)

    # Base Ollama payload
    ollama_payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # Supported Ollama options from OpenAI fields
    supported_option_mappings = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "stop": "stop",
        "max_tokens": "max_new_tokens",
        "max_completion_tokens": "max_new_tokens",
        "seed": "seed",
        "logit_bias": "logit_bias",  # If supported
    }

    for openai_key, ollama_key in supported_option_mappings.items():
        if openai_key in openai_payload:
            ollama_payload.setdefault("options", {})[ollama_key] = openai_payload[openai_key]

    # Handle tool_choice, tools, functions
    # Ollama currently has no native tool/function support (like OpenAI tool-calling)
    # But we include them for forward compatibility if needed by custom handler
    for passthrough_key in ["tools", "tool_choice", "functions", "function_call", "response_format", "n"]:
        if passthrough_key in openai_payload:
            ollama_payload[passthrough_key] = openai_payload[passthrough_key]

    return ollama_payload


def ollama_to_openai_response(ollama_response: dict) -> dict:
    """
    Convert Ollama's chat response to a fully OpenAI-compatible /v1/chat/completions response.
    
    Args:
        ollama_response (dict): Response from Ollama's /api/chat endpoint.
    
    Returns:
        dict: OpenAI-compatible response.
    """
    
    # Generate metadata
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = ollama_response.get("model", "unknown-model")

    # Extract message
    message = ollama_response.get("message", {})
    role = message.get("role", "assistant")
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", None)

    # Handle finish_reason
    finish_reason = "stop" if ollama_response.get("done", True) else None

    # Build the choice message
    choice_message = {
        "role": role,
        "content": content
    }

    if tool_calls:
        # OpenAI v1 supports `tool_calls`
        for tool in tool_calls:
            tool["id"] = f"call_{uuid.uuid4().hex}"
            tool["type"] = "function"
        choice_message["tool_calls"] = tool_calls
        finish_reason = ollama_response.get("done_reason")


    choice = {
        "index": 0,
        "message": choice_message,
        "finish_reason": finish_reason
    }

    # Handle token usage if present
    usage = {}
    if "prompt_eval_count" in ollama_response:
        usage["prompt_tokens"] = ollama_response["prompt_eval_count"]
    if "eval_count" in ollama_response:
        usage["completion_tokens"] = ollama_response["eval_count"]
    if usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    # Build full response
    openai_response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [choice],
    }

    if usage:
        openai_response["usage"] = usage

    return openai_response

def ollama_stream_to_openai_chunks(ollama_stream_lines):
    """
    Converts an iterable of Ollama stream JSON lines to OpenAI SSE streaming chunks.

    Args:
        ollama_stream_lines (iterable[str]): Streamed JSON lines from Ollama.

    Yields:
        str: OpenAI-compatible `data: ...\n\n` formatted SSE chunks.
    """

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    for line in ollama_stream_lines:
        line = str(line).strip()
        if not line or line.startswith("data:"):
            continue

        try:
            #print(line)
            ollama_chunk = json.loads(line)
            #print
        except json.JSONDecodeError:
            continue

        content_piece = ollama_chunk.get("message", {}).get("content", "")
        role = ollama_chunk.get("message", {}).get("role")
        tool_calls = ollama_chunk.get("message", {}).get("tool_calls")
        model = ollama_chunk.get("model", "unknown-model")
        finish_reason = ollama_chunk.get("done_reason", None)

        delta = {}
        if content_piece:
            delta["content"] = content_piece
        if role:
            delta["role"] = role
        if tool_calls:
            for tool in tool_calls:
                tool["id"] = f"call_{uuid.uuid4().hex}"
                tool["type"] = "function"
            delta["tool_calls"] = tool_calls
        
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }

        yield f"{json.dumps(chunk)}\n\n"

        if ollama_chunk.get("done") is True:
            # Final chunk — stop streaming
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason #"stop"
                }]
            }
            yield f"{json.dumps(final_chunk)}\n\n"
            yield "[DONE]\n\n"
            break

def handle_ollama_response(response, stream=False):
    """
    Handles an Ollama response and converts it into either:
    - a single OpenAI-compatible JSON object (non-streaming), or
    - an iterable of SSE chunks (streaming).

    Args:
        response: `requests.Response` object from Ollama.
        stream (bool): Whether streaming was requested.

    Returns:
        dict | generator[str]: OpenAI-compatible response (full or streaming).
    """
    if stream:
        # Generator that yields OpenAI-style SSE chunks
        def stream_chunks():
            # Read the Flask Response iterable (bytes or str)
            for raw in response.response:
                # Ensure bytes are decoded
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                raw = raw.strip()
                if raw:
                    yield from ollama_stream_to_openai_chunks([raw])
        return stream_chunks()
    else:
        # Full JSON response
        #ollama_response = response.json()
        ollama_response = json.loads(response.get_data().decode("utf-8"))
        return jsonify(ollama_to_openai_response(ollama_response))


def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))
    
################################## Tool Calls #####################################
def RawJSONDecoder(index):
    class _RawJSONDecoder(json.JSONDecoder):
        end = None
 
        def decode(self, s, *_):
            data, self.__class__.end = self.raw_decode(s, index)
            return data
    return _RawJSONDecoder
 
def extract_json_tools_from_text(s, index=0):
    while (index := s.find('{', index)) != -1:
        try:
            yield json.loads(s, cls=(decoder := RawJSONDecoder(index)))
            index = decoder.end
        except json.JSONDecodeError:
            index += 1


def get_tool_calls_generic(response):
    """ Return a list of formatted function calls by the LLM in the response.
        It a generic function to search any JSON response from any LLM with the required format:
        {"name": <function_name>, "parameters": <dictionary_of_argument_name_value>} 
        or
        {"name": <function_name>, "arguments": <dictionary_of_argument_name_value>} 
        For example:

        { "name": "get_current_weather", "arguments": { "location": "Paris, France", "format": "celsius" }

        Qwen models use <tool_call></tool_call> tags in chat template but for example Llama3.2 doesn't. That's why this generic implementation.


        Final response of a request must something like this: (https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools)

        {
            "model": "llama3.2",
            "created_at": "2024-07-22T20:33:28.123648Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                {
                    "function": {
                    "name": "get_current_weather",
                    "arguments": {
                        "format": "celsius",
                        "location": "Paris, FR"
                    }
                    }
                }
                ]
            },
            "done_reason": "stop",
            "done": true,
            "total_duration": 885095291,
            "load_duration": 3753500,
            "prompt_eval_count": 122,
            "prompt_eval_duration": 328493000,
            "eval_count": 33,
            "eval_duration": 552222000
        }


}
    """

    logger.debug(f"Searching tools with generic method: get_tool_calls_generic")

    # Get all the json objects
    json_tool_list = list(extract_json_tools_from_text(response))

    # Set the required keys in json object to identify tool calls
    required_keys_for_tools_option1 = set(["name", "arguments"]) # Other like Qwen
    required_keys_for_tools_option2 = set(["name", "parameters"]) # Llama default chat template
     
    tool_calls = []
    tool_calls += [{ "function": tool } for tool in json_tool_list if required_keys_for_tools_option1.issubset(tool.keys()) or required_keys_for_tools_option2.issubset(tool.keys())]

    # Rename the key "parameters" for "arguments" for standard
    tool_calls_renamed = []
    for tool in tool_calls:
      if "parameters" in tool["function"]:
          tool["function"]["arguments"] = tool["function"].pop("parameters")
      tool_calls_renamed.append(tool)
    return tool_calls_renamed


def get_tool_calls_standard(response):
    """ Get all the tool calls indicated by the LLM in the response. 
        Only work if the chat template of the LLM uses <tool_call></tool_call> tags (Like Qwen models)
    """
    
    logger.debug(f"Searching tools with standard method: get_tool_calls_standard")

    tool_calls = []
    for tools in re.findall("<tool_call>(.*?)</tool_call>", response, re.DOTALL):
      # tool_calls += [{ "function": json.loads(tool) } for tool in tools.split('\n') if tool]
      # To make more smartt LLM in case LLM response inside <tool_call><t/ool_call> but with an extra word. 
      # For example: <tool_call> Output : {"name": <function_name>, "arguments": <dictionary_of_argument_name_value>} </tool_call>
      tool_calls += get_tool_calls_generic(tools) 
    return tool_calls


def get_tool_calls(response):
    """ Get all the tool calls indicated by the LLM in the response """
    
    # We try the standard form first
    tool_calls = get_tool_calls_standard(response)

    if not tool_calls:
        # No standard format tool call found. Search for more generic way
        tool_calls = get_tool_calls_generic(response)

    return tool_calls