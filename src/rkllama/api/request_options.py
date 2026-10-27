"""Explicit request-option contract for the RKLLM backend (not llama.cpp)."""
import copy
import math
import json


SAMPLING_OPTIONS = frozenset({
    "temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty",
    "repeat_penalty", "mirostat", "mirostat_tau", "mirostat_eta", "max_new_tokens",
})
TOKEN_ALIASES = {"max_tokens": "max_new_tokens", "max_completion_tokens": "max_new_tokens"}


def strict_json(text,**kwargs):
    def invalid(value):raise ValueError('Non-JSON numeric value')
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('Duplicate JSON key')
            out[key]=value
        return out
    return json.loads(text,parse_constant=invalid,object_pairs_hook=pairs,**kwargs)



def validate_sampling_options(options, prefix="options."):
    """Reject options that would be ignored or narrowed by the native ABI."""
    if not isinstance(options, dict):
        raise RequestOptionError("options", "options must be an object")
    result = {}
    for key in sorted(options, key=lambda key: (key == "max_completion_tokens", key)):
        value = options[key]
        native = "max_new_tokens" if key == "num_predict" else TOKEN_ALIASES.get(key, key)
        param = prefix + key
        if native not in SAMPLING_OPTIONS:
            raise RequestOptionError(param, "Unsupported per-request RKLLM option")
        if native in result:
            raise RequestOptionError(param, "Specify only one value for " + native)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestOptionError(param, "Expected a JSON number, not a boolean or string")
        if native in {"top_k", "mirostat", "max_new_tokens"}:
            minimum = 1 if native == "max_new_tokens" else 0
            maximum = 2 if native == "mirostat" else 2**31 - 1
            valid = isinstance(value, int) and minimum <= value <= maximum
        else:
            valid = abs(value) <= 3.4028234663852886e38 and math.isfinite(value)
            if native in {"temperature", "top_p"}:
                valid = valid and value >= 0
            if native == "top_p":
                valid = valid and value <= 1
            if native in {"repeat_penalty", "mirostat_tau", "mirostat_eta"}:
                valid = valid and value > 0
        if not valid:
            raise RequestOptionError(param, "Invalid numeric value for " + native)
        result[native] = value
    return result


def validate_legacy_format(value):
    """Accept only the legacy best-effort format subset actually interpreted."""
    if value == "" or value == "json" or value == {"type": "json"}:
        return
    if not isinstance(value, dict) or value.get("type") != "object":
        raise RequestOptionError("format", "Supported format: json or a flat best-effort object schema")
    if set(value) - {"type", "properties", "required"}:
        raise RequestOptionError("format", "Unsupported schema keyword; strict decoding is unavailable")
    properties = value.get("properties", {})
    required = value.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise RequestOptionError("format", "properties must be an object and required must be an array")
    for details in properties.values():
        if (not isinstance(details, dict) or set(details) != {"type"}
                or details["type"] not in ("string", "integer", "number", "boolean")):
            raise RequestOptionError("format", "Only flat scalar property types are supported; enums and constraints are unsupported")
    if any(not isinstance(key, str) or key not in properties for key in required):
        raise RequestOptionError("format", "required must name declared properties")


def validate_tool_schema(schema, param):
    """Reject schema shapes the tool parser cannot interpret before inference."""
    if not isinstance(schema, dict):
        raise RequestOptionError(param, "Tool schema must be an object")
    allowed = {"type", "properties", "required", "additionalProperties", "items", "description", "title", "enum"}
    kind = schema.get("type", "object" if "properties" in schema else None)
    if set(schema) - allowed or (kind is not None and (not isinstance(kind, str) or kind not in {"string", "integer", "number", "boolean", "object", "array", "null"})):
        raise RequestOptionError(param, "Unsupported tool schema keywords or type; union types are unsupported")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list) or any(not isinstance(x, str) for x in required):
        raise RequestOptionError(param, "properties must be an object and required a string array")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise RequestOptionError(param, "additionalProperties must be boolean")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise RequestOptionError(param, "enum must be a nonempty array")
    for key, value in properties.items():
        validate_tool_schema(value, param + ".properties." + key)
    if "items" in schema:
        validate_tool_schema(schema["items"], param + ".items")


def tool_value_matches(value, schema):
    kind = schema.get("type", "object" if "properties" in schema else None)
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list, "null": type(None)}
    if kind and (not isinstance(value, types[kind]) or (kind in ("number", "integer") and isinstance(value, bool))):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and set(value) - set(props):
            return False
        return all(tool_value_matches(v, props[k]) for k, v in value.items() if k in props)
    if isinstance(value, list) and "items" in schema:
        return all(tool_value_matches(v, schema["items"]) for v in value)
    return True


def validate_request_options(payload, *, openai=False, chat=True):
    """Return a private normalized payload; never silently drop client settings."""
    if not isinstance(payload, dict):
        raise RequestOptionError(None, "Request body must be a JSON object")
    common = {"model", "system", "stream", "options", "format", "response_format",
              "enable_thinking", "think", "images"}
    allowed = common | ({"messages", "tools"} if chat else {"prompt"})
    if openai:
        allowed |= SAMPLING_OPTIONS | TOKEN_ALIASES.keys() | {"n", "stream_options"}
        if chat:
            allowed.add("tool_choice")
    for key in payload:
        if key not in allowed:
            raise RequestOptionError(key, "Unsupported RKLLM request field")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        raise RequestOptionError("stream", "stream must be a boolean")
    if "n" in payload and (type(payload["n"]) is not int or payload["n"] != 1):
        raise RequestOptionError("n", "Only one completion (n=1) is supported")
    result = copy.deepcopy(payload)
    if "stream_options" in payload:
        stream_options = payload["stream_options"]
        if payload.get("stream") is not True:
            raise RequestOptionError("stream_options", "stream_options requires stream=true")
        if not isinstance(stream_options, dict):
            raise RequestOptionError("stream_options", "stream_options must be an object")
        for key, value in stream_options.items():
            if key != "include_usage" or not isinstance(value, bool):
                raise RequestOptionError("stream_options." + key, "Only boolean include_usage is supported")
    if "tool_choice" in payload:
        choice = payload["tool_choice"]
        if choice not in ("auto", "none"):
            raise RequestOptionError("tool_choice", "Forced tool selection is unsupported; use auto or none")
        if choice == "none":
            result.pop("tools", None)
        result.pop("tool_choice", None)
    if "tools" in result:
        tools = result["tools"]
        if not isinstance(tools, list):
            raise RequestOptionError("tools", "tools must be an array")
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                raise RequestOptionError("tools", "Only function tool definitions are supported")
            function = tool["function"]
            if not isinstance(function.get("name"), str) or not function["name"]:
                raise RequestOptionError("tools", "Each function requires a nonempty name")
            if "strict" in function and function["strict"] is not False:
                raise RequestOptionError(f"tools.{index}.function.strict", "strict tool schemas are unsupported")
            validate_tool_schema(function.get("parameters", {"type": "object"}), f"tools.{index}.function.parameters")
    options = validate_sampling_options(payload.get("options", {}))
    if openai:
        top = {key: payload[key] for key in payload if key in SAMPLING_OPTIONS or key in TOKEN_ALIASES}
        # Validate before mapping so aliases cannot silently overwrite each other.
        mapped = validate_sampling_options(top, prefix="")
        for key in options:
            if key in mapped:
                raise RequestOptionError("options." + key, "Option is also specified at the top level")
        options.update(mapped)
    result["options"] = options
    result.update(normalize_thinking(payload))
    result.pop("think", None)
    if "format" in payload:
        validate_legacy_format(payload["format"])
    result.update(normalize_response_format(payload))
    result.pop("response_format", None)
    return result


class RequestOptionError(ValueError):
    """An unsupported, conflicting, or invalid client option."""

    def __init__(self, param, message, code="invalid_request_option"):
        self.param = param
        self.code = code
        super().__init__(message)


def validate_context_budget(prompt_tokens, output_tokens, context_tokens, param):
    """Reserve the requested output budget before dispatching token input."""
    if prompt_tokens + output_tokens > context_tokens:
        raise RequestOptionError(
            param,
            f"Prompt has {prompt_tokens} tokens and reserves {output_tokens} output tokens; "
            f"the configured context limit is {context_tokens} tokens. Shorten the prompt or reduce the output limit.",
            code="context_length_exceeded",
        )


def normalize_thinking(payload):
    """Keep explicit false distinct from an omitted model-default setting."""
    keys = [key for key in ("enable_thinking", "think") if key in payload]
    for key in keys:
        if not isinstance(payload[key], bool):
            raise RequestOptionError(key, "Thinking control must be a boolean")
    if len(keys) == 2 and payload[keys[0]] != payload[keys[1]]:
        raise RequestOptionError("think", "Conflicting think and enable_thinking values")
    return {"enable_thinking": payload[keys[0]]} if keys else {}


def normalize_response_format(payload):
    """Translate OpenAI's format name without promising constrained decoding."""
    if "response_format" not in payload:
        return {}
    if "format" in payload:
        raise RequestOptionError("response_format", "Use response_format or format, not both")
    value = payload["response_format"]
    if value == {"type": "text"}:
        return {}
    if value == {"type": "json_object"}:
        return {"format": "json"}
    raise RequestOptionError(
        "response_format",
        "RKLLM supports only response_format type text or json_object (best effort); "
        "json_schema, strict schemas, and constrained decoding are unsupported",
    )
