# Tool Calling Guide - RKLLama

RKLLama provides comprehensive tool/function calling capabilities with full Ollama API compatibility. This guide covers everything you need to know about using tools with RKLLama.

## Overview

Tool calling allows your language models to interact with external functions and APIs in a structured way. RKLLama's implementation supports:

- **Multiple LLM formats** (Qwen, Llama 3.2+, others)
- **Ollama API compatibility** 
- **Streaming and non-streaming modes**
- **Robust JSON extraction and validation**
- **Automatic format normalization**

## Quick Start

### 1. Basic Tool Call Request

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

### 2. Tool Call Response

When a model decides to call a tool, the response includes `tool_calls`:

```json
{
  "model": "qwen2.5:3b",
  "created_at": "2024-12-21T10:30:00.000Z",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {
            "location": "Paris, FR",
            "format": "celsius"
          }
        }
      }
    ]
  },
  "done_reason": "tool_calls",
  "done": true,
  "total_duration": 885095291,
  "load_duration": 3753500,
  "prompt_eval_count": 122,
  "prompt_eval_duration": 328493000,
  "eval_count": 33,
  "eval_duration": 552222000
}
```

## Tool Definition Format

### Function Schema

Each tool must follow this structure:

```json
{
  "type": "function",
  "function": {
    "name": "function_name",
    "description": "Clear description of what the function does",
    "parameters": {
      "type": "object",
      "properties": {
        "param1": {
          "type": "string|number|boolean|array|object",
          "description": "Parameter description",
          "enum": ["option1", "option2"]  // Optional for string types
        },
        "param2": {
          "type": "number",
          "description": "Another parameter"
        }
      },
      "required": ["param1"]  // List of required parameters
    }
  }
}
```

### Supported Parameter Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text values | `"Hello world"` |
| `number` | Floating point numbers | `3.14` |
| `integer` | Whole numbers | `42` |
| `boolean` | True/false values | `true` |
| `array` | Lists of values | `[1, 2, 3]` |
| `object` | Nested objects | `{"key": "value"}` |

## Model Compatibility

RKLLama's tool calling works with multiple model formats:

| Model Type | Format | Detection Method | Support Level |
|------------|--------|------------------|---------------|
| **Qwen 2.5+** | `<tool_call></tool_call>` tags | Standard detection | ✅ Native |
| **Llama 3.2+** | Generic JSON | JSON pattern matching | ✅ Full |
| **Other LLMs** | Generic JSON | Fallback parsing | ✅ Compatible |

### Qwen Models (Recommended)

Qwen models use built-in tool calling with `<tool_call>` tags:

```
<tool_call>
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
</tool_call>
```

### Llama 3.2+ Models

These models output JSON directly:

```json
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
```

## Advanced Features

### Multiple Tool Calls

Models can call multiple tools in a single response:

```json
{
  "message": {
    "role": "assistant",
    "tool_calls": [
      {
        "function": {
          "name": "get_weather",
          "arguments": {"location": "Paris"}
        }
      },
      {
        "function": {
          "name": "get_time",
          "arguments": {"timezone": "Europe/Paris"}
        }
      }
    ]
  }
}
```

### Streaming Tool Calls

Tool calls work seamlessly with streaming responses. The tool call information is sent as the final chunk:

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b",
    "messages": [{"role": "user", "content": "Get weather for NYC"}],
    "tools": [/* tool definitions */],
    "stream": true
  }'
```

### Tool Call Detection Methods

RKLLama uses a multi-stage detection system:

1. **Standard Format Detection**: Looks for `<tool_call>...</tool_call>` tags
2. **Generic JSON Extraction**: Parses any JSON with `name` + `arguments`/`parameters`
3. **Format Normalization**: Converts `parameters` to `arguments` for consistency

## Complete Examples

### Example 1: Weather Tool

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b",
    "messages": [
      {"role": "user", "content": "What is the weather like in Tokyo?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city name"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

### Example 2: Calculator Tool

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b",
    "messages": [
      {"role": "user", "content": "Calculate 25 * 4 + 10"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculator",
          "description": "Perform mathematical calculations",
          "parameters": {
            "type": "object",
            "properties": {
              "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
              }
            },
            "required": ["expression"]
          }
        }
      }
    ]
  }'
```

### Example 3: Multiple Tools

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b",
    "messages": [
      {"role": "user", "content": "Get the weather in Paris and current time there"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather information",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "get_time",
          "description": "Get current time in a timezone",
          "parameters": {
            "type": "object",
            "properties": {
              "timezone": {"type": "string"}
            },
            "required": ["timezone"]
          }
        }
      }
    ]
  }'
```

## Error Handling

### Common Issues and Solutions

#### 1. Tool Not Called
**Problem**: Model doesn't call the tool despite appropriate prompt.

**Solutions**:
- Make tool descriptions more specific
- Use more explicit language in your prompt
- Try different model (Qwen models work best)

#### 2. Invalid JSON Format
**Problem**: Model outputs malformed JSON.

**Solution**: RKLLama includes robust JSON parsing with automatic fixes for:
- Single quotes → double quotes
- Missing quotes around keys
- Extra text around JSON

#### 3. Missing Required Parameters
**Problem**: Tool call missing required fields.

**Solution**: Check the response for validation errors:
```json
{
  "error": "Missing required field: location"
}
```

## Best Practices

### 1. Tool Design

- **Clear descriptions**: Be specific about what each tool does
- **Descriptive parameters**: Include good descriptions for all parameters
- **Proper types**: Use appropriate JSON schema types
- **Required fields**: Mark essential parameters as required

### 2. Prompt Engineering

- **Be specific**: "Get the weather in Paris" vs "What's the weather?"
- **Include context**: Mention when you want tools to be used
- **Use examples**: Show the model how to use tools

### 3. Model Selection

- **Qwen models**: Best native tool calling support
- **Llama 3.2+**: Good JSON output, works well with generic detection
- **Other models**: May work but require more specific prompting

## Debugging

### Enable Debug Mode

For detailed tool calling logs:

```bash
rkllama serve --debug
```

This provides detailed logs showing:
- Tool call detection process
- JSON parsing attempts
- Validation results
- Error details

### Debug Logs Example

```
2024-12-21 10:30:00 - rkllama.format_utils - DEBUG - Searching tools with standard method: get_tool_calls_standard
2024-12-21 10:30:00 - rkllama.format_utils - DEBUG - Searching tools with generic method: get_tool_calls_generic
2024-12-21 10:30:00 - rkllama.server_utils - DEBUG - Tool calls detected: [{'function': {'name': 'get_weather', 'arguments': {'location': 'Paris'}}}]
```

## Troubleshooting

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `"No tools called"` | Model didn't use any tools | Improve prompt or tool descriptions |
| `"Invalid JSON in tool call"` | Malformed JSON | Check model output, try different model |
| `"Missing required field: X"` | Tool call missing required parameter | Verify tool definition and model output |
| `"Tool call detection failed"` | Unable to parse tool calls | Enable debug mode for detailed logs |

### Tips for Success

1. **Start simple**: Begin with one tool and basic parameters
2. **Test incrementally**: Add complexity gradually
3. **Use debug mode**: Enable logging to see what's happening
4. **Check model compatibility**: Qwen models work best
5. **Validate responses**: Always check for `tool_calls` in responses

## Integration Examples

### Python Client

```python
import requests

def call_with_tools(prompt, tools):
    response = requests.post('http://localhost:8080/api/chat', json={
        'model': 'qwen2.5:3b',
        'messages': [{'role': 'user', 'content': prompt}],
        'tools': tools
    })
    
    data = response.json()
    if 'tool_calls' in data.get('message', {}):
        return data['message']['tool_calls']
    return None

# Example usage
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}

tools_called = call_with_tools("What's the weather in Tokyo?", [weather_tool])
if tools_called:
    print("Tools called:", tools_called)
```

### Node.js Client

```javascript
const axios = require('axios');

async function callWithTools(prompt, tools) {
    try {
        const response = await axios.post('http://localhost:8080/api/chat', {
            model: 'qwen2.5:3b',
            messages: [{role: 'user', content: prompt}],
            tools: tools
        });
        
        if (response.data.message?.tool_calls) {
            return response.data.message.tool_calls;
        }
        return null;
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
        return null;
    }
}

// Example usage
const weatherTool = {
    type: "function",
    function: {
        name: "get_weather",
        description: "Get weather information",
        parameters: {
            type: "object",
            properties: {
                location: {type: "string"}
            },
            required: ["location"]
        }
    }
};

callWithTools("What's the weather in Tokyo?", [weatherTool])
    .then(tools => console.log('Tools called:', tools));
```

## Summary

RKLLama's tool calling system provides:

- ✅ **Full Ollama compatibility**
- ✅ **Multiple model format support**
- ✅ **Robust JSON parsing**
- ✅ **Streaming support**
- ✅ **Comprehensive error handling**
- ✅ **Production-ready reliability**

With zero configuration required, tool calling works out of the box with any model that can output structured JSON, making RKLLama a powerful choice for building AI applications that need to interact with external systems.
