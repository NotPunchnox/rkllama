# **RKLLama REST API Documentation**

## **Base URL**

```
http://localhost:8080/
```

---

## **Quick Reference**
### Key Commands:
- **List available models**: `GET /models`  
- **Load a model**: `POST /load_model`  
- **Unload the current model**: `POST /unload_model`  
- **Get the current loaded model**: `GET /current_model`  
- **Generate output**: `POST /generate`  
- **Download a model from Hugging Face**: `POST /pull`  
- **Delete a model**: `POST /rm`  

---

## **API Endpoints**

### **1. GET /models**
#### **Description**
Returns the list of available models in the `~/RKLLAMA/models` directory.

#### **Request**
```http
GET /models
```

#### **Response**
- **200 OK**: List of available models.
  ```json
  {
    "models": [
      "model1.rkllm",
      "model2.rkllm",
      "model3.rkllm"
    ]
  }
  ```

- **500 Internal Server Error**: Directory not found.
  ```json
  {
    "error": "The directory ~/RKLLAMA/models is not found."
  }
  ```

#### **Example**
```bash
curl -X GET http://localhost:8080/models
```

---

### **2. POST /load_model**
#### **Description**
Loads a specific model into memory.

#### **Request**
```http
POST /load_model
Content-Type: application/json
```

##### **Parameters**
```json
{
  "model_name": "model_name.rkllm"
}
```

#### **Response**
- **200 OK**: Model loaded successfully.
  ```json
  {
    "message": "Model <model_name> loaded successfully."
  }
  ```

- **400 Bad Request**: A model is already loaded or parameters are missing.
  ```json
  {
    "error": "A model is already loaded. Please unload it first."
  }
  ```

- **400 Bad Request**: Model not found.
  ```json
  {
    "error": "Model <model_name> not found in the /models directory."
  }
  ```

#### **Example**
```bash
curl -X POST http://localhost:8080/load_model \
-H "Content-Type: application/json" \
-d '{"model_name": "model1.rkllm"}'
```

---

### **3. POST /unload_model**
#### **Description**
Unloads the currently loaded model.

#### **Request**
```http
POST /unload_model
```

#### **Response**
- **200 OK**: Success.
  ```json
  {
    "message": "Model unloaded successfully."
  }
  ```

- **400 Bad Request**: No model is loaded.
  ```json
  {
    "error": "No model is currently loaded."
  }
  ```

#### **Example**
```bash
curl -X POST http://localhost:8080/unload_model
```

---

### **4. GET /current_model**
#### **Description**
Returns the name of the currently loaded model.

#### **Request**
```http
GET /current_model
```

#### **Response**
- **200 OK**: Success.
  ```json
  {
    "model_name": "model_name"
  }
  ```

- **404 Not Found**: No model is loaded.
  ```json
  {
    "error": "No model is currently loaded."
  }
  ```

#### **Example**
```bash
curl -X GET http://localhost:8080/current_model
```

---

### **5. POST /generate**
#### **Description**
Generates a response using the loaded model.

#### **Request**
```http
POST /generate
Content-Type: application/json
```

##### **Parameters**
```json
{
  "messages": "prompt or chat_template",
  "stream": true
}
```

#### **Response**
- **200 OK**: Generated response.
  ```json
  {
    "id": "rkllm_chat",
    "object": "rkllm_chat",
    "created": null,
    "choices": [{
      "role": "assistant",
      "content": "rkllama_output",
      "finish_reason": "stop"
    }],
    "usage": {
      "prompt_tokens": null,
      "completion_tokens": null,
      "total_tokens": null
    }
  }
  ```

- **400 Bad Request**: No model is loaded.
  ```json
  {
    "error": "No model is currently loaded."
  }
  ```

#### **Example**
```bash
curl -X POST http://localhost:8080/generate \
-H "Content-Type: application/json" \
-d '{"messages": "Hello, how are you?", "stream": false}'
```

---

### **6. POST /pull**
#### **Description**
Downloads and installs a model from Hugging Face.

#### **Request**
```http
POST /pull
Content-Type: application/json
```

##### **Parameters**
```json
{
  "model": "hf/username/repo/file.rkllm"
}
```

#### **Response**
- **200 OK**: Download in progress.
```txt
Downloading <file> (<size> MB)...
<progress>%
```

- **400 Bad Request**: Download error.
```txt
Error during download: <error>
```

#### **Example**
```bash
curl -X POST http://localhost:8080/pull \
-H "Content-Type: application/json" \
-d '{"model": "hf/username/repo/file.rkllm"}'
```

---

### **7. DELETE /rm**
#### **Description**
Deletes a specific model.

#### **Request**
```http
POST /rm
Content-Type: application/json
```

##### **Parameters**
```json
{
  "model": "model_name.rkllm"
}
```

#### **Response**
- **200 OK**: Success.
  ```json
  {
    "message": "The model has been successfully deleted."
  }
  ```

- **404 Not Found**: Model not found.
  ```json
  {
    "error": "The model: {model} cannot be found."
  }
  ```

#### **Example**
```bash
curl -X DELETE http://localhost:8080/rm \
-H "Content-Type: application/json" \
-d '{"model": "model1.rkllm"}'
```

---

## **Tool/Function Calling**

RKLLama supports comprehensive tool calling functionality with full Ollama API compatibility. Tools allow models to call external functions with structured parameters.

### **Supported Endpoints**
- **`/api/chat`** - Full tool calling support  
- **`/api/generate`** - Tool detection in generated text

### **Basic Tool Call Request**

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
                "description": "Temperature unit to use"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

### **Tool Call Response Format**

When a model calls a tool, the response includes `tool_calls`:

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
  "done": true
}
```

### **Model Compatibility**

| Model Type | Format | Support Level |
|------------|--------|---------------|
| **Qwen 2.5+** | `<tool_call></tool_call>` tags | ✅ Native |
| **Llama 3.2+** | Generic JSON detection | ✅ Full |
| **Other LLMs** | Fallback JSON parsing | ✅ Compatible |

### **Features**

- **Multiple detection methods** for different model formats
- **Streaming support** - tool calls work in both streaming and non-streaming modes
- **Format normalization** - automatically handles `parameters` vs `arguments`
- **Robust JSON parsing** with fallback methods
- **Multiple tool calls** in a single response

For complete documentation, see the [Tool Calling Guide](./tools.md).

---

### **8. GET /**
#### **Description**
Displays a welcome message and a link to the GitHub project.

#### **Response**
- **200 OK**:
  ```json
  {
    "message": "Welcome to RK-LLama!",
    "github": "https://github.com/notpunhnox/rkllama"
  }
  ```

#### **Example**
```bash
curl -X GET http://localhost:8080/
```

---

## **Error Handling**
- **400**: Bad Request due to incorrect parameters.  
- **404**: Resource not found.  
- **500**: Internal server error.

---

## **Practical Tips**
- **Parameter Validation**: Always double-check model names and file paths.  
- **Troubleshooting**: Check server logs for more details on internal errors.