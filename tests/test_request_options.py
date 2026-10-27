"""Request contract regressions; no NPU or model download required."""
import copy

import pytest

from rkllama.api.format_utils import (
    openai_to_ollama_chat_request,
    openai_to_ollama_generate_request,
)


@pytest.mark.parametrize("convert", [openai_to_ollama_chat_request, openai_to_ollama_generate_request])
def test_json_object_reaches_native_format(convert):
    payload = {"model": "test", "messages": [{"role": "user", "content": "Hello"}],
               "prompt": "Hello", "response_format": {"type": "json_object"}}
    # Each endpoint receives only its own input field.
    payload.pop("prompt" if convert is openai_to_ollama_chat_request else "messages")
    original = copy.deepcopy(payload)
    result = convert(payload)
    assert result["format"] == "json"
    assert "response_format" not in result
    assert payload == original


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("thinking", [False, True])
def test_thinking_reaches_template(monkeypatch, path, thinking):
    from rkllama.server import server
    from rkllama.api.server_utils import EndpointHandler, ChatEndpointHandler, GenerateEndpointHandler
    from flask import jsonify
    from unittest.mock import Mock

    tokenizer = Mock(chat_template="test template")
    tokenizer.apply_chat_template.return_value = "rendered prompt"
    tokenizer.return_value = {"input_ids": [1, 2]}
    monkeypatch.setattr(EndpointHandler, "get_tokenizer", lambda name: tokenizer)
    monkeypatch.setattr(server, "is_gguf_model", lambda name: False)
    monkeypatch.setattr(server, "get_property_modelfile", lambda *args: "true")
    monkeypatch.setattr(server.variables.worker_manager_rkllm, "exists_model_loaded", lambda name: True)
    handler = ChatEndpointHandler if "chat" in path else GenerateEndpointHandler
    monkeypatch.setattr(handler, "handle_complete", lambda *args: (jsonify({"ok": True}), 200))
    monkeypatch.setattr("rkllama.api.server_utils.handle_ollama_response", lambda response, **kwargs: response)
    payload = {"model": "test", "stream": False, "enable_thinking": thinking}
    payload.update({"messages": [{"role": "user", "content": "Hello"}]} if "chat" in path else {"prompt": "Hello"})
    response = server.app.test_client().post(path, json=payload)
    assert response.status_code == 200, response.get_json()
    assert tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"] is thinking


@pytest.mark.parametrize("convert", [openai_to_ollama_chat_request, openai_to_ollama_generate_request])
@pytest.mark.parametrize("name,value,native", [
    ("temperature", 0.25, "temperature"), ("top_p", 0.7, "top_p"),
    ("top_k", 12, "top_k"), ("presence_penalty", 0.4, "presence_penalty"),
    ("frequency_penalty", -0.5, "frequency_penalty"),
    ("repeat_penalty", 1.2, "repeat_penalty"), ("mirostat", 2, "mirostat"),
    ("mirostat_tau", 4.0, "mirostat_tau"), ("mirostat_eta", 0.2, "mirostat_eta"),
    ("max_tokens", 17, "max_new_tokens"),
    ("max_completion_tokens", 19, "max_new_tokens"),
])
def test_options_reach_native_sampling(convert, name, value, native):
    from rkllama.api.rkllm import RKLLM
    from rkllama.api.model_utils import get_model_full_options
    payload = {"model": "test", name: value}
    payload.update({"messages": [{"role": "user", "content": "Hello"}]} if convert is openai_to_ollama_chat_request else {"prompt": "Hello"})
    converted = convert(payload)
    merged = get_model_full_options("test", "/nonexistent", converted["options"])
    sampling, tokens = RKLLM.__new__(RKLLM).get_sampling_parameters(merged)
    actual = tokens if native == "max_new_tokens" else getattr(sampling, native)
    assert actual == pytest.approx(value)
