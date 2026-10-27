"""HTTP errors must be explicit and occur before model loading/generation."""
from unittest.mock import Mock

import pytest

PATHS = ["/v1/chat/completions", "/v1/completions", "/api/chat", "/api/generate"]


@pytest.fixture
def route(monkeypatch):
    from rkllama.server import server
    from rkllama.api.server_utils import ChatEndpointHandler, GenerateEndpointHandler
    monkeypatch.setattr(server, "is_gguf_model", lambda name: False)
    monkeypatch.setattr(server, "get_property_modelfile", lambda *args: "false")
    load = Mock(return_value=(None, None))
    monkeypatch.setattr(server, "load_model", load)
    monkeypatch.setattr(server.variables.worker_manager_rkllm, "exists_model_loaded", lambda name: False)
    generate = Mock(return_value=({"ok": True}, 200))
    monkeypatch.setattr(ChatEndpointHandler, "handle_request", generate)
    monkeypatch.setattr(GenerateEndpointHandler, "handle_request", generate)
    return server.app.test_client(), load, generate


def request_payload(path, extra):
    data = {"model": "test", "stream": False}
    data.update({"messages": [{"role": "user", "content": "Hello"}]} if "chat" in path else {"prompt": "Hello"})
    data.update(extra)
    return data


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("extra,param", [
    ({"grammar": "root ::= 'yes'"}, "grammar"),
    ({"seed": 1}, "seed"), ({"stop": ["stop"]}, "stop"),
    ({"logit_bias": {"1": 10}}, "logit_bias"),
    ({"typo_option": True}, "typo_option"),
    ({"options": {"seed": 1}}, "options.seed"),
    ({"options": {"grammar": "enum"}}, "options.grammar"),
    ({"options": {"num_ctx": 64}}, "options.num_ctx"),
    ({"response_format": {"type": "json_schema", "json_schema": {"strict": True}}}, "response_format"),
    ({"format": {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["YES"]}}}}, "format"),
    ({"response_format": {"type": "json_object", "strict": True}}, "response_format"),
    ({"enable_thinking": "false"}, "enable_thinking"),
    ({"enable_thinking": False, "think": True}, "think"),
    ({"options": None}, "options"),
    ({"stream": "false"}, "stream"),
    ({"format": "xml"}, "format"),
])
def test_reject_unsupported_options_before_dispatch(route, path, extra, param):
    client, load, generate = route
    response = client.post(path, json=request_payload(path, extra))
    assert response.status_code == 400, response.get_json()
    error = response.get_json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == param
    assert error["message"]
    load.assert_not_called()
    generate.assert_not_called()


@pytest.mark.parametrize("path", PATHS[:2])
@pytest.mark.parametrize("extra,param", [
    ({"max_tokens": 3, "max_completion_tokens": 4}, "max_completion_tokens"),
    ({"temperature": 0.2, "options": {"temperature": 0.3}}, "options.temperature"),
    ({"n": 2}, "n"), ({"tool_choice": "required"}, "tool_choice"),
    ({"best_of": 2}, "best_of"), ({"logprobs": True}, "logprobs"),
    ({"response_format": {"type": "json_object"}, "format": "json"}, "response_format"),
])
def test_reject_openai_ambiguity(route, path, extra, param):
    client, load, generate = route
    response = client.post(path, json=request_payload(path, extra))
    assert response.status_code == 400, response.get_json()
    assert response.get_json()["error"]["param"] == param
    load.assert_not_called()
    generate.assert_not_called()


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("options,param", [
    ({"temperature": True}, "temperature"), ({"temperature": -1}, "temperature"),
    ({"temperature": float("nan")}, "temperature"),
    ({"temperature": float("inf")}, "temperature"),
    ({"top_p": 1.1}, "top_p"), ({"top_k": 1.2}, "top_k"),
    ({"max_new_tokens": 0}, "max_new_tokens"),
    ({"max_new_tokens": 2**40}, "max_new_tokens"),
    ({"mirostat": 3}, "mirostat"), ({"repeat_penalty": 0}, "repeat_penalty"),
])
def test_invalid_sampling_values(route, path, options, param):
    client, load, generate = route
    response = client.post(path, json=request_payload(path, {"options": options}))
    assert response.status_code == 400, response.get_json()
    assert response.get_json()["error"]["param"] == "options." + param
    load.assert_not_called()
    generate.assert_not_called()
