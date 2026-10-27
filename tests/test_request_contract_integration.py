"""Real HTTP/handler/ctypes paths; only model I/O and the C call are replaced."""
from unittest.mock import Mock

import pytest
from flask import jsonify

PATHS = ["/v1/chat/completions", "/v1/completions", "/api/chat", "/api/generate"]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("stream", [False, True])
def test_options_reach_native_call(monkeypatch, tmp_path, path, stream):
    from rkllama.server import server
    from rkllama.api.server_utils import EndpointHandler, ChatEndpointHandler, GenerateEndpointHandler
    from rkllama.api.rkllm import RKLLM
    from rkllama.api.classes import RKLLMInferParam, RKLLMInputType, RKLLMInferMode

    tokenizer = Mock(chat_template="test template")
    tokenizer.apply_chat_template.return_value = "rendered prompt"
    tokenizer.return_value = {"input_ids": [1, 2]}
    monkeypatch.setattr(EndpointHandler, "get_tokenizer", lambda name: tokenizer)
    monkeypatch.setattr(server, "is_gguf_model", lambda name: False)
    monkeypatch.setattr(server, "get_property_modelfile", lambda *args: "true")
    monkeypatch.setattr(server.variables.worker_manager_rkllm, "exists_model_loaded", lambda name: True)
    monkeypatch.setattr("rkllama.api.server_utils.handle_ollama_response", lambda response, **kwargs: iter([response.get_data()]) if kwargs.get("stream") else response)
    captured = []
    native = RKLLM.__new__(RKLLM)
    native.handle = None
    native.prompt_cache_dir = str(tmp_path)
    native.rkllm_infer_params = RKLLMInferParam()

    def capture_call(handle, inputs, params, userdata):
        params = params._obj
        captured.append((params.sampling_params.contents.temperature, params.max_new_tokens))

    native.rkllm_run = capture_call

    def complete(model, prompt, format_spec, *args):
        options = args[-1]
        native.run(RKLLMInferMode.RKLLM_INFER_GENERATE,
                   RKLLMInputType.RKLLM_INPUT_TOKEN, (prompt, None), options)
        assert format_spec == "json"
        return jsonify({"ok": True}), 200

    handler = ChatEndpointHandler if "chat" in path else GenerateEndpointHandler
    monkeypatch.setattr(handler, "handle_complete", complete)
    monkeypatch.setattr(handler, "handle_streaming", lambda *args: complete(*args)[0])
    # Exercise explicit overrides and a following request to detect option leakage.
    for temperature, tokens, think in [(0.25, 7, False), (0.75, 11, True)]:
        data = {"model": "test", "stream": stream, "think": think,
                "options": {"temperature": temperature, "num_predict": tokens},
                "response_format": {"type": "json_object"}}
        data.update({"messages": [{"role": "user", "content": "Hello"}]} if "chat" in path else {"prompt": "Hello"})
        response = server.app.test_client().post(path, json=data, buffered=True)
        assert response.status_code == 200
        assert captured[-1] == pytest.approx((temperature, tokens))
        assert tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"] is think
        assert "valid JSON" in tokenizer.apply_chat_template.call_args.args[0][-1]["content"]
        assert not native.rkllm_infer_params.sampling_params
        assert native.rkllm_infer_params.max_new_tokens == 0


@pytest.mark.parametrize("path", PATHS)
def test_gguf_options_are_delegated(monkeypatch, path):
    from rkllama.server import server
    monkeypatch.setattr(server, "is_gguf_model", lambda name: True)
    forward = Mock(return_value=({"forwarded": True}, 200))
    monkeypatch.setattr(server, "forward_request_to_llama_cpp_worker", forward)
    response = server.app.test_client().post(path, json={"model": "test", "grammar": "backend-owned"})
    assert response.status_code == 200
    forward.assert_called_once()


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("body", [[], None, "not an object"])
def test_non_object_body_is_client_error(path, body):
    from rkllama.server import server
    import json
    response = server.app.test_client().post(path, data=json.dumps(body), content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error"]["type"] == "invalid_request_error"
