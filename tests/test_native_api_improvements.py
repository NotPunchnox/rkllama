"""Native API compatibility regressions; fixtures are not inference results."""
import copy
import json
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flask import Response

from rkllama.api.format_utils import (
    handle_ollama_response, openai_to_ollama_chat_request,
    openai_to_ollama_generate_request, ollama_chat_to_openai_v1_chat_completion,
)
from rkllama.api.request_options import RequestOptionError, validate_request_options


@pytest.mark.parametrize("convert", [openai_to_ollama_chat_request, openai_to_ollama_generate_request])
@pytest.mark.parametrize("include", [True, False])
def test_stream_options_survive_conversion(convert, include):
    payload = {"model": "test", "stream": True, "stream_options": {"include_usage": include}}
    payload.update({"messages": []} if convert is openai_to_ollama_chat_request else {"prompt": "test"})
    assert convert(payload)["stream_options"] == {"include_usage": include}


@pytest.mark.parametrize("value,param", [
    (None, "stream_options"), ([], "stream_options"),
    ({"include_usage": 1}, "stream_options.include_usage"),
    ({"include_obfuscation": True}, "stream_options.include_obfuscation"),
])
def test_invalid_stream_options(value, param):
    with pytest.raises(RequestOptionError) as exc:
        validate_request_options({"stream": True, "stream_options": value}, openai=True)
    assert exc.value.param == param


def test_stream_options_requires_streaming():
    with pytest.raises(RequestOptionError, match="stream=true"):
        validate_request_options({"stream_options": {"include_usage": True}}, openai=True)


@pytest.mark.parametrize("is_chat", [True, False])
@pytest.mark.parametrize("include", [True, False])
def test_sse_identity_usage_and_done(is_chat, include):
    from rkllama.server.server import app
    field = {"message": {"role": "assistant", "content": "Hi"}} if is_chat else {"response": "Hi"}
    rows = [{"model": "test", "done": False, **field},
            {"model": "test", "done": True, "done_reason": "stop", "prompt_eval_count": 11,
             "eval_count": 3, "eval_duration": 1000000000}]
    with app.test_request_context():
        output = list(handle_ollama_response(Response((json.dumps(r)+"\n" for r in rows)),
                                            stream=True, is_chat=is_chat, include_usage=include))
    assert output[-1] == "data: [DONE]\n\n"
    chunks = [json.loads(s.removeprefix("data: ")) for s in output[:-1]]
    assert len({c["id"] for c in chunks}) == 1
    assert len({c["created"] for c in chunks}) == 1
    assert all(c["object"] == ("chat.completion.chunk" if is_chat else "text_completion") for c in chunks)
    if include:
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["prompt_tokens"] == 11
        assert chunks[-1]["usage"]["completion_tokens"] == 3
        assert chunks[-1]["usage"]["total_tokens"] == 14
        assert all(c["usage"] is None for c in chunks[:-1])
    else:
        assert all("usage" not in c for c in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate", "/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("stream", [True, False])
def test_oversized_prompt_rejected_before_inference(monkeypatch, path, stream):
    from rkllama.server import server
    from rkllama.api.server_utils import EndpointHandler
    tokenizer = Mock(chat_template="test")
    tokenizer.apply_chat_template.return_value = "rendered prompt"
    tokenizer.return_value = {"input_ids": list(range(100))}
    monkeypatch.setattr(EndpointHandler, "get_tokenizer", lambda name: tokenizer)
    monkeypatch.setattr(server, "is_gguf_model", lambda name: False)
    monkeypatch.setattr(server, "get_property_modelfile", lambda *args: "false")
    monkeypatch.setattr(server, "get_model_full_options", lambda *args: {"num_ctx": 128, "max_new_tokens": 32})
    monkeypatch.setattr(server.variables.worker_manager_rkllm, "exists_model_loaded", lambda name: True)
    inference = Mock(side_effect=AssertionError("must not start inference"))
    monkeypatch.setattr(server.variables.worker_manager_rkllm, "inference", inference)
    body = {"model": "test", "stream": stream}
    body.update({"messages": [{"role": "user", "content": "hello"}]} if "chat" in path else {"prompt": "hello"})
    response = server.app.test_client().post(path, json=body)
    assert response.status_code == 400, response.get_data(as_text=True)
    error = response.get_json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["param"] == ("messages" if "chat" in path else "prompt")
    assert "100" in error["message"] and "32" in error["message"] and "128" in error["message"]
    inference.assert_not_called()


def test_exact_context_budget_is_valid():
    from rkllama.api.request_options import validate_context_budget
    validate_context_budget(96, 32, 128, "messages")


def test_stateless_chat_does_not_select_unverified_disk_cache(monkeypatch):
    from rkllama.api.server_utils import EndpointHandler
    tokenizer = Mock(chat_template="test")
    tokenizer.apply_chat_template.return_value = "rendered"
    tokenizer.return_value = {"input_ids": [1, 2]}
    monkeypatch.setattr(EndpointHandler, "get_tokenizer", lambda name: tokenizer)
    assert EndpointHandler.prepare_prompt("test", [{"role": "user", "content": "hello"}])[1] is None


@pytest.mark.parametrize("choice", ["auto", "none"])
def test_honest_tool_choice_subset(choice):
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    payload = {"model": "test", "messages": [], "tools": tools, "tool_choice": choice}
    original = copy.deepcopy(payload)
    normalized = openai_to_ollama_chat_request(payload)
    assert normalized.get("tools") == (tools if choice == "auto" else None)
    assert "tool_choice" not in normalized
    assert payload == original


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "lookup"}}])
def test_forced_tool_choice_stays_explicitly_unsupported(choice):
    with pytest.raises(RequestOptionError, match="[Ff]orced"):
        validate_request_options({"tool_choice": choice}, openai=True)


def test_strict_tool_schema_is_not_silently_accepted():
    with pytest.raises(RequestOptionError, match="strict"):
        validate_request_options({"tools": [{"type": "function", "function": {"name": "lookup", "strict": True}}]}, openai=True)


def test_openai_tool_arguments_are_valid_json_without_mutation():
    row = {"model": "test", "done": True, "done_reason": "tool_calls", "message": {
        "content": "", "tool_calls": [{"function": {"name": "lookup", "arguments": {"label": "it's fine", "flag": True}}}]}}
    original = copy.deepcopy(row)
    result = ollama_chat_to_openai_v1_chat_completion(row)
    call = result["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"label": "it's fine", "flag": True}
    assert row == original


def test_tool_result_roundtrip_normalizes_json_arguments():
    payload = {"messages": [{"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_test", "type": "function", "function": {"name": "lookup", "arguments": '{"count":2}'}}]},
        {"role": "tool", "tool_call_id": "call_test", "content": "found"}]}
    original = copy.deepcopy(payload)
    result = openai_to_ollama_chat_request(payload)
    assert result["messages"][0]["content"] == ""
    assert result["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"count": 2}
    assert payload == original


def test_qwen35_xml_tool_calls_preserve_declared_types():
    from rkllama.api.format_utils import get_tool_calls
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"code": {"type": "string"}, "count": {"type": "integer"},
        "active": {"type": "boolean"}, "names": {"type": "array"}}}}}]
    text = '<tool_call>\n<function=lookup>\n<parameter=code>\n001\n</parameter>\n<parameter=count>\n2\n</parameter>\n<parameter=active>\ntrue\n</parameter>\n<parameter=names>\n["a","b"]\n</parameter>\n</function>\n</tool_call>'
    assert get_tool_calls(text, tools) == [{"function": {"name": "lookup", "arguments": {
        "code": "001", "count": 2, "active": True, "names": ["a", "b"]}}}]
    assert get_tool_calls(text.replace('</function>', ''), tools) == []


def test_streamed_tool_markup_split_across_many_tokens_does_not_leak(monkeypatch):
    from rkllama.api.server_utils import ChatEndpointHandler, variables
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"label": {"type": "string"}}}}}]
    text = '<tool_call>\n<function=lookup>\n<parameter=label>\n' + 'x'*120 + '\n</parameter>\n</function>\n</tool_call>'
    tokens = iter(list(text) + [("finished", 10, 20, 100, 200)])
    pipe = SimpleNamespace(poll=lambda timeout: True, recv=lambda: next(tokens), close=lambda: None)
    monkeypatch.setattr(variables.worker_manager_rkllm, "inference", lambda *args: pipe)
    monkeypatch.setattr(variables.worker_manager_rkllm, "update_expiration_date_for_model", lambda *args: None, raising=False)
    result = None
    monkeypatch.setattr(variables, "global_status", -1, raising=False)
    result = ChatEndpointHandler.handle_streaming("test", [1], None, tools, False)
    chunks = [json.loads(line) for line in result.response]
    assert all(chunk["message"]["content"] == "" for chunk in chunks)
    assert chunks[0]["message"]["tool_calls"][0]["function"]["arguments"] == {"label": "x"*120}
    assert chunks[-1]["done_reason"] == "tool_calls"


def test_worker_drains_tokens_arriving_during_final_join_and_resets_buffers(monkeypatch):
    import importlib.util
    from pathlib import Path
    from rkllama.api import callback, rkllm
    spec = importlib.util.spec_from_file_location("rkllama.api.worker_regression", Path(__file__).parents[1] / "src/rkllama/api/worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    callback.global_text[:] = ["stale prefix"]
    callback.global_metrics[:] = [999, 999, 999, 999]
    callback.split_byte_data = b"\xe2"
    sent = []
    connection = SimpleNamespace(send=sent.append, close=lambda: None)
    model = Mock()
    model.rkllm_param = SimpleNamespace(max_context_len=128)
    monkeypatch.setattr(rkllm, "RKLLM", lambda *args: model)
    class FinalJoinThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            assert callback.global_text == []
            assert callback.global_metrics == []
            assert callback.split_byte_data == b""
        def join(self, **kwargs):
            callback.global_text.extend(["Answer", "."])
            callback.global_metrics[:] = [10, 2, 5, 6]
        def is_alive(self):
            return False
    monkeypatch.setattr(worker.threading, "Thread", FinalJoinThread)
    tasks = Queue()
    tasks.put((connection, worker.WORKER_TASK_INFERENCE, 0, 1, ([1], None), {}))
    tasks.put((connection, worker.WORKER_TASK_UNLOAD_MODEL, 0, 1, None, {}))
    worker.run_rkllm_worker("test", tasks, Queue(), SimpleNamespace(value=False), "/test/model", "/test")
    assert sent == ["Answer", ".", (worker.WORKER_TASK_FINISHED, 10, 2, 5000000, 6000000, "stop")]
    assert callback.global_text == []
