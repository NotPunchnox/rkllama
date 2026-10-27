"""Regression gates from independent review; no model/native execution."""
import importlib.util
import json
import threading
import time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from flask import Flask
from rkllama.api.format_utils import get_tool_calls
from rkllama.api.request_options import RequestOptionError, validate_request_options

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}}]

@pytest.mark.parametrize("schema", [{"type": "object", "properties": {"x": {"type": ["integer", "null"]}}}, {"type": "object", "properties": []}, [], {"type": "object", "properties": {"x": None}}])
def test_unsupported_schema_shape_rejected_before_inference(schema):
    with pytest.raises(RequestOptionError):
        validate_request_options({"tools": [{"type": "function", "function": {"name": "lookup", "parameters": schema}}]}, openai=True)

@pytest.mark.parametrize("text", [
    '{"name":"other","arguments":{"city":"Berlin"}}',
    '<tool_call><function=lookup><parameter=city>{"name":"lookup","arguments":{"city":"Berlin"}}',
    '<tool_call><function=lookup><parameter=city>Berlin</parameter><parameter=city>Paris</parameter></function></tool_call>',
    '<tool_call>{"name":"lookup","arguments":{"city":"Berlin"}}</tool_call><tool_call>{"name":"other","arguments":{}}</tool_call>',
    '<tool_call>{"name":"lookup","arguments":{"city":"Berlin"}}</tool_call><tool_call>',
    '{"name":"lookup","arguments":[]}',
    '{"name":"lookup","arguments":{"city":123}}',
])
def test_invalid_tool_call_never_salvaged_by_generic_fallback(text):
    assert get_tool_calls(text, TOOLS) == []


def load_worker():
    spec = importlib.util.spec_from_file_location("rkllama.api.worker_review", Path(__file__).parents[1]/"src/rkllama/api/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_abort_joins_delayed_callbacks_before_reuse(monkeypatch):
    from rkllama.api import callback, rkllm
    worker = load_worker()
    done = threading.Event()
    started = threading.Event()
    sent = []
    flag = SimpleNamespace(value=True)
    def run(*args):
        started.set()
        done.wait(1)
        time.sleep(.02)
        callback.global_text.append("late old callback")
    model = Mock()
    model.rkllm_param = SimpleNamespace(max_context_len=128)
    model.run.side_effect = run
    model.abort.side_effect = lambda: (started.wait(1), done.set())
    model.clear_cache.side_effect = lambda: (_ for _ in ()).throw(AssertionError("thread still active")) if not done.is_set() or callback.global_text != ["late old callback"] else None
    monkeypatch.setattr(rkllm, "RKLLM", lambda *a: model)
    connection = SimpleNamespace(send=sent.append, close=lambda: None)
    tasks = Queue()
    tasks.put((connection, worker.WORKER_TASK_INFERENCE, 0, 1, ([1], None), {}))
    tasks.put((connection, worker.WORKER_TASK_UNLOAD_MODEL, 0, 1, None, {}))
    worker.run_rkllm_worker("test", tasks, Queue(), flag, "/model", "/test")
    assert sent and isinstance(sent[0], dict) and sent[0]["error"]["code"] == "inference_aborted"
    assert callback.global_text == []


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("timeout", [True, False])
def test_worker_errors_are_not_successful_answers(monkeypatch, stream, timeout):
    from rkllama.api.server_utils import ChatEndpointHandler, variables
    pipe = SimpleNamespace(poll=lambda t: not timeout, recv=lambda: {"error": {"code": "native_inference_error", "message": "failed"}}, close=lambda: None)
    monkeypatch.setattr(variables.worker_manager_rkllm, "inference", lambda *a: pipe)
    monkeypatch.setattr(variables.worker_manager_rkllm, "update_expiration_date_for_model", lambda *a: None, raising=False)
    monkeypatch.setattr(variables.worker_manager_rkllm, "workers", {"test": SimpleNamespace(abort_flag=SimpleNamespace(value=False))}, raising=False)
    with Flask(__name__).app_context():
        if stream:
            response = ChatEndpointHandler.handle_streaming("test", [1], None, None, False)
            chunks = [json.loads(x) for x in response.response]
            assert len(chunks) == 1 and "error" in chunks[0]
            assert not chunks[0].get("done") and "eval_count" not in chunks[0]
        else:
            response, status = ChatEndpointHandler.handle_complete("test", [1], None, None, False)
            assert status == (504 if timeout else 500)
            assert "error" in response.get_json()


def test_preflight_uses_loaded_worker_capacity(monkeypatch):
    from rkllama.api.server_utils import EndpointHandler, variables
    monkeypatch.setattr(variables.worker_manager_rkllm, "workers", {"test": SimpleNamespace(native_context_tokens=128)}, raising=False)
    with pytest.raises(RequestOptionError):
        EndpointHandler.validate_prompt_capacity(list(range(100)), {"num_ctx": 8192, "max_new_tokens": 32}, "messages", "test")


def test_length_limit_is_not_reported_as_stop():
    from rkllama.api.server_utils import ChatEndpointHandler, GenerateEndpointHandler
    metrics = {"finish_reason": "length"}
    assert ChatEndpointHandler.format_streaming_chunk("test", "", True, metrics)["done_reason"] == "length"
    assert GenerateEndpointHandler.format_complete_response("test", "a", metrics)["done_reason"] == "length"
    assert GenerateEndpointHandler.format_streaming_chunk("test", "", True, metrics)["done_reason"] == "length"


def test_nonstream_openai_conversion_keeps_length_reason():
    from rkllama.api.format_utils import ollama_chat_to_openai_v1_chat_completion
    result = ollama_chat_to_openai_v1_chat_completion({"done": True, "done_reason": "length", "message": {"role": "assistant", "content": "1"}})
    assert result["choices"][0]["finish_reason"] == "length"

def test_decode_limit_counts_first_token_even_if_sdk_perf_omits_it(monkeypatch):
    from rkllama.api import callback, rkllm
    worker = load_worker()
    sent = []
    connection = SimpleNamespace(send=sent.append, close=lambda: None)
    model = Mock()
    model.rkllm_param = SimpleNamespace(max_context_len=8192)
    monkeypatch.setattr(rkllm, "RKLLM", lambda *a: model)
    class FinishedThread:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def join(self, **kwargs):
            callback.global_metrics[:] = [85, 95, 10, 100]
            callback.global_generation_steps = 96
        def is_alive(self): return False
    monkeypatch.setattr(worker.threading, "Thread", FinishedThread)
    tasks = Queue()
    tasks.put((connection, worker.WORKER_TASK_INFERENCE, 0, 1, ([1], None), {"max_new_tokens": 96}))
    tasks.put((connection, worker.WORKER_TASK_UNLOAD_MODEL, 0, 1, None, {}))
    worker.run_rkllm_worker("test", tasks, Queue(), SimpleNamespace(value=False), "/model", "/test")
    assert sent[-1][-1] == "length"
