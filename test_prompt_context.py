"""
Off-device regression tests for an inference that ends without a result,
first seen with a prompt longer than the model's context (Open WebUI with its
builtin tools: 7216 prompt tokens for a 4096-token model).

rkllm_run rejects such a prompt with only a log line and no RKLLM_RUN_FINISH
callback. The worker then crashed on global_metrics[0] (IndexError) or, with
metrics left over from the previous run, reported a fake success with an
empty answer. Now RKLLM.run refuses the prompt itself and records why, and
the worker sends (WORKER_TASK_ERROR, reason) to the client.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no
ctypes runtime).
"""
import importlib.util
import multiprocessing
import sys
import types

import pytest

from test_worker_pipe_guard import SRC, _install_fakes, _load_worker_module


def _fresh_worker_module():
    FakeRKLLM = _install_fakes()
    return FakeRKLLM, _load_worker_module()


# --- the worker reports why an inference produced no result -------------------------

def _run_one_inference(worker_mod, task_queue_items, abort_flag=None):
    task_queue = multiprocessing.Queue()
    abort_flag = abort_flag or multiprocessing.Value("b", False)
    for item in task_queue_items:
        task_queue.put(item)
    _, unload_child = multiprocessing.Pipe()
    task_queue.put((unload_child, worker_mod.WORKER_TASK_UNLOAD_MODEL, None, None, None, None))
    worker_mod.run_rkllm_worker("fake-model", task_queue, multiprocessing.Queue(), abort_flag,
                                model_path="unused", model_dir="unused")


def test_rejected_prompt_reports_reason_not_stale_metrics(monkeypatch):
    FakeRKLLM, worker_mod = _fresh_worker_module()
    # Metrics left over from a previous successful run (e.g. the title task)
    sys.modules["rkllama.api.callback"].global_metrics[:] = [10, 5, 100, 50]

    reason = "The prompt has 7230 tokens, more than the model's context length of 4096 tokens."

    def rejected_run(self, *args):
        # Like rkllm_run on an over-long prompt: no RKLLM_RUN_FINISH callback
        self.last_run_error = reason

    monkeypatch.setattr(FakeRKLLM, "run", rejected_run)

    parent, child = multiprocessing.Pipe()
    _run_one_inference(worker_mod, [(child, worker_mod.WORKER_TASK_INFERENCE, "m", "t", ([1, 2, 3], None), {})])

    assert parent.poll(2), "worker sent nothing back"
    message = parent.recv()
    assert worker_mod.worker_error_message(message) == reason, message


def test_rejected_prompt_without_reason_still_reports_error(monkeypatch):
    FakeRKLLM, worker_mod = _fresh_worker_module()
    sys.modules["rkllama.api.callback"].global_metrics.clear()
    monkeypatch.setattr(FakeRKLLM, "run", lambda self, *args: None)

    parent, child = multiprocessing.Pipe()
    _run_one_inference(worker_mod, [(child, worker_mod.WORKER_TASK_INFERENCE, "m", "t", ([1], None), {})])

    assert parent.poll(2), "worker sent nothing back"
    assert worker_mod.worker_error_message(parent.recv()) is not None


def test_aborted_inference_still_reports_finished_not_error(monkeypatch):
    """An abort also ends without RKLLM_RUN_FINISH metrics. It must keep
    reporting a normal finish with zero stats, not the new 'no result' error."""
    FakeRKLLM, worker_mod = _fresh_worker_module()
    sys.modules["rkllama.api.callback"].global_metrics.clear()
    abort_flag = multiprocessing.Value("b", False)

    def aborted_run(self, *args):
        abort_flag.value = True  # client timed out / disconnected mid-inference

    monkeypatch.setattr(FakeRKLLM, "run", aborted_run)

    parent, child = multiprocessing.Pipe()
    _run_one_inference(worker_mod, [(child, worker_mod.WORKER_TASK_INFERENCE, "m", "t", ([1], None), {})],
                       abort_flag=abort_flag)

    assert parent.poll(2), "worker sent nothing back"
    assert parent.recv() == (worker_mod.WORKER_TASK_FINISHED, 0, 0, 0, 0)
    assert abort_flag.value == 0, "abort flag not reset for the next request"


def test_worker_error_message_keeps_plain_sentinel_and_ignores_finish_tuple():
    _, worker_mod = _fresh_worker_module()
    assert worker_mod.worker_error_message(worker_mod.WORKER_TASK_ERROR)
    assert worker_mod.worker_error_message((worker_mod.WORKER_TASK_ERROR, "why")) == "why"
    assert worker_mod.worker_error_message((worker_mod.WORKER_TASK_FINISHED, 1, 2, 3, 4)) is None
    assert worker_mod.worker_error_message("a normal token") is None


# --- RKLLM.run itself (real rkllm.py, fake runtime) ----------------------------------

def _real_rkllm(monkeypatch, tmp_path, max_context, rkllm_run_returns=0):
    """Load the REAL rkllm.py next to the fakes and build an RKLLM without its
    constructor (which would call the native rkllm_init). Only the attributes
    run() touches on the token-input path are set; rkllm_run is a recorder."""
    _install_fakes()
    try:
        import numpy  # noqa: F401  (rkllm.py imports it; the token path never uses it)
    except ImportError:
        monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    spec = importlib.util.spec_from_file_location("rkllama.api.rkllm_real", str(SRC / "rkllama" / "api" / "rkllm.py"))
    rkllm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rkllm_mod)
    classes = sys.modules["rkllama.api.classes"]

    model = rkllm_mod.RKLLM.__new__(rkllm_mod.RKLLM)
    model.handle = None
    model.rkllm_param = classes.RKLLMParam()
    model.rkllm_param.max_context_len = max_context
    model.rkllm_infer_params = classes.RKLLMInferParam()
    model.prompt_cache_dir = str(tmp_path)

    calls = []

    def fake_rkllm_run(handle, rkllm_input, infer_params, userdata):
        calls.append(rkllm_input)
        return rkllm_run_returns

    model.rkllm_run = fake_rkllm_run

    def run_tokens(n_tokens):
        model.run(classes.RKLLMInferMode.RKLLM_INFER_GENERATE, classes.RKLLMInputType.RKLLM_INPUT_TOKEN,
                  (list(range(n_tokens)), None), {})

    return model, calls, run_tokens


def test_rkllm_run_rejects_overlong_prompt_without_calling_runtime(monkeypatch, tmp_path):
    model, calls, run_tokens = _real_rkllm(monkeypatch, tmp_path, max_context=4096)

    run_tokens(7216)
    assert calls == [], "the over-long prompt was still handed to the native runtime"
    assert "7216 tokens" in model.last_run_error and "4096 tokens" in model.last_run_error, model.last_run_error

    run_tokens(4096)  # exactly the context length still fits
    assert len(calls) == 1, "a prompt of exactly max_context tokens was rejected"
    assert model.last_run_error is None, "the previous run's error was not cleared"


@pytest.mark.parametrize("returned, error_expected", [(-1, True), (0, False)])
def test_rkllm_run_records_runtime_failure_code(monkeypatch, tmp_path, returned, error_expected):
    model, calls, run_tokens = _real_rkllm(monkeypatch, tmp_path, max_context=4096, rkllm_run_returns=returned)

    run_tokens(10)

    assert len(calls) == 1
    if error_expected:
        assert model.last_run_error and str(returned) in model.last_run_error, model.last_run_error
    else:
        assert model.last_run_error is None, model.last_run_error
