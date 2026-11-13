"""
Off-device regression tests for an inference that ends without a result,
first seen with a prompt longer than the model's context (Open WebUI with its
builtin tools: 7216 prompt tokens for a 4096-token model).

The worker reports such a failure either as the bare WORKER_TASK_ERROR
sentinel or as (WORKER_TASK_ERROR, reason). The response readers in
server_utils.py must turn both into an error for the client, not a token.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no
ctypes runtime).
"""
from test_worker_pipe_guard import _install_fakes, _load_worker_module


def _fresh_worker_module():
    FakeRKLLM = _install_fakes()
    return FakeRKLLM, _load_worker_module()


def test_worker_error_message_keeps_plain_sentinel_and_ignores_finish_tuple():
    _, worker_mod = _fresh_worker_module()
    assert worker_mod.worker_error_message(worker_mod.WORKER_TASK_ERROR)
    assert worker_mod.worker_error_message((worker_mod.WORKER_TASK_ERROR, "why")) == "why"
    assert worker_mod.worker_error_message((worker_mod.WORKER_TASK_FINISHED, 1, 2, 3, 4)) is None
    assert worker_mod.worker_error_message("a normal token") is None
