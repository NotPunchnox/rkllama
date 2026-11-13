"""
Off-device regression test: a worker child forked from the server must die on
SIGTERM instead of running the server's shutdown handler.

Workers are forked from the server process and inherit the SIGTERM/SIGINT
handler WorkerManager installs there. When the parent terminate()s a child
(failed model load, finished image generation), the child ran stop_all() on its
copy of the workers dict and sent ABORT/UNLOAD to the task queues of every
other loaded model, which unloaded them.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no ctypes
runtime). Needs the "fork" start method, which is what Linux (the board) uses
by default.
"""
import multiprocessing
import queue
import signal
import sys
import time
import types

import pytest

from test_worker_pipe_guard import _install_fakes, _load_worker_module


def _fresh_worker_module():
    FakeRKLLM = _install_fakes()
    return FakeRKLLM, _load_worker_module()


@pytest.fixture
def env(monkeypatch):
    """Fresh fakes + worker module + a WorkerManager built by the REAL __init__
    (server SIGTERM handler), minus its host side effects: no orphan
    killing, no monitor thread. Restores signal handlers and empties the
    manager afterwards so its atexit stop_all() has nothing to stop."""
    FakeRKLLM, worker_mod = _fresh_worker_module()
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    monkeypatch.setattr(worker_mod, "_kill_orphaned_workers", lambda: None)
    monkeypatch.setattr(worker_mod.WorkerManager, "start_models_monitor", lambda self, interval=60: None)
    manager = worker_mod.WorkerManager()
    yield types.SimpleNamespace(FakeRKLLM=FakeRKLLM, worker_mod=worker_mod, manager=manager)
    manager.workers.clear()
    for sig, handler in saved.items():
        signal.signal(sig, handler)



def _slow_rkllm_child(env, monkeypatch, ctx, loading):
    def slow_load(self, *args, **kwargs):
        loading.set()
        time.sleep(30)  # parent terminate()s us mid-load

    monkeypatch.setattr(env.FakeRKLLM, "__init__", slow_load)
    return env.worker_mod.run_rkllm_worker, (
        "failing-model", ctx.Queue(), ctx.Queue(), ctx.Value("b", False), "unused", "unused")


def _slow_rknn_child(env, monkeypatch, ctx, loading):
    class SlowRKNNLite:
        def __init__(self, verbose=False):
            pass

        def load_rknn(self, path):
            loading.set()
            time.sleep(30)

        def init_runtime(self):
            pass

    rknn_lite = types.ModuleType("rknnlite.api.rknn_lite")
    rknn_lite.RKNNLite = SlowRKNNLite
    for name, module in {
        "rknnlite": types.ModuleType("rknnlite"),
        "rknnlite.api": types.ModuleType("rknnlite.api"),
        "rknnlite.api.rknn_lite": rknn_lite,
        "onnxruntime": types.ModuleType("onnxruntime"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(env.worker_mod, "get_rknn_onnx_files_from_model", lambda model_dir: ["model.rknn"])
    return env.worker_mod.run_rknn_worker, ("failing-rknn-model", ctx.Queue(), ctx.Queue(), "unused")


def _slow_image_child(env, monkeypatch, ctx, loading):
    def slow_generate(*args):
        loading.set()
        time.sleep(30)

    image_generator = types.ModuleType("rkllama.api.image_generator")
    image_generator.generate_image = slow_generate
    monkeypatch.setitem(sys.modules, "rkllama.api.image_generator", image_generator)
    _, child_conn = ctx.Pipe()
    return env.worker_mod.run_image_generator, (("sd-model", "a cat", "512x512", 1, 1, 7.5), child_conn)


@pytest.mark.parametrize("make_child", [_slow_rkllm_child, _slow_rknn_child, _slow_image_child],
                         ids=["rkllm-worker", "rknn-worker", "image-generator"])
def test_terminated_child_does_not_unload_other_models(env, monkeypatch, make_child):
    ctx = multiprocessing.get_context("fork")

    # Another model already loaded: its task queue is shared with every forked child
    other_queue = ctx.Queue()
    env.manager.workers["other-model"] = types.SimpleNamespace(process=None, task_queue=other_queue)

    loading = ctx.Event()
    target, args = make_child(env, monkeypatch, ctx, loading)
    child = ctx.Process(target=target, args=args)
    child.start()
    assert loading.wait(10), "child never reached its slow load"
    child.terminate()
    child.join(10)

    assert child.exitcode == -signal.SIGTERM, (
        f"child exit code {child.exitcode}: it ran a SIGTERM handler instead of dying on the signal"
    )
    try:
        leaked = other_queue.get(timeout=1)
    except queue.Empty:
        leaked = None
    assert leaked is None, f"terminated child sent {leaked!r} to another model's worker"
