"""
Regression test for issue #169: a worker process crashes with an uncaught
BrokenPipeError when the API layer times out on a request, closes its end
of the per-request Pipe, and the worker's abort/error-report path later
tries to write to that now-dead pipe.

`worker.py` cannot be imported as `rkllama.api.worker` through the normal
package path: `rkllama/api/__init__.py` eagerly imports `.classes`, which
calls `ctypes.CDLL()` on `librkllmrt.so`, an AArch64-only Rockchip NPU
runtime library that cannot load on this (or any non-Rockchip) machine.
So this harness loads `worker.py` directly by file path and pre-registers
fake `rkllama.api.{classes,rkllm,callback}` modules in sys.modules,
mirroring the real module boundary the deferred `from .rkllm import RKLLM`
/ `from .callback import ...` imports inside `run_rkllm_worker` resolve
against. The worker's own control flow (the code under test) is real
and unmodified, only the native-hardware model backend is a stand-in.

Also verifies, by construction with a plain `multiprocessing.Pipe()`
(POSITIVE CONTROL), that sending on a connection whose peer already
closed really does raise BrokenPipeError in this environment, so a
clean run below is not a dead instrument.
"""
import importlib.util
import multiprocessing
import sys
import threading
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))


def _install_fakes():
    """Pre-register the modules worker.py's package-relative imports
    resolve against, so importing it never touches the real ctypes/NPU
    stack or the transformers/flask-heavy rest of the `rkllama.api`
    package. Returns the FakeRKLLM class so tests can drive its timing.
    """
    import ctypes

    import rkllama.config  # noqa: F401  (real, pure-python, safe to import)

    api_pkg = types.ModuleType("rkllama.api")
    api_pkg.__path__ = [str(SRC / "rkllama" / "api")]
    sys.modules["rkllama.api"] = api_pkg

    # Load the REAL classes.py (all its ctypes.Structure/CFUNCTYPE
    # declarations, incl. LLMResultCallback_type that worker.py needs at
    # import time via `from .classes import *`) with ctypes.CDLL stubbed
    # out for the one line that would otherwise try to dlopen the
    # AArch64-only librkllmrt.so.
    real_cdll = ctypes.CDLL
    ctypes.CDLL = lambda *a, **k: types.SimpleNamespace()
    try:
        classes_spec = importlib.util.spec_from_file_location(
            "rkllama.api.classes", str(SRC / "rkllama" / "api" / "classes.py")
        )
        classes_mod = importlib.util.module_from_spec(classes_spec)
        sys.modules["rkllama.api.classes"] = classes_mod
        classes_spec.loader.exec_module(classes_mod)
    finally:
        ctypes.CDLL = real_cdll

    callback_mod = types.ModuleType("rkllama.api.callback")
    callback_mod.callback_impl = lambda *a, **k: None
    callback_mod.global_text = []
    callback_mod.last_embeddings = []
    callback_mod.global_metrics = []
    sys.modules["rkllama.api.callback"] = callback_mod

    class FakeRKLLM:
        """Stands in for the real ctypes-backed model. `run()` is invoked
        on a background thread by `run_rkllm_worker`, exactly like the
        real `RKLLM.run`, it just sleeps instead of calling the NPU, then
        populates `global_metrics` the way the real C callback does on
        RKLLM_RUN_FINISH, since `run_rkllm_worker` indexes into it
        unconditionally once the thread completes.
        """

        run_duration = 0.2

        def __init__(self, callback, model_path, model_dir, options=None,
                     lora_model_path=None, prompt_cache_path=None, base_domain_id=0):
            pass

        def run(self, inference_mode, model_input_type, model_input, options):
            time.sleep(self.run_duration)
            metrics = callback_mod.global_metrics
            metrics.clear()
            metrics.extend([10, 5, 100, 50])  # prefill/generate tokens, prefill/generate ms

        def abort(self):
            pass

        def clear_cache(self):
            pass

        def release(self):
            pass

    rkllm_mod = types.ModuleType("rkllama.api.rkllm")
    rkllm_mod.RKLLM = FakeRKLLM
    sys.modules["rkllama.api.rkllm"] = rkllm_mod

    model_utils_spec = importlib.util.spec_from_file_location(
        "rkllama.api.model_utils", str(SRC / "rkllama" / "api" / "model_utils.py")
    )
    model_utils_mod = importlib.util.module_from_spec(model_utils_spec)
    sys.modules["rkllama.api.model_utils"] = model_utils_mod
    model_utils_spec.loader.exec_module(model_utils_mod)

    return FakeRKLLM


def _load_worker_module():
    spec = importlib.util.spec_from_file_location(
        "rkllama.api.worker", str(SRC / "rkllama" / "api" / "worker.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rkllama.api.worker"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_positive_control_dead_peer_send_raises():
    """Same-process send-after-close on a real OS pipe raises
    BrokenPipeError. If this ever stops being true here, the crash test
    below would pass vacuously (green for the wrong reason).
    """
    parent, child = multiprocessing.Pipe()
    parent.close()
    try:
        child.send("x")
    except BrokenPipeError:
        pass
    else:
        raise AssertionError(
            "positive control failed: send on a peer-closed Pipe did not raise "
            "BrokenPipeError in this environment, so the crash test below cannot "
            "be trusted until this is understood"
        )
    finally:
        child.close()


def _run_one_timed_out_task(worker_mod, run_duration):
    """Drive `run_rkllm_worker` through exactly the sequence issue #169
    describes: a client sends an inference task, times out waiting on its
    pipe (client-side behavior simulated here, mirroring what
    ChatEndpointHandler.handle_complete does), and closes its end. Then
    the worker's model finishes late, notices the abort flag, and reports
    the outcome back over the now-dead pipe.
    """
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    abort_flag = multiprocessing.Value("b", False)

    parent_conn, child_conn = multiprocessing.Pipe()
    task_queue.put((
        child_conn,
        worker_mod.WORKER_TASK_INFERENCE,
        "text",  # inference_mode
        "text",  # model_input_type
        "hello", # model_input
        {},      # options
    ))
    _unload_parent_conn, unload_child_conn = multiprocessing.Pipe()
    task_queue.put((
        unload_child_conn,
        worker_mod.WORKER_TASK_UNLOAD_MODEL,
        None, None, None, None,
    ))

    # Simulate the API-layer client: poll with a timeout shorter than the
    # model's run_duration, then behave exactly like ChatEndpointHandler's
    # timeout branch (set the abort flag, close its own end of the pipe).
    def client():
        if not parent_conn.poll(0.05):
            abort_flag.value = True
            parent_conn.close()

    client_thread = threading.Thread(target=client)
    client_thread.start()

    worker_mod.run_rkllm_worker(
        "fake-model", task_queue, result_queue, abort_flag,
        model_path="unused", model_dir="unused",
    )
    client_thread.join(timeout=5)


def test_worker_survives_timed_out_peer():
    FakeRKLLM = _install_fakes()
    worker_mod = _load_worker_module()
    FakeRKLLM.run_duration = 0.2  # finishes AFTER the client's 0.05s timeout

    # Must return normally (having processed the UNLOAD task next in the
    # queue) rather than let a BrokenPipeError from the dead-peer send
    # propagate out and kill the worker process.
    _run_one_timed_out_task(worker_mod, run_duration=0.2)


def test_worker_still_delivers_result_on_the_happy_path():
    """Control: a client that does NOT time out must still receive the
    real finished-inference tuple over the pipe, unchanged. `_safe_send`
    only swallows (BrokenPipeError, OSError); it must not mask a live
    peer's normal delivery.
    """
    FakeRKLLM = _install_fakes()
    worker_mod = _load_worker_module()
    FakeRKLLM.run_duration = 0.05  # finishes BEFORE the client's timeout

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    abort_flag = multiprocessing.Value("b", False)

    parent_conn, child_conn = multiprocessing.Pipe()
    task_queue.put((
        child_conn,
        worker_mod.WORKER_TASK_INFERENCE,
        "text", "text", "hello", {},
    ))
    _unload_parent_conn, unload_child_conn = multiprocessing.Pipe()
    task_queue.put((
        unload_child_conn,
        worker_mod.WORKER_TASK_UNLOAD_MODEL,
        None, None, None, None,
    ))

    worker_thread = threading.Thread(
        target=worker_mod.run_rkllm_worker,
        args=("fake-model", task_queue, result_queue, abort_flag),
        kwargs={"model_path": "unused", "model_dir": "unused"},
    )
    worker_thread.start()

    # Client waits well past the model's run_duration, exactly like a
    # request that completes normally within its configured timeout.
    assert parent_conn.poll(2.0), "client never received the finished signal"
    finished = parent_conn.recv()
    assert isinstance(finished, tuple) and finished[0] == worker_mod.WORKER_TASK_FINISHED, finished
    parent_conn.close()
    worker_thread.join(timeout=5)
    assert not worker_thread.is_alive(), "worker did not exit after the unload task"


if __name__ == "__main__":
    test_positive_control_dead_peer_send_raises()
    print("PASS: positive control (dead-peer send raises BrokenPipeError)")
    test_worker_survives_timed_out_peer()
    print("PASS: run_rkllm_worker survives a client that already closed its pipe")
    test_worker_still_delivers_result_on_the_happy_path()
    print("PASS: a live client still receives the real finished-inference result")
