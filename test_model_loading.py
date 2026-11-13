"""
Off-device regression tests for model loading in WorkerManager.

Load race: two simultaneous requests for a model that is not loaded yet (a
client such as Open WebUI sends the chat and its title/tags task together)
both loaded it. Both picked the same free NPU domain; either both failed
("failed to malloc npu memory") or both loaded and the second replaced the
first in WorkerManager.workers, leaving an untracked worker holding NPU
memory until the container restarted.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no
ctypes runtime).
"""
import signal
import threading
import time
import types
from datetime import datetime

import pytest

from test_worker_pipe_guard import _install_fakes, _load_worker_module


# --- helpers ------------------------------------------------------------------

def _fresh_worker_module():
    FakeRKLLM = _install_fakes()
    return FakeRKLLM, _load_worker_module()


@pytest.fixture
def env(monkeypatch):
    """Fresh fakes + worker module + a WorkerManager built by the REAL __init__
    (load lock, server SIGTERM handler), minus its host side effects: no orphan
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


def _use_fake_workers(env, monkeypatch, created, load_seconds=0.0, outcomes=None, size=0, ram_ok=True):
    """Replace Worker with a stand-in whose load only records itself, waits
    `load_seconds` and returns (or raises) the next scripted outcome."""
    outcomes = list(outcomes or [])

    class FakeWorker:
        def __init__(self, model_name, base_domain_id, loaded_by=None):
            self.worker_model_info = types.SimpleNamespace(
                model=model_name, size=size, base_domain_id=[base_domain_id], last_call=datetime.now())

        def create_worker_process(self, *args, **kwargs):
            created.append(self)
            time.sleep(load_seconds)
            outcome = outcomes.pop(0) if outcomes else True
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(env.worker_mod, "Worker", FakeWorker)
    monkeypatch.setattr(env.worker_mod, "is_rkllm_model", lambda name: True)
    monkeypatch.setattr(env.worker_mod, "is_gguf_model", lambda name: False)
    if ram_ok:
        monkeypatch.setattr(env.manager, "is_memory_available_for_model", lambda size: True)


def _run_concurrently(*calls, timeout=5):
    """Start all calls at the same instant (barrier) and return their results in call order."""
    start = threading.Barrier(len(calls))
    results = [None] * len(calls)

    def runner(i, call):
        start.wait()
        results[i] = call()

    # daemon: a call stuck on a leaked lock fails the test instead of hanging the test run
    threads = [threading.Thread(target=runner, args=(i, c), daemon=True) for i, c in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    assert not any(t.is_alive() for t in threads), "a concurrent call never returned (deadlock?)"
    return results


# --- load lock ------------------------------------------------------------------

def test_concurrent_loads_create_one_worker(env, monkeypatch):
    created = []
    _use_fake_workers(env, monkeypatch, created, load_seconds=0.3)  # both requests overlap the load

    results = _run_concurrently(
        lambda: env.manager.add_worker("qwen", "model.rkllm", "dir"),
        lambda: env.manager.add_worker("qwen", "model.rkllm", "dir"),
    )

    assert len(created) == 1, f"model loaded {len(created)} times for 2 simultaneous requests"
    assert results == [True, True], results
    assert env.manager.workers["qwen"] is created[0]


def test_concurrent_loads_of_different_models_both_load_on_distinct_domains(env, monkeypatch):
    """The lock serializes loads, it must not refuse them; and the second load
    must see the first one registered so it gets a different NPU domain."""
    created = []
    _use_fake_workers(env, monkeypatch, created, load_seconds=0.2)

    results = _run_concurrently(
        lambda: env.manager.add_worker("qwen-2b", "a.rkllm", "dir-a"),
        lambda: env.manager.add_worker("qwen-4b", "b.rkllm", "dir-b"),
    )

    assert results == [True, True], results
    assert set(env.manager.workers) == {"qwen-2b", "qwen-4b"}
    domains = [w.worker_model_info.base_domain_id[0] for w in env.manager.workers.values()]
    assert len(set(domains)) == 2, f"both models were given NPU domain {domains[0]}"


def test_failed_load_registers_nothing_and_next_load_proceeds(env, monkeypatch):
    created = []
    _use_fake_workers(env, monkeypatch, created,
                      outcomes=[False, RuntimeError("load crashed"), True])

    assert env.manager.add_worker("qwen", "model.rkllm", "dir") is False
    assert "qwen" not in env.manager.workers, "a failed load was registered"

    with pytest.raises(RuntimeError, match="load crashed"):
        env.manager.add_worker("qwen", "model.rkllm", "dir")
    assert "qwen" not in env.manager.workers, "a crashed load was registered"

    # Runs in a thread: if the exception above left the lock held, this hangs
    (result,) = _run_concurrently(lambda: env.manager.add_worker("qwen", "model.rkllm", "dir"))
    assert result is True
    assert env.manager.workers["qwen"] is created[2]
