"""
Off-device regression tests for model loading in WorkerManager.

Load race: two simultaneous requests for a model that is not loaded yet (a
client such as Open WebUI sends the chat and its title/tags task together)
both loaded it. Both picked the same free NPU domain; either both failed
("failed to malloc npu memory") or both loaded and the second replaced the
first in WorkerManager.workers, leaving an untracked worker holding NPU
memory until the container restarted.

RAM check: the free-RAM test before a load added psutil's `free` to
`available`, which already includes it, so the least recently used model
was evicted too late or not at all.

Uses the fake-module harness from test_worker_pipe_guard.py (no NPU, no
ctypes runtime).
"""
import signal
import threading
import time
import types
from datetime import datetime, timedelta

import pytest

from test_worker_pipe_guard import _install_fakes, _load_worker_module

GB = 1_000_000_000


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


# --- RAM check and least-recently-used eviction -----------------------------------

def test_memory_check_does_not_count_free_ram_twice(monkeypatch):
    """psutil's `available` (MemAvailable) already includes `free` (MemFree).
    Adding them made a 5.4 GB load look like it fits into 4.0 GB available,
    so no model was evicted and the load could run the box out of RAM."""
    _, worker_mod = _fresh_worker_module()
    fake_vm = types.SimpleNamespace(available=4 * GB, free=int(2.5 * GB))
    monkeypatch.setattr(worker_mod.psutil, "virtual_memory", lambda: fake_vm)
    manager = worker_mod.WorkerManager.__new__(worker_mod.WorkerManager)  # the check uses no instance state

    model_size = int(4.5 * GB)  # needs 4.5 * 1.2 = 5.4 GB
    assert not manager.is_memory_available_for_model(model_size), "5.4 GB needed, only 4.0 GB available"
    assert manager.is_memory_available_for_model(3 * GB), "3.6 GB needed, 4.0 GB available"


def test_add_worker_evicts_least_recently_used_model_until_ram_fits(env, monkeypatch):
    # 3 GB really available (2 GB of it free); the new model needs 3.5 * 1.2 = 4.2 GB.
    # Evicting the oldest model (2 GB) is enough; the double-counting formula
    # (3 + 2 = 5 GB) would have evicted nothing.
    fake_vm = types.SimpleNamespace(available=3 * GB, free=2 * GB)
    monkeypatch.setattr(env.worker_mod.psutil, "virtual_memory", lambda: fake_vm)
    monkeypatch.setattr(env.worker_mod.time, "sleep", lambda seconds: None)  # eviction waits 1 s per model

    now = datetime.now()
    for name, minutes_ago, size, domain in [("newest", 1, 1 * GB, 7), ("oldest", 30, 2 * GB, 9), ("middle", 10, 1 * GB, 8)]:
        env.manager.workers[name] = types.SimpleNamespace(worker_model_info=types.SimpleNamespace(
            model=name, size=size, base_domain_id=[domain], last_call=now - timedelta(minutes=minutes_ago)))

    evicted = []

    def fake_stop_worker(model_name, timeout=30):
        evicted.append(model_name)
        fake_vm.available += env.manager.workers.pop(model_name).worker_model_info.size

    monkeypatch.setattr(env.manager, "stop_worker", fake_stop_worker)
    created = []
    _use_fake_workers(env, monkeypatch, created, size=int(3.5 * GB), ram_ok=False)

    assert env.manager.add_worker("qwen-4b", "4b.rkllm", "dir") is True
    assert evicted == ["oldest"], f"evicted {evicted}, expected only the least recently used model"
    assert set(env.manager.workers) == {"newest", "middle", "qwen-4b"}
