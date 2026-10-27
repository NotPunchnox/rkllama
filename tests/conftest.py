"""Isolate eager hardware imports; exercise real converters and Flask routes.

RKLLama's api/__init__.py eagerly imports audio models and starts worker
managers. Bypass only that initializer, and replace hardware/model boundaries.
No inference or deployed-template verification is claimed by these tests.
"""
import pathlib
import logging
import pytest
import sys
import types
from unittest.mock import Mock, patch

import rkllama

api = types.ModuleType("rkllama.api")
api.__path__ = [str(pathlib.Path(next(iter(rkllama.__path__))) / "api")]
sys.modules["rkllama.api"] = api
rkllama.api = api

# Load the real ctypes structures without loading the platform-specific library.
with patch("ctypes.CDLL", return_value=Mock()):
    import rkllama.api.classes

worker = types.ModuleType("rkllama.api.worker")
worker.WorkerManager = lambda *args, **kwargs: Mock(workers={})
sys.modules[worker.__name__] = worker

process = types.ModuleType("rkllama.api.process")
process.Request = Mock()
sys.modules[process.__name__] = process

transformers = types.ModuleType("transformers")
transformers.AutoTokenizer = Mock()
sys.modules[transformers.__name__] = transformers


@pytest.fixture(autouse=True)
def no_application_log_files(monkeypatch):
    # Server import eagerly constructs a FileHandler even when pytest already
    # configured logging. Keep filesystem logging out of contract tests.
    monkeypatch.setattr(logging, "FileHandler", lambda *args, **kwargs: logging.NullHandler())
