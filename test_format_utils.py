"""Regression tests for the Ollama-to-OpenAI response conversion in format_utils.

Run with `python -m pytest test_format_utils.py` (needs flask, cv2, numpy,
pydantic, requests, pillow installed; no NPU hardware required).

format_utils.py is loaded by file path, not `import rkllama.api.format_utils`,
because `rkllama/api/__init__.py` pulls in sibling modules that load the RKNN
NPU runtime at import time, which is unavailable off Rockchip hardware.
format_utils's own imports do not depend on that runtime.
"""
import importlib.util
import json
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_SPEC = importlib.util.spec_from_file_location(
    "rkllama_format_utils_under_test",
    os.path.join(_SRC_DIR, "rkllama", "api", "format_utils.py"),
)
format_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(format_utils)


def _base_ollama_response(**overrides):
    response = {
        "model": "qwen2.5:0.5b",
        "message": {"role": "assistant", "content": "hi"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 1,
        "eval_duration": 123_000_000,
        "prompt_eval_duration": 456_000_000,
        "total_duration": 789_000_000,
        "load_duration": 0,
    }
    response.update(overrides)
    return response


def test_chat_completion_handles_none_eval_duration():
    # Reporter's exact case (rkllama#162): a short/aborted generation or a
    # prompt-cache hit returns the key present but valued None.
    result = format_utils.ollama_chat_to_openai_v1_chat_completion(
        _base_ollama_response(eval_duration=None)
    )
    assert result["usage"]["eval_duration"] == 0


def test_chat_completion_handles_none_for_every_duration_field():
    # Same fault class on the other three duration fields in this function.
    for key in ("eval_duration", "prompt_eval_duration", "total_duration", "load_duration"):
        result = format_utils.ollama_chat_to_openai_v1_chat_completion(
            _base_ollama_response(**{key: None})
        )
        assert result["usage"][key] == 0


def test_chat_completion_still_converts_real_durations():
    result = format_utils.ollama_chat_to_openai_v1_chat_completion(_base_ollama_response())
    assert result["usage"]["eval_duration"] == 0.123
    assert result["usage"]["total_duration"] == 0.789


def _final_stream_line(**overrides):
    chunk = {
        "model": "qwen2.5:0.5b",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "eval_count": 1,
        "prompt_eval_count": 5,
        "eval_duration": 123,
        "prompt_eval_duration": 456,
        "total_duration": 789,
        "load_duration": 0,
    }
    chunk.update(overrides)
    return json.dumps(chunk)


def test_stream_final_chunk_handles_none_eval_duration():
    # Same crash, reached through the streaming path's final-chunk handling.
    lines = list(
        format_utils.ollama_chat_stream_to_openai_chat_completions_chunks(
            [_final_stream_line(eval_duration=None)]
        )
    )
    assert any('"eval_duration": 0' in line for line in lines)
    assert lines[-1] == "data: [DONE]\n\n"
