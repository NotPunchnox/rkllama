"""RKLLM 1.3.1 ABI regressions, independent of accelerator hardware."""
import ctypes
from unittest.mock import Mock

import pytest
from rkllama.api import classes


def test_lp64_multimodal_input_layout():
    if ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_size_t) != 8:
        pytest.skip("Expected sizes are from the vendor's LP64 header")
    assert ctypes.sizeof(classes.RKLLMAudioInput) == 48
    assert classes.RKLLMMultiModalInput.audio.offset == 144
    assert ctypes.sizeof(classes.RKLLMMultiModalInput) == 192
    assert classes.RKLLMInput.input_data.offset == 16
    assert ctypes.sizeof(classes.RKLLMInput) == 208


def test_initial_top_k_has_integer_type():
    assert dict(classes.RKLLMParam._fields_)["top_k"] is ctypes.c_int32
    p = classes.RKLLMParam()
    p.top_k = 40
    assert ctypes.c_int32.from_buffer(p, classes.RKLLMParam.top_k.offset).value == 40


def test_initial_top_k_reaches_native_init(monkeypatch, tmp_path):
    from rkllama.api import rkllm
    from rkllama.api.rkllm import RKLLM
    seen = []

    def initialize(handle, params, callback):
        seen.append(params._obj.top_k)
        return -1  # Stop before optional initialization paths; no native I/O.

    monkeypatch.setattr(rkllm.rkllm_lib, "rkllm_init", Mock(side_effect=initialize))
    callback = classes.LLMResultCallback_type(lambda *args: 0)
    with pytest.raises(RuntimeError, match="Failed to initialize RKLLM model"):
        RKLLM(callback, "unused.rkllm", str(tmp_path), options={"top_k": 40})
    assert seen == [40]
