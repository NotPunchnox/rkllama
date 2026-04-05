import ast
import json
import logging
import os
import time
import unittest
from pathlib import Path
from typing import List

import numpy as np


def load_rknn2model_class():
    source_path = Path(__file__).resolve().parents[1] / "src" / "rkllama" / "api" / "image_generator.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))
    class_node = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RKNN2Model"
    )

    namespace = {
        "List": List,
        "np": np,
        "json": json,
        "os": os,
        "time": time,
        "logger": logging.getLogger("test-rknn2model"),
        "RKNNLite": type("RKNNLite", (), {"NPU_CORE_AUTO": object()}),
    }
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["RKNN2Model"]


class FakeTensor:
    def __init__(self, array):
        self._array = np.array(array)
        self.shape = self._array.shape
        self.dtype = self._array.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeRKNNLite:
    def __init__(self, outputs=None):
        self.outputs = outputs if outputs is not None else [np.zeros((1,), dtype=np.float32)]
        self.calls = []

    def inference(self, *, inputs, data_format):
        self.calls.append({"inputs": inputs, "data_format": data_format})
        return self.outputs


class RKNN2ModelInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = load_rknn2model_class()

    def make_model(self, outputs=None):
        model = object.__new__(self.model_class)
        model.loaded = True
        model.unload_after_first_call = False
        model.rknnlite = FakeRKNNLite(outputs=outputs)
        return model

    def test_converts_float_inputs_to_float32_before_inference(self):
        model = self.make_model()
        numpy_input = np.array([[1.0, 2.0]], dtype=np.float64)
        tensor_input = FakeTensor([[3.0, 4.0]])

        result = model(arr=numpy_input, tensor=tensor_input)

        self.assertEqual(len(result), 1)
        sent_inputs = model.rknnlite.calls[0]["inputs"]
        self.assertEqual(sent_inputs[0].dtype, np.float32)
        self.assertEqual(sent_inputs[1].dtype, np.float32)

    def test_keeps_integer_numpy_inputs_unchanged(self):
        model = self.make_model()
        token_ids = np.array([[1, 2, 3]], dtype=np.int64)

        model(tokens=token_ids)

        sent_inputs = model.rknnlite.calls[0]["inputs"]
        self.assertEqual(sent_inputs[0].dtype, np.int64)

    def test_raises_clear_error_when_inference_returns_none(self):
        model = self.make_model(outputs=None)
        model.rknnlite.outputs = None

        with self.assertRaisesRegex(RuntimeError, "RKNN inference returned no outputs"):
            model(arr=np.array([[1.0]], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
