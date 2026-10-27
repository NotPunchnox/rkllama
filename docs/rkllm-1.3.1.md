# RKLLM 1.3.1 native runtime

The runtime binary and audio ABI definitions are inherited from upstream
[PR #180](https://github.com/NotPunchnox/rkllama/pull/180), pinned at
`c0806ffb431cebd6a4935dcaa01b1bb2e4a42fad`. They are not a second runtime
upgrade in this branch. The remaining corrections here are the initialization
`top_k` type/value, regression tests, and provenance documentation.

This branch couples the aarch64 package-local runtime with its matching ctypes
bindings. Do not update only the shared library: the added audio member changes
`RKLLMMultiModalInput` and the enclosing input union even for text inference.

- Vendor: <https://github.com/airockchip/rknn-llm>
- Release: `release-v1.3.1`
- Vendor commit: `f7390530443bf84f0394255a449d7cbe81e69d1c`
- Binary source: `rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so`
- Binary SHA256: `f25e9b099db08aaacd0a3ac62b4697d3951d6ae61ae41ea09f6702cfa89eb32c`

The initialization `top_k` field is now `int32_t`, matching the vendor header,
and initialization supplies an integer rather than a floating-point value.
The LP64 audio member occupies 48 bytes; the complete multimodal input occupies
192 bytes and `RKLLMInput` occupies 208 bytes.

Run the hardware-free regression suite with `PYTHONPATH=src python -m pytest tests`
after installing `tests/requirements.txt`. It verifies layout, field type,
initialization value, and the existing request-contract regressions. These tests
do not establish model compatibility or accelerator execution. Validate the
actual non-root service, selected native model, output semantics, and resource
containment on the target hardware before switching traffic.

This change adds the missing audio ABI definition; it does not implement an
audio API or claim audio inference support. Model quantization and conversion
quality remain independent of wrapper/runtime compatibility.
