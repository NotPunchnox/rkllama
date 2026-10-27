# Native RKLLM API compatibility

This change targets the native RKLLM path, not requests forwarded to llama.cpp.
It does not select a production backend, change weights, or enlarge resource limits.
It builds on upstream [PR #180](https://github.com/NotPunchnox/rkllama/pull/180),
pinned at `c0806ffb431cebd6a4935dcaa01b1bb2e4a42fad`, for the RKLLM 1.3.1
runtime and audio ABI definitions. This branch adds the request-option contract,
the initialization `top_k` type/value correction, and ABI regression tests;
see `request-options.md` and `rkllm-1.3.1.md`. Review against that prerequisite
commit while it is unmerged, then rebase onto upstream `main` after it merges.
Optional prefix caching is documented in `native-prefix-cache.md`.
Post-generation validation/retry is not included.

## Request and response contract

- Text chat/generation requests reserve their requested output tokens against the
  configured context limit, using tokenized input **after** template and format
  instructions. Overflow returns HTTP 400 with `context_length_exceeded` before
  inference starts, including for streaming requests. Multimodal capacity is not
  estimated from text alone; that path remains outside this preflight guarantee.
- OpenAI chat and legacy completions accept `stream_options.include_usage` as a
  boolean, only with `stream: true`. Requested usage comes from native callback
  counters, once in a final empty-choices event before `[DONE]`. Other events have
  null usage; unrequested usage is omitted. A stream uses one ID and creation time.
- Tool choice supports `auto` and `none`. Automatic calls remain model-generated,
  not guaranteed. `none` removes tool definitions from the prompt. Forced/named
  selection and strict tool schemas remain explicit HTTP 400 errors.
- OpenAI tool arguments are JSON strings, including in streamed responses, and
  conversion does not mutate the underlying Ollama response.
- Qwen3.5 function/parameter blocks and older JSON calls share declared-name
  and object-argument validation. Malformed or incomplete framed output is not
  salvaged through a generic JSON fallback. A mixture of valid and invalid calls
  returns no callable list. Supported tool schemas use scalar type names,
  properties, required, boolean additionalProperties, items, and enum; union types
  and unsupported schema keywords are rejected before inference.
- Tool-enabled streams buffer until the complete response is parsed. This adds
  latency for ordinary text when tools are supplied, but avoids leaking partial
  tool markup as content. Tools are returned to the caller, never executed here.
- Strict JSON Schema/constrained decoding remains unsupported, not approximated
  by accepting an ignored option or silently retrying unvalidated generations.

## Inference isolation

The worker clears callback text, metrics, embedding, and partial UTF-8 buffers
before inference, and drains tokens arriving during its final thread join before
sending completion. Previously those last tokens could be missing from one
answer and appear at the start of the next request.

Abort joins the inference thread before clearing/resetting shared state. If the
thread cannot be stopped within the bounded join, the worker exits rather than
accepting overlapping inference. Native errors and timeouts are errors, not text
answers or successful zero-usage completion events. Token-limit exhaustion is
conservatively reported as `length` from generation callbacks as well as native
performance counters: the SDK's performance count can be one lower than the
normal generation-callback count.
Usage remains the SDK-reported counters; it is not silently inflated to match
callback counts. The SDK does not expose a separate finish-reason field here.
Capacity checks use the context size reported by the loaded native worker, not
an edited Modelfile that has not yet been reloaded.

The legacy message-count-based cache selector remains disabled. Exact-prefix
reuse is experimental, explicitly opt-in and off by default; see
`native-prefix-cache.md` for ownership, invalidation, accounting and cancellation
limits. No client-controlled cache path or conversation identity is accepted.

## Unit and contract verification

```sh
python -m pip install -r tests/requirements.txt
PYTHONPATH=src python -m pytest tests -q
```

These tests stub hardware dependencies. Real target-hardware inference must be
reported separately; passing unit tests is not an RKLLM quality benchmark.
