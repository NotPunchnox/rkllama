# RKLLM generation request contract

This contract applies to RKLLM-backed `/v1/chat/completions`, `/v1/completions`,
`/api/chat`, and `/api/generate`. GGUF requests continue to be delegated to
llama.cpp before RKLLM option validation; that backend has its own capabilities.

## Request shape

Use `model` and either `messages` (chat) or `prompt` (completion). The existing
`system`, `images`, and chat `tools` inputs remain available. This change does
not change Modelfile `SYSTEM` precedence or promise tool-choice enforcement.
`stream` must be a JSON boolean. Its existing default remains false on the
OpenAI routes and true on the native routes. OpenAI `n` may only be integer `1`.

The following settings are supported per request:

- `temperature`: finite nonnegative number.
- `top_p`: finite number in `[0, 1]`.
- `top_k`: integer in `[0, 2147483647]`.
- `presence_penalty`, `frequency_penalty`: finite numbers.
- `repeat_penalty`, `mirostat_tau`, `mirostat_eta`: finite positive numbers.
- `mirostat`: integer `0`, `1`, or `2`.
- `max_new_tokens`: integer in `[1, 2147483647]`.

Numbers must fit the native float32/int32 fields; booleans and numeric strings
are rejected. These are ABI/type checks, not a guarantee that a model or runtime
will produce identical results for different sampling configurations.

For native routes, put these settings in `options`. `options.num_predict`,
`options.max_tokens`, and `options.max_completion_tokens` are aliases for
`options.max_new_tokens`. For OpenAI routes, the same `options` object is accepted
as an extension; sampling fields may instead be supplied at the top level,
including `max_tokens` or `max_completion_tokens`. Specify each effective option
only once: duplicate aliases and top-level/nested duplicates are rejected even
when their values agree. Unspecified settings retain the existing
Modelfile/server-default resolution.

Per-request `num_ctx` is rejected: it is an initialization-time setting, not a
reliable per-request override for an already-loaded worker. Set context size in
the model/server configuration instead. The output-token range above is not a
context-fit check; input truncation and total context budgeting are outside
this patch.

## Thinking control

Top-level `enable_thinking` or its `think` alias must be a JSON boolean.
An explicit false survives OpenAI conversion and reaches
`tokenizer.apply_chat_template(..., enable_thinking=False)`. Omission retains
the model's existing default. If both aliases are supplied they must agree.

This is a forwarding guarantee, not proof that every model's chat template
honors the switch. It does not strip reasoning text or globally change thinking
for other requests. Validate the actual model template separately.

## Response formatting is best effort, not constrained decoding

`response_format` accepts exactly one of:

```json
{"type": "text"}
```

```json
{"type": "json_object"}
```

`text` adds no format instruction; `json_object` maps to native `format: "json"`.
Do not send `response_format` and `format` together.

Legacy `format` accepts an empty string (no format instruction), `"json"`,
`{"type":"json"}`, or a flat object description with only `type`, `properties`,
and `required`. Each property must contain only a scalar `type` of `string`,
`integer`, `number`, or `boolean`; required names must be declared properties.
Nested schemas, enum/const constraints, additionalProperties, and other schema
keywords are rejected rather than silently ignored.

These formats use the existing prompt instructions and post-generation
validation/cleanup. They **do not constrain generation** and may return invalid
or nonconforming output. Do not use this mechanism as an enforced enum or safety
boundary. This patch does not change response cleanup or invalid-output handling.

`json_schema` (including non-strict schemas), `strict`, and grammar requests are
unsupported on this RKLLM path and receive HTTP 400. No claim is made about
capabilities of other runtime versions or serving backends.

## Errors and compatibility change

Unknown top-level fields and unsupported nested options now fail before model
loading or generation instead of being silently dropped. In particular, `seed`,
`stop`, `logit_bias`, `grammar`, `tool_choice`, `functions`, `function_call`,
`best_of`, `echo`, `logprobs`, and `stream_options` are not implemented here.
Clients relying on ignored fields must remove them or select a backend that
implements them. Invalid option values return HTTP 400, for example:

```json
{
  "error": {
    "type": "invalid_request_error",
    "param": "options.seed",
    "message": "Unsupported per-request RKLLM option"
  }
}
```

Errors name fields without echoing their supplied values. Non-object JSON bodies
also return a client error. This is not a general message-schema validator.

Example OpenAI chat request:

```json
{
  "model": "example-model",
  "messages": [{"role": "user", "content": "Return a JSON greeting."}],
  "stream": false,
  "temperature": 0,
  "max_tokens": 64,
  "enable_thinking": false,
  "response_format": {"type": "json_object"}
}
```

## Regression tests

In a dedicated test environment, install `tests/requirements.txt`, then run:

```sh
PYTHONPATH=src python -m pytest tests -q
python -m compileall -q src/rkllama/api/request_options.py
```

Tests bypass the eager API-package initializer and replace hardware worker/model
boundaries; they use the real conversion functions, Flask routes, prompt builder,
model-option merge, ctypes structures, sampling construction, and native run
argument preparation. They cover streaming and non-streaming dispatch,
sequential overrides, GGUF delegation, format translation, thinking arguments,
unsupported options, conflicting aliases, and numeric errors.

They require no model download or NPU. They do **not** establish native inference
correctness, deployed-template behavior, constrained decoding, queue/deadline
semantics, or approval-classification safety.
