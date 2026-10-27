# Experimental native prefix cache

This feature builds on the native API correctness changes. It does not add an
output-validation/retry adapter, strict JSON Schema generation, or forced tool
selection. Unsupported generation contracts remain explicit request errors.

## Enablement

Automatic prefix reuse is **off by default**. Set
`RKLLAMA_EXPERIMENTAL_PREFIX_CACHE=1` only for a worker/model combination that
has passed target-hardware correctness and isolation tests. Unset it and reload
the worker to return to the ordinary uncached path.

The implementation uses the RKLLM 1.3.1 native prompt-cache API. It reuses exact
token prefixes, not semantically similar text or message-count identities. It
can reduce repeated prefill work; it does not accelerate decoding or improve
answer quality. A cold request may incur an extra priming pass.

## Ownership and lifecycle

- Each worker owns one temporary snapshot under its configured temp directory.
  Clients cannot provide a cache filename or conversation identity.
- Inputs of at least 256 tokens prime all but the final 32 tokens. Callback
  output is suppressed during priming, then reset before visible generation.
- A hit requires the saved prefix to be strictly shorter than, and exactly
  match, the new token input. After loading, the complete input is submitted;
  native conversation history remains disabled.
- Loaded snapshots are released and native KV state cleared before each request.
  A changed prefix replaces the owned snapshot. Routine post-request cleanup
  preserves the disk snapshot; explicit clear, abort and unload invalidate it.
- Unsupported saving or a snapshot exceeding 1 GiB disables caching for that
  handle instead of repeatedly paying priming costs. Failed cleanup taints the
  worker, which exits rather than accepting another request.
- Cancellation prevents later priming/load/generation stages. Cached native
  prefill may not stop immediately: abort is retried with a bounded join of up
  to 300 seconds when caching is enabled, versus 30 seconds without it. An
  unquiescent worker exits without waiting indefinitely on native threads.
- Normal unload removes the snapshot directory. Forced process/host termination
  can leave an orphan; cache snapshots are temporary, not backup artifacts.

## Accounting

Usage includes native counters for hidden priming plus visible generation.
Per-request cache accounting is reset on both cached and bypass paths. Cache-hit
prefill counters reflect newly processed tokens rather than the complete client
transcript; these are native performance counters, not billing-grade context
counts. Visible generation, not priming output, determines the output-limit
finish reason.

## Verification and review boundary

```sh
python -m pip install -r tests/requirements.txt
PYTHONPATH=src python -m pytest tests -q -W error
```

The hardware-free suite covers prefix identity, replacement, suppression,
failed/oversize saves, stale accounting, cancellation transitions, ordinary
cleanup versus explicit invalidation, and bounded native-thread shutdown.
It does not execute the NPU. Validate the actual non-root service, repeated
prefixes, changed prefixes, independent requests, cancellation/recovery and
resource limits on target hardware before enabling the feature.

For a separate cache-only review, use `fix/native-api-correctness` as the base
for `feat/native-prefix-cache`. If the correctness changes are merged first,
rebase this branch onto that merged base before opening its upstream PR.
