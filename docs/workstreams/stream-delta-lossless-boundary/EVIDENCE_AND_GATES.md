# Stream Delta Lossless Boundary - Evidence And Gates

Status: Closed
Last updated: 2026-05-17

## Smallest Current Repro

Issue #19 reproduced OpenAI-compatible streaming output losing newline-only content deltas. The
regression class is now tracked as a stream invariant:

```bash
cargo nextest run -p siumai-protocol-openai --all-features
```

## Gate Set

### Targeted Protocol Gate

```bash
cargo nextest run -p siumai-protocol-openai --all-features
```

Proves OpenAI-compatible stream conversion and field extraction preserve generated whitespace
deltas.

Result: passed, 431 tests.

### Shared Stream Factory Gate

```bash
cargo nextest run -p siumai-core streaming::factory
```

Proves the provider-agnostic stream factory only filters true empty raw event data and forwards
whitespace-bearing JSON frames to the converter.

Result: passed, 5 tests.

### Provider Public Surface Gate

```bash
cargo nextest run -p siumai-provider-openai-compatible --all-features
```

Proves the provider crate still exposes the fixed stream semantics through its public path,
including chat stream text/reasoning deltas and completion stream text deltas.

Result: passed, 213 tests.

### Formatting Gate

```bash
cargo fmt -p siumai-core -p siumai-protocol-openai -p siumai-provider-openai-compatible
```

Result: passed. `cargo fmt --all` was attempted first but failed on Windows with OS error 206
because of workspace path length; formatting was scoped to the changed packages.

### Closeout Check

```bash
git diff --check
```

Proves the final patch has no whitespace errors before commit.

Result: passed.

## Evidence Anchors

- `docs/workstreams/stream-delta-lossless-boundary/DESIGN.md`
- `docs/workstreams/stream-delta-lossless-boundary/TODO.md`
- `siumai-core/src/streaming/factory.rs`
- `siumai-protocol-openai/src/standards/openai/compat/types.rs`
- `siumai-protocol-openai/src/standards/openai/compat/streaming.rs`
- `siumai-protocol-openai/src/standards/openai/compat/streaming_tests.rs`
- `siumai-protocol-openai/src/standards/openai/compat/transformers.rs`
- `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client.rs`

## Notes

- `c64b8980` fixed the immediate issue #19 regression.
- This workstream turns that fix into an explicit seam so future refactors do not reintroduce
  trim-based generated delta filtering.
- During all-features validation, completion streaming exposed the same boundary shape: `[DONE]`
  was not converter-owned and empty `choices[].text` deltas were not field-presence based. That
  was fixed in this lane.
