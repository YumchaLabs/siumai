# FCAB-050 Review

Status: Accepted
Date: 2026-05-21
Scope: OpenAI-compatible protocol-owned completion response and streaming conversion.

## Workstream Compliance

- No blocking findings.
- The task goal is satisfied for this slice: OpenAI-compatible `/completions` response conversion
  and SSE parser state now live in protocol-owned
  `siumai-protocol-openai/src/standards/openai/compat/completion.rs`.
- The OpenAI-compatible provider runtime keeps HTTP execution, request routing, warning handling,
  and capability dispatch, but delegates protocol response and stream conversion through
  `CompletionResponseConversion` and `CompletionSseConverter`.
- The refactor aligns with the `repo-ref/ai/packages/openai-compatible` responsibility split in
  spirit: protocol conversion is separated from provider runtime execution. The Rust crate shape is
  intentionally different, but the boundary is now deep rather than a shallow module rename.
- The change stayed inside FCAB-050 scope:
  `siumai-protocol-openai`, `siumai-provider-openai-compatible`, `siumai-provider-openai` focused
  tests, source guards, and workstream docs.

## Code Quality

- No blocking findings.
- The new protocol module is a real responsibility owner, not a pass-through shim: it owns
  non-streaming completion response materialization, provider metadata attachment, provider-specific
  finish-reason mapping, usage policy application, SSE parser state, raw-chunk emission, and stream
  terminal metadata.
- Deleting the provider-owned `completion/streaming.rs` module removes the previous duplicate
  parser state home and makes the runtime adapter smaller.
- The parse-error path now emits stream-start events before an error even when raw chunks are not
  requested, matching native OpenAI completion stream behavior.
- Source guards cover both sides of the seam:
  - protocol must own `CompletionSseConverter`, parser state, response conversion, usage policy
    application, and provider finish-reason parsing;
  - provider runtime must import the protocol converter and must not reintroduce local streaming
    state, finish-reason parsing, usage policy logic, or a local streaming module.

## Missing Gates

- No missing FCAB-050 task-local gates.
- The protocol OpenAI-compatible standard module is feature-gated; focused protocol checks therefore
  require `--features openai-standard`. The all-features protocol/provider package gates also
  passed.

## Residual Risk

- FCAB-060 still needs to clean vendor Modules so promoted OpenAI-compatible vendors own only
  presets, quirks, typed options, metadata, and facade extension exports.
- Native OpenAI keeps provider-specific completion conversion intentionally. This task only moved
  OpenAI-compatible shared conversion; a native OpenAI seam task should decide whether any native
  conversion belongs elsewhere.
- Broader content directionality, provider-utils isolation, bridge adapter ownership, and facade
  taxonomy cleanup remain in later FCAB milestones.
