# Fearless Refactor V4 - Handoff

Last updated: 2026-05-17

## Current State

The workstream remains active. The public V4 story is already mostly aligned around
family-model execution, registry-first app code, config-first provider code, and explicit
compatibility surfaces.

The current continuation lane is OpenAI-compatible internal boundary cleanup:

- Provider model catalog split is complete and committed.
- `OpenAiCompatibleClient` production shell has been split into focused submodules:
  - `openai_client/types.rs`
  - `openai_client/runtime.rs`
  - `openai_client/compatibility.rs`
- The parent OpenAI-compatible client regression suite has been moved to `openai_client/tests.rs`,
  leaving `openai_client.rs` as a thin shell plus `mod tests;`.
- The completion runtime is now a directory module:
  - `openai_client/completion/mod.rs` owns completion request/response orchestration.
  - `openai_client/completion/streaming.rs` owns the SSE stream state and converter.
  - `openai_client/completion/tests.rs` owns completion no-network contracts and source guards.
- Existing capability modules remain execution owners:
  - `chat`
  - `completion`
  - `embedding`
  - `image`
  - `audio`
  - `rerank`
  - `models`

## Next Step

Reassess whether `audio`, `builder`, `config`, or `settings` have real ownership/coupling problems.
Do not split by file size alone; prefer the next cut only when it improves locality or narrows a
real interface.

## Guardrails

- Keep public paths and behavior compatible.
- Keep `LlmClient` as a compatibility bridge, not the architectural center.
- Keep capability execution logic in capability modules.
- Keep provider catalog data in provider-family catalog modules.
- Keep completion SSE state and event conversion in `openai_client/completion/streaming.rs`.
- Prefer focused no-network tests and source guards over broad rewrites.

## Validation Commands

- `cargo fmt -p siumai-provider-openai-compatible`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast completion`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast openai_client`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
- `git diff --check`
