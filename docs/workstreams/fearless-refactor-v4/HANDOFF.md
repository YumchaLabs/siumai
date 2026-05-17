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
- Existing capability modules remain execution owners:
  - `chat`
  - `completion`
  - `embedding`
  - `image`
  - `audio`
  - `rerank`
  - `models`

## Next Step

Finish validation for the client shell split, commit it, then reassess whether any remaining
large OpenAI-compatible runtime modules have real ownership/coupling problems. Do not split by
file size alone.

## Guardrails

- Keep public paths and behavior compatible.
- Keep `LlmClient` as a compatibility bridge, not the architectural center.
- Keep capability execution logic in capability modules.
- Keep provider catalog data in provider-family catalog modules.
- Prefer focused no-network tests and source guards over broad rewrites.

## Validation Commands

- `cargo fmt -p siumai-provider-openai-compatible`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast openai_client`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
- `git diff --check`
