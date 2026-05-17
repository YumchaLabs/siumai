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
- Repeated simple provider settings adapters now share `simple_compat_provider_settings!`; only
  generic OpenAI-compatible settings, Vertex MaaS settings, and Alibaba settings stay hand-written
  because they own different construction behavior.
- Provider family default-model data has been split into `config/family_defaults.rs`; `config.rs`
  remains the compatibility lookup facade used by runtime/default-model callers.
- Built-in provider registry data has been split into `config/builtin_providers.rs`; `config.rs`
  still owns generic provider config construction and lookup/capability facade functions.
- Existing capability modules remain execution owners:
  - `chat`
  - `completion`
  - `embedding`
  - `image`
  - `audio`
  - `rerank`
  - `models`

## Next Step

Reassess whether any remaining OpenAI-compatible internal modules have real ownership/coupling
problems. The config facade is now thin enough that further config splitting should happen only if
new provider families need dedicated registry ownership.

## Guardrails

- Keep public paths and behavior compatible.
- Keep `LlmClient` as a compatibility bridge, not the architectural center.
- Keep capability execution logic in capability modules.
- Keep provider catalog data in provider-family catalog modules.
- Keep completion SSE state and event conversion in `openai_client/completion/streaming.rs`.
- Keep simple provider settings adapters on the macro path unless the provider has real divergent
  construction behavior.
- Keep provider family defaults in `config/family_defaults.rs`; `config.rs` should remain a lookup
  facade rather than a mixed data owner.
- Keep built-in provider registry data in `config/builtin_providers.rs`; `config.rs` should not own
  `build_builtin_providers` or the static built-in provider map.
- Prefer focused no-network tests and source guards over broad rewrites.

## Validation Commands

- `cargo fmt -p siumai-provider-openai-compatible`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast completion`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast settings`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast config`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast openai_client`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
- `git diff --check`
