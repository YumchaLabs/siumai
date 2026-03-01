# Fearless Refactor V3 — TODO

Last updated: 2026-03-01

This TODO list is intentionally written as a sequence of “mergeable chunks”.

## 0) Decide names (one-time)

- [x] Choose Rust-first names for the six model families:
  - `text`, `embedding`, `image`, `rerank`, `speech`, `transcription`
- [x] Choose module layout in `siumai`:
  - `siumai::{text, embedding, image, rerank, speech, transcription}`
- [x] Choose function naming per family:
  - `text::{generate, stream}`
  - `embedding::{embed, embed_many}`
  - `image::generate`
  - `rerank::rerank`
  - `speech::synthesize`
  - `transcription::transcribe`
- [ ] Choose result type naming conventions (`*Result`, `*Output`, etc.)

## 1) Fix the foundation: decouple `LlmClient` from chat

- [x] Refactor `siumai-core::client::LlmClient` to remove `: ChatCapability`
- [x] Ensure `siumai-registry` model handles can still cache + delegate correctly
- [x] Keep downcast-style `as_*_capability` methods (or replace with new model-family traits)

## 2) Introduce model-family V3 traits and adapters

- [x] Define V3 traits for the six families (text must support generate + stream)
- [x] Provide adapters from existing provider clients to these traits
- [x] Provide adapters for registry handles to implement these traits
- [x] Add minimal unit tests (no network) for the adapters

## 3) Add the new recommended public surface in `siumai`

- [ ] Add function-style entry points:
  - [x] `text::generate`, `text::stream`, `text::stream_with_cancel`
  - [x] `embedding::embed`, `embedding::embed_many`
  - [x] `image::generate`
  - [x] `rerank::rerank`
  - [x] `speech::synthesize`
  - [x] `transcription::transcribe`
- [ ] Expose a small options struct per family (timeouts/retry/tooling/telemetry)
  - [x] Retry-only options for each family (first cut)
- [ ] Provide a compatibility facade (`siumai::compat::*`) to keep older examples building temporarily

## 4) Tools: unify definition + execution (without new crates)

- [x] Introduce a single executable tool representation:
  - JSON schema as data (already in spec)
  - async execute hook (runtime)
- [x] Provide a typed tool wrapper (optional) for apps that want it
- [ ] Provide bridging adapters from:
  - current `Tool` + `ToolResolver` pattern
  - provider-defined/hosted tool patterns

## 5) Migrate `siumai-extras` orchestrator to new APIs

- [x] Make orchestrator call the new text APIs instead of calling chat traits directly
- [x] Keep stop conditions + approvals + streaming behavior intact
- [ ] Add a small “contract suite” for tool-loop behavior (no network)

## 6) Provider-by-provider migration

- [ ] Migrate OpenAI providers first (highest usage surface)
- [ ] Migrate Anthropic + Gemini next
- [ ] Migrate remaining providers as time allows

## 7) Cleanup and deprecation

- [ ] Move application-flavored helpers out of core traits (keep them in extras/compat)
- [ ] Deprecate `Siumai::builder()` as the recommended entry (keep as convenience for provider construction)
- [ ] Update docs and examples to the new recommended surface
