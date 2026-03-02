# Fearless Refactor V3 — TODO

Last updated: 2026-03-02

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
- [x] Choose construction story (beta.6):
  - config-first constructors on provider clients (recommended)
  - builder-style as compat only (time-bounded)
  - registry-first as the default path in docs/examples

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
- [x] Provide a compatibility facade (`siumai::compat::*`) to keep older examples building temporarily

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

## 6.5) Construction ergonomics (remove global builder dependency)

Goal: new code should not require `Siumai::builder()` / `Provider::*()`.

- [ ] Add `from_config(...)` (or equivalent) constructor for core providers:
  - [x] `OpenAiClient`
  - [x] `AnthropicClient`
  - [x] `GeminiClient`
  - [x] `AzureOpenAiClient`
  - [x] `OllamaClient`
  - [x] `GroqClient`
  - [x] `MinimaxiClient`
  - [x] `GoogleVertexClient`
  - [x] `VertexAnthropicClient`
- [ ] Ensure constructors build HTTP client/interceptors/middlewares from `*_Config` + `HttpConfig`
- [x] Fix provider `LlmClient` wiring regressions (e.g. `VertexAnthropicClient` must expose `as_chat_capability` and correct `provider_id`)
- [x] Update key docs/examples to use config-first construction
- [ ] Keep builder path under `compat` (document removal target)
- [x] Add a config-first shortcut for OpenAI-compatible vendors (built-in registry)

## 7) Cleanup and deprecation

- [ ] Move application-flavored helpers out of core traits (keep them in extras/compat)
- [ ] Deprecate `Siumai::builder()` as the recommended entry (keep as convenience for provider construction)
- [ ] Update docs and examples to the new recommended surface
  - [x] Add `siumai::compat` as an explicit legacy surface
  - [x] Migrate README + key examples to family APIs
  - [x] Migrate `examples/02-core-api/*` to registry construction
  - [x] Migrate `examples/07-applications/*` to registry construction
