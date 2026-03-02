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
- [x] Choose result type naming conventions (`*Result`, `*Output`, etc.)
  - Convention: `*Response` for outputs, `*Stream` for streams, `*Options` for call options.
  - Note: where a spec type already exists (e.g. `ChatResponse`), we keep it.
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

- [x] Add function-style entry points:
  - [x] `text::generate`, `text::stream`, `text::stream_with_cancel`
  - [x] `embedding::embed`, `embedding::embed_many`
  - [x] `image::generate`
  - [x] `rerank::rerank`
  - [x] `speech::synthesize`
  - [x] `transcription::transcribe`
- [x] Provide a compatibility facade (`siumai::compat::*`) to keep older examples building temporarily
- [x] Expose a small options struct per family (timeouts/retry/tooling/telemetry)
  - [x] Retry-only options for each family (first cut)
  - [x] Per-request HTTP overrides: `timeout`, `headers`
  - [x] Text tooling surface: `tools`, `tool_choice`
  - [x] Telemetry override per request (text)

## 4) Tools: unify definition + execution (without new crates)

- [x] Introduce a single executable tool representation:
  - JSON schema as data (already in spec)
  - async execute hook (runtime)
- [x] Provide a typed tool wrapper (optional) for apps that want it
- [x] Provide bridging adapters from:
  - current `Tool` + `ToolResolver` pattern
  - provider-defined/hosted tool patterns

## 5) Migrate `siumai-extras` orchestrator to new APIs

- [x] Make orchestrator call the new text APIs instead of calling chat traits directly
- [x] Keep stop conditions + approvals + streaming behavior intact
- [x] Add a small “contract suite” for tool-loop behavior (no network)

## 6) Provider-by-provider migration

- [x] Migrate OpenAI providers first (highest usage surface)
- [x] Migrate Anthropic + Gemini next
- [x] Migrate remaining providers as time allows

## 6.5) Construction ergonomics (remove global builder dependency)

Goal: new code should not require `Siumai::builder()` / `Provider::*()`.

- [x] Add `from_config(...)` (or equivalent) constructor for core providers:
  - [x] `OpenAiClient`
  - [x] `AnthropicClient`
  - [x] `GeminiClient`
  - [x] `AzureOpenAiClient`
  - [x] `OllamaClient`
  - [x] `GroqClient`
  - [x] `MinimaxiClient`
  - [x] `GoogleVertexClient`
  - [x] `VertexAnthropicClient`
- [x] Ensure constructors build HTTP client/transport from `*_Config` + `HttpConfig`
- [x] Add config-driven wiring for interceptors/middlewares
- [x] Fix provider `LlmClient` wiring regressions (e.g. `VertexAnthropicClient` must expose `as_chat_capability` and correct `provider_id`)
- [x] Update key docs/examples to use config-first construction
- [x] Keep builder path as compat-only and document removal target
- [x] Add a config-first shortcut for OpenAI-compatible vendors (built-in registry)

## 7) Cleanup and deprecation

- [x] Move application-flavored helpers out of core traits (keep them in extras/compat)
- [x] Deprecate `Siumai::builder()` as the recommended entry (keep as convenience for provider construction)
- [x] Update docs and examples to the new recommended surface
  - [x] Add `siumai::compat` as an explicit legacy surface
  - [x] Migrate README + key examples to family APIs
  - [x] Migrate `examples/02-core-api/*` to registry construction
  - [x] Migrate `examples/03-advanced-features/*` to registry/config-first construction
  - [x] Migrate `examples/05-integrations/telemetry/*` to registry construction
  - [x] Migrate `examples/07-applications/*` to registry construction
  - [x] Migrate `examples/04-provider-specific/openai/*` to registry/config-first construction
  - [x] Migrate `examples/04-provider-specific/anthropic/*` to registry/config-first construction
  - [x] Migrate `examples/04-provider-specific/google/*` to registry/config-first construction
  - [x] Migrate `examples/04-provider-specific/ollama/*` to config-first construction
  - [x] Migrate `examples/04-provider-specific/minimaxi/*` to config-first construction
  - [x] Update `examples/04-provider-specific/openai-compatible/README.md` to avoid builder in troubleshooting snippets
  - [x] Keep a single explicit builder example as compat (`moonshot-siumai-builder.rs`)
  - [x] Remove `Siumai::builder()` from stubbed MCP snippets (use registry handle in guidance blocks)
