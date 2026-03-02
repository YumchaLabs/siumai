# Fearless Refactor V3 — Design

Last updated: 2026-03-02

## Context

This workstream corresponds to the `v0.11.0-beta.6` refactor line: **Rust-first, model-family APIs**
with provider extensions, inspired by (but not copying) the Vercel AI SDK V3 provider abstraction.

The `siumai` workspace already completed a medium-granularity split:

- `siumai-spec`: provider-agnostic types/tools/errors
- `siumai-core`: runtime (HTTP/streaming/retry/middleware)
- `siumai-registry`: routing + factories + model handles (with cache/TTL)
- `siumai-extras`: orchestrator, MCP, server adapters, structured output helpers

This V3 workstream focuses on **API shape + model interface foundations**, not on further
monorepo/package fragmentation.

## Goals

1. **Establish versioned model interfaces as the primary foundation**
   - A stable, explicit “model object” interface is the architectural anchor.
   - The six model families stay first-class: text (language), embedding, image, rerank, speech, transcription.

2. **Provide a single recommended entry style**
   - Function-style call sites (Rust-friendly naming) that accept a model instance/handle.
   - Registry returns family-specific model handles that are directly callable.

3. **Make construction builder-optional (config-first)**
   - Users can construct provider clients and registry handles without calling `Siumai::builder()`.
   - Builder-style construction can remain as a *compatibility convenience*, but is not the recommended entry.

3. **Unify tools without creating many “tool packages”**
   - Tools remain one cohesive system: definition + schema + execution binding.
   - Advanced orchestration remains in `siumai-extras`.

4. **Make internal layering enforceable**
   - Remove coupling that makes “text/chat” the base of everything.
   - Ensure other families can evolve without being forced through chat-centric traits.

## Non-goals

- Do not copy Vercel AI SDK naming verbatim.
- Do not introduce a large number of new crates for tools/UI/adapters.
- Do not attempt to unify every provider’s “non-family resources” into the stable surface.
  Those remain provider extensions.

## Key design decisions

## Public API naming (final)

This workstream intentionally uses Rust-first naming and module layout.
We do **not** mirror Vercel AI SDK naming.

### Top-level modules in `siumai`

The stable, recommended surface is grouped by model family:

- `siumai::text`
- `siumai::embedding`
- `siumai::image`
- `siumai::rerank`
- `siumai::speech`
- `siumai::transcription`

Compatibility shims live under:

- `siumai::compat` (temporary, time-bounded)
- `siumai::provider_ext::<provider>` (provider-specific typed extensions; long-term)

### Function naming (final)

- `siumai::text::generate`
- `siumai::text::stream`
- `siumai::embedding::embed`
- `siumai::embedding::embed_many`
- `siumai::image::generate`
- `siumai::rerank::rerank`
- `siumai::speech::synthesize`
- `siumai::transcription::transcribe`

### Request/response type naming (guideline)

- Prefer `TextRequest`, `TextResponse`, `TextStream` for the text family.
- Prefer `EmbeddingRequest`, `EmbeddingResponse` (already present in spec).
- Prefer `ImageGenerationRequest`, `ImageGenerationResponse` (already present in spec).
- Prefer `RerankRequest`, `RerankResponse` (already present in spec).
- Prefer `TtsRequest`, `TtsResponse` (already present in spec).
- Prefer `SttRequest`, `SttResponse` (already present in spec).

If a spec type already exists and matches the semantics, we do not invent a new alias.

### D1 — Stop making `LlmClient` “chat-first”

Current issue: `LlmClient: ChatCapability` makes “text chat” the implicit base.
This complicates model-family separation and pushes unrelated features through chat plumbing.

Decision:

- Make `LlmClient` a **metadata + capability downcast** trait (no inheritance from `ChatCapability`).
- Each model family is expressed via explicit family traits (see D2).

### D2 — Introduce V3 model-family traits (within existing crates)

We will represent model families via stable traits (names TBD; examples below):

- `TextModelV3` (generate + stream)
- `EmbeddingModelV3`
- `ImageModelV3`
- `RerankModelV3`
- `SpeechModelV3`
- `TranscriptionModelV3`

These traits live in `siumai-core` (or `siumai-spec` only if strictly type-level and dependency-light).
They are intentionally versioned to enable future evolution without breaking everything.

### D3 — A Rust-first public API shape (not Vercel naming)

We will expose a small, consistent function API in `siumai` as the recommended surface.
Illustrative shape (names match the final naming section):

- `siumai::text::generate(&model, request, options) -> TextResponse`
- `siumai::text::stream(&model, request, options) -> TextStream`
- `siumai::embedding::embed(&model, request, options) -> EmbeddingResponse`
- `siumai::embedding::embed_many(&model, requests, options) -> BatchEmbeddingResponse`
- `siumai::image::generate(&model, request, options) -> ImageGenerationResponse`

Compatibility:

- Existing `Siumai::builder()` and method-style calls remain temporarily behind `compat` modules.
- `siumai-extras` continues to provide orchestrator APIs; it will call the new `siumai::text::*` APIs.

### D4 — Tools: bind schema + execution, keep type-safety optional

We keep a single tool system (no extra crates):

- **Core tool definition**: name/description/JSON schema + async execute
- **Typed wrapper** (optional): `Tool<TArgs, TOut>` where `TArgs/TOut: serde` are converted to/from JSON

This supports:

- Simple apps: a `Vec<Tool>` that “just works”
- Larger apps: typed tool sets with compile-time guarantees

Orchestrator and approval workflows remain in `siumai-extras`.

### D5 — Construction vs invocation (ergonomics)

We keep builders where Rust benefits from them (configuration), but we do not require
builder-style APIs for invocation.

- **Construction** (recommended):
  - Provider configs/clients (e.g. `providers::openai::{OpenAiConfig, OpenAiClient}`) configure credentials/transport/provider defaults.
  - Registry handles (e.g. `registry.language_model("openai:gpt-4o")`) select models and apply middleware/caching.

- **Invocation** (recommended):
  - Function-style family APIs (`siumai::text::*`, `siumai::embedding::*`, ...).
  - The same invocation APIs accept both provider-built models and registry handles.

- **Compatibility** (temporary):
  - `Siumai::builder()` exists under `siumai::compat` and is not the recommended entry.

### D6 — Provider-specific features without breaking the unified surface

Provider-specific capabilities (e.g. Anthropic prompt caching, OpenAI Responses vs Chat Completions)
are handled via three mechanisms, in order of preference:

1. **Provider-agnostic hints** expressed as data on requests/messages/metadata (best-effort).
2. **Open provider options map** (`provider_options_map`) for forward-compatible extensions.
3. **Provider extensions** in `siumai::provider_ext::<provider>` providing typed helpers that populate
   `provider_options_map` and typed accessors for provider metadata in responses/streams.

Unified calls remain stable; provider-specific behavior is opt-in and does not pollute the family APIs.

### D7 — “No `Siumai::builder()` required” (Rust ergonomics)

We will treat `Siumai::builder()` / `Provider::*()` builders as **compatibility conveniences**, not the primary entry.
The recommended construction style is **config-first** and provider-owned:

```rust,no_run
use siumai::prelude::*;
use siumai::providers::openai::{OpenAiClient, OpenAiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Construction (config-first)
    let cfg = OpenAiConfig::new("OPENAI_API_KEY")
        .with_model("gpt-4o-mini")
        .with_temperature(0.7);

    // (Planned) one-shot constructor that builds HTTP internals from config.
    // The exact name is up to us; the key is: no global builder required.
    let client = OpenAiClient::from_config(cfg)?;

    // Invocation (family API)
    let req = ChatRequest::new(vec![user!("hello")]);
    let resp = siumai::text::generate(&client, req, siumai::text::GenerateOptions::default()).await?;
    println!("{}", resp.content_text().unwrap_or_default());
    Ok(())
}
```

Notes:

- Provider-specific resources remain available on the provider client type (e.g. OpenAI Files/Moderation/Responses admin).
- Advanced provider-only request knobs are attached via `provider_options_map` using typed extension traits:
  - `providers::openai::OpenAiChatRequestExt::with_openai_options(...)`
  - `providers::anthropic::AnthropicChatRequestExt::with_anthropic_options(...)`

## Target layering (minimal crate changes)

1. `siumai-spec`
   - Types (requests/responses/messages/stream parts) and core error types
   - Tool schema structs (JSON Schema as data)

2. `siumai-core`
   - Model-family V3 traits + adapters to provider clients
   - Execution building blocks: streaming, retry, middleware
   - Tool execution plumbing (but not high-level multi-step orchestration)

3. `siumai-registry`
   - Registry API returns family model handles that implement the V3 traits
   - Caching/TTL lives here

4. `siumai-extras`
   - Orchestrator, MCP, Axum adapters, advanced structured output

## Migration plan (staged, but fearless)

1. **Foundation**
   - Refactor `LlmClient` to decouple from chat.
   - Introduce V3 family traits + minimal adapters.

2. **New public surface**
   - Add `siumai::text::*` / `siumai::embed::*` / etc. entry points.
   - Keep old entry points under `compat` and migrate internal usages.
   - Introduce small per-call options per family (timeout/headers + text tooling/telemetry).

3. **Tools unification**
   - Introduce executable tool definition (schema + execute).
   - Provide adapters from the existing `Tool` + `ToolResolver` style.

4. **Provider migration**
   - Migrate core providers first (OpenAI/Anthropic/Gemini) to the new family traits.
   - Expand to other providers and keep feature gating consistent.

5. **Construction cleanup (beta.6)**
   - Add config-first constructors (`*_Client::from_config(...)`) for core providers.
   - Update README and key examples to avoid `Siumai::builder()` for new code.
   - Keep builder path under `compat` and time-bound it (removal target is a future beta).
   - Allow config-driven wiring for interceptors/middlewares (runtime-only; not serialized).

## Risks & mitigations

- **Public breaking changes**: mitigate by keeping `compat` shims for one minor release.
- **Trait explosion**: keep six families only; everything else stays in provider extensions.
- **Compile-time cost**: keep generics optional; default path uses trait objects + `Arc`.
