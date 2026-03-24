# Migration Guide: `0.11.0-beta.5` → `0.11.0-beta.6` (Fearless Refactor V3 — Family APIs)

This guide focuses on the **recommended surface change** introduced in `0.11.0-beta.6`.
Most existing code should continue to compile, but new code should prefer the Rust-first
model-family APIs.

## TL;DR

- Construction (recommended):
  - registry: `registry::global().language_model("openai:gpt-4o-mini")?`
  - provider config-first: `OpenAiClient::from_config(OpenAiConfig::new(api_key).with_model("..."))?`
  - provider config-first (example): `MinimaxiClient::from_config(MinimaxiConfig::new(api_key))?`
- Builders remain available as compatibility conveniences:
  - unified builder: `Siumai::builder()...build().await?` (deprecated in `0.11.0-beta.6`, but still supported)
  - provider builders: `Provider::openai()...build().await?`
- Invocation is now recommended to go through model-family functions:
  - `siumai::text::{generate, stream, stream_with_cancel}`
  - `siumai::embedding::{embed, embed_many}`
  - `siumai::image::generate`
  - `siumai::rerank::rerank`
  - `siumai::speech::synthesize`
  - `siumai::transcription::transcribe`
- Legacy method-style entry points are treated as compatibility surface:
  - explicit module: `siumai::compat`
- If you hold a registry handle or a dynamic client, you can still call
  `client.chat_request(req).await?` / `client.chat_stream_request(req).await?` as convenience methods.

- OpenAI-compatible vendors (DeepSeek/OpenRouter/Moonshot/etc.) now have a config-first shortcut:
  - `siumai::providers::openai_compatible::OpenAiCompatibleClient::from_builtin_env(...).await?`

`from_builtin_env` reads API keys from env using this precedence:
1) `ProviderConfig.api_key_env` (when present)
2) `ProviderConfig.api_key_env_aliases` (fallbacks)
3) `${PROVIDER_ID}_API_KEY` (uppercased, `-` replaced with `_`)

To discover built-in OpenAI-compatible provider ids, call
`siumai::providers::openai_compatible::list_provider_ids()`.

Note: `registry::global().language_model("provider:model")` returns a
`LanguageModelHandle`, not a raw `Arc<dyn LlmClient>`. In practice this usually does not matter:
the handle is the recommended unified entry point for `text::*` and also supports
`chat_request(...)` / `chat_stream_request(...)` convenience calls.

## Before/after cheatsheet

The most common migration is “keep construction, swap invocation”:

- `client.chat(messages)` → `text::generate(&client, ChatRequest::new(messages), ..)`
- `client.chat_stream(messages, tools)` → `text::stream(&client, ChatRequest::new(messages).with_tools(tools), ..)`
- `client.chat_stream_with_cancel(messages, tools)` → `text::stream_with_cancel(&client, ChatRequest::new(messages).with_tools(tools), ..)`
- `client.embed(...)` / `client.embed_with_config(..)` → `embedding::embed(&client, EmbeddingRequest { .. }, ..)`
- `client.generate_images(..)` → `image::generate(&client, ImageGenerationRequest { .. }, ..)`
- `client.rerank(..)` → `rerank::rerank(&client, RerankRequest::new(..), ..)`
- `client.tts(..)` → `speech::synthesize(&client, TtsRequest::new(..), ..)`
- `client.stt(..)` → `transcription::transcribe(&client, SttRequest::from_audio(..), ..)`

## 1) Text generation: `client.chat_*` → `siumai::text::*`

`text::*` is the new recommended surface, but request building usually does **not** need to change:
today `TextRequest` is still an alias of `ChatRequest`.

### Non-streaming

```rust,ignore
use siumai::prelude::unified::*;
use siumai::providers::openai::{OpenAiClient, OpenAiConfig};

let cfg = OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)
    .with_model("gpt-4o-mini");
let client = OpenAiClient::from_config(cfg)?;

let req = ChatRequest::new(vec![user!("hi")]);
let resp = text::generate(&client, req, text::GenerateOptions::default()).await?;
```

### Streaming

```rust,ignore
use futures::StreamExt;
use siumai::prelude::unified::*;

let req = ChatRequest::new(vec![user!("stream a poem")]);
let mut stream = text::stream(&client, req, text::StreamOptions::default()).await?;
while let Some(ev) = stream.next().await {
    if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = ev {
        print!("{delta}");
    }
}
```

### Streaming cancellation

```rust,ignore
use futures::StreamExt;
use siumai::prelude::unified::*;

let req = ChatRequest::new(vec![user!("stream...")]);
let handle = text::stream_with_cancel(&client, req, text::StreamOptions::default()).await?;
let ChatStreamHandle { mut stream, cancel } = handle;

let reader = tokio::spawn(async move { while stream.next().await.is_some() {} });
cancel.cancel();
let _ = reader.await;
```

## 1b) Other model families (quick examples)

```rust,ignore
use siumai::prelude::unified::*;

// Embeddings
let _ = embedding::embed(
    &client,
    EmbeddingRequest {
        input: vec!["hello".to_string()],
        ..Default::default()
    },
    embedding::EmbedOptions::default(),
)
.await?;

// Rerank
let _ = rerank::rerank(
    &client,
    RerankRequest::new("rerank-model".into(), "q".into(), vec!["a".into(), "b".into()]),
    rerank::RerankOptions::default(),
)
.await?;
```

## 2) Provider-specific features

Provider-specific knobs remain opt-in and do not pollute the family APIs.
Keep using one of:

- provider-agnostic open map: `request.with_provider_option("<provider_id>", json!(...))`
- typed helpers (when available): `siumai::provider_ext::<provider>::options::*`
- provider response metadata: `response.provider_metadata` (or typed accessors under `provider_ext`)

## 3) Compatibility surface

If you have older code that relies on method-style APIs, keep it as-is for now.
For explicitness (and to prepare for future deprecation), choose the import style that matches
what you still use:

```rust,ignore
// If you only need `Siumai` / `SiumaiBuilder` compatibility imports:
use siumai::compat::*;

// If you also still use `Provider::<provider>()` builders:
use siumai::prelude::compat::*;
```

Compatibility note: the `siumai::compat` module is intended to be temporary.
Planned removal target is **no earlier than `0.12.0`**.

## 4) New per-call options (timeouts/headers/tools/telemetry)

In `0.11.0-beta.6`, each model family exposes a small `*Options` struct.
This started as "retry-only" in earlier drafts, but now includes **per-call HTTP overrides**
and (for text) tool/telemetry overrides.

### 4.1 Per-call timeout and headers

All families support:

- `timeout`: sets a per-request timeout (applied at the HTTP layer, including streaming requests)
- `headers`: merges extra headers into the request (per call)

Example (text):

```rust,ignore
use siumai::prelude::unified::*;
use std::time::Duration;

let req = ChatRequest::new(vec![user!("hi")]);
let resp = text::generate(
    &client,
    req,
    text::GenerateOptions {
        timeout: Some(Duration::from_secs(30)),
        headers: [("x-trace-id".to_string(), "abc".to_string())]
            .into_iter()
            .collect(),
        ..Default::default()
    },
)
.await?;
```

### 4.2 Text: tools and tool choice per call

Text options also support:

- `tools`: append tools for this call
- `tool_choice`: override tool choice for this call

If you already set tools on the `ChatRequest`, options-level tools are appended.

### 4.3 Text: telemetry override per call

Text options also support:

- `telemetry`: set `ChatRequest.telemetry` (runtime-only; not serialized)

This is useful when you want to override tracing/metrics sampling or attach per-request tags.

## 5) Config-driven wiring for interceptors/middlewares

Config-first constructors now support wiring interceptors directly from `*_Config` types, and
many provider configs also support model middlewares there (runtime-only; not serialized).

Important: this is **not yet perfectly uniform across every provider config type**. If a specific
provider config does not expose `with_model_middlewares(...)`, use the provider builder or attach
middleware through the higher-level registry/builder path for that provider.

This lets you keep construction simple while still enabling cross-cutting concerns like:

- request/response logging
- custom auth headers
- model parameter normalization
- policy enforcement

Example (OpenAI):

```rust,ignore
use siumai::providers::openai::{OpenAiClient, OpenAiConfig};
use std::sync::Arc;

let cfg = OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)
    .with_model("gpt-4o-mini")
    .with_http_interceptors(vec![/* Arc<dyn HttpInterceptor> */])
    .with_model_middlewares(vec![/* Arc<dyn LanguageModelMiddleware> */]);

let client = OpenAiClient::from_config(cfg)?;
```

## 6) Additional migration notes vs `0.11.0-beta.5`

These are not all required source changes, but they are the main upgrade points that were easy to
miss when moving from `beta.5` to `beta.6`:

- `Siumai::builder()` is now explicitly `#[deprecated]`.
  - Existing code can keep using it for now.
  - New code should prefer `registry::global().language_model("provider:model")` or
    `*Client::from_config(...)`.
- Builder-style provider construction is still available, but it now clearly lives on the
  compatibility path.
  - If you still use provider builders such as `Provider::openai()`, prefer importing
    `siumai::prelude::compat::*` in migration-oriented code.
- OpenAI-compatible shortcut constructors are async.
  - `OpenAiCompatibleClient::from_config(...)`
  - `OpenAiCompatibleClient::from_builtin(...)`
  - `OpenAiCompatibleClient::from_builtin_env(...)`
- The new family API is primarily an invocation migration, not a request-shape migration.
  - In most cases you can keep building `ChatRequest` exactly as before and only change the call
    site from `client.chat_*` to `text::*`.
- If you need provider-specific features that are outside the six stable model families, keep using
  `siumai::provider_ext::<provider>::*` (or `siumai::providers::<provider>::*`) and
  `siumai::prelude::extensions::*`.
- If you are building protocol gateways or cross-provider proxies, `beta.6` also adds an
  experimental bridge surface:
  - `siumai::experimental::bridge`
  - `siumai-extras::server::axum`
  This is additive and does not require migration for normal model calls.
- If you work with OpenAI Responses WebSocket streaming, prefer the dedicated provider-specific
  helpers under `siumai::providers::openai::*`:
  - `OpenAiWebSocketSession`
  - `OpenAiIncrementalWebSocketSession`
  - `OpenAiWebSocketRecoveryConfig`
- Gemini and Vertex embedding flows now expose stronger public batch helper behavior.
  - This is usually a free upgrade rather than a source migration, but it is part of the
    `beta.5` → `beta.6` behavioral delta.

## Checklist

- Keep construction code unchanged.
- Prefer `siumai::text::*` for chat-style inference.
- Use family `*Options` for per-call timeout/headers (and text tooling/telemetry) instead of ad-hoc request mutation.
- If you reference method-style APIs in docs/examples, migrate them to the family functions.
- If you still rely on builder-style compatibility imports, make sure you import the right surface:
  `siumai::compat::*` for `Siumai`/`SiumaiBuilder`, or `siumai::prelude::compat::*` when you also
  need `Provider`.

## Notes on examples in this repository

- As of `0.11.0-beta.6`, `siumai/examples/*` prefers:
  - `registry::global()` handles for quick starts
  - config-first provider clients (`*Client::from_config(...)`) when provider-specific wiring is needed
- `Siumai::builder()` is intentionally kept as a compatibility convenience, but it is not the
  recommended default for new code. A single explicit builder-based example is kept as a
  comparison reference under `examples/04-provider-specific/openai-compatible/moonshot-siumai-builder.rs`.
