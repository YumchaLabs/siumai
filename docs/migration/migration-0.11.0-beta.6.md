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
  - unified builder: `Siumai::builder()...build().await?`
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

- OpenAI-compatible vendors (DeepSeek/OpenRouter/Moonshot/etc.) now have a config-first shortcut:
  - `siumai::providers::openai_compatible::OpenAiCompatibleClient::from_builtin_env(...)`

`from_builtin_env` reads API keys from env using this precedence:
1) `ProviderConfig.api_key_env` (when present)
2) `ProviderConfig.api_key_env_aliases` (fallbacks)
3) `${PROVIDER_ID}_API_KEY` (uppercased, `-` replaced with `_`)

To discover built-in OpenAI-compatible provider ids, call
`siumai::providers::openai_compatible::list_provider_ids()`.

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
For explicitness (and to prepare for future deprecation), you can switch your imports to:

```rust,ignore
use siumai::compat::*;
```

Compatibility note: the `siumai::compat` module is intended to be temporary.
Planned removal target is **no earlier than `0.12.0`**.

## Checklist

- Keep construction code unchanged.
- Prefer `siumai::text::*` for chat-style inference.
- If you reference method-style APIs in docs/examples, migrate them to the family functions.

## Notes on examples in this repository

- As of `0.11.0-beta.6`, `siumai/examples/*` prefers:
  - `registry::global()` handles for quick starts
  - config-first provider clients (`*Client::from_config(...)`) when provider-specific wiring is needed
- `Siumai::builder()` is intentionally kept as a compatibility convenience, but it is not the
  recommended default for new code. A single explicit builder-based example is kept as a
  comparison reference under `examples/04-provider-specific/openai-compatible/moonshot-siumai-builder.rs`.
