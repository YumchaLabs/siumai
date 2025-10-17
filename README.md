# Siumai — Unified LLM Interface for Rust

[![Crates.io](https://img.shields.io/crates/v/siumai.svg)](https://crates.io/crates/siumai)
[![Documentation](https://docs.rs/siumai/badge.svg)](https://docs.rs/siumai)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Siumai (烧卖) is a unified, type‑safe Rust library for working with multiple LLM providers through a consistent API. It features capability‑based traits, robust streaming, and flexible HTTP configuration.

## Why Siumai

- Multi‑provider: OpenAI, Anthropic, Google Gemini, Ollama, Groq, xAI, and more
- Capability‑oriented: chat, streaming, vision, embeddings, files, audio
- Type‑safe builders and shared parameters with provider extensions
- First‑class streaming with start/delta/usage/end events and cancellation
- HTTP customization: custom `reqwest::Client`, headers, proxy, interceptors
- Unified retry facade and structured error classification

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
siumai = "0.10"
tokio = { version = "1.0", features = ["full"] }
```

Hello world (unified interface):

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Choose a provider (OpenAI here) and build a unified client
    let client = Siumai::builder()
        .openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let resp = client.chat(vec![user!("Hello, world!")]).await?;
    println!("{}", resp.content_text().unwrap_or("<no content>"));
    Ok(())
}
```

## Choose Your Style

- Provider‑specific client (access to provider‑only features):

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let openai = Provider::openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4")
        .temperature(0.7)
        .build()
        .await?;

    let resp = openai.chat(vec![user!("Hi!")]).await?;
    println!("{}", resp.content_text().unwrap_or_default());
    Ok(())
}
```

- Unified interface (portable across providers):

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder().anthropic()
        .api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .model("claude-3-5-sonnet")
        .build()
        .await?;

    let resp = client.chat(vec![user!("What's Rust?")]).await?;
    println!("{}", resp.content_text().unwrap_or_default());
    Ok(())
}
```

## Streaming (with Cancellation)

```rust
use futures::StreamExt;
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder().openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let handle = client
        .chat_stream_with_cancel(vec![user!("Stream a long answer")], None)
        .await?;

    tokio::select! {
        _ = async {
            futures::pin_mut!(handle.stream);
            while let Some(ev) = handle.stream.next().await {
                if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = ev { print!("{}", delta); }
            }
            println!();
        } => {}
        _ = tokio::time::sleep(std::time::Duration::from_millis(600)) => {
            handle.cancel.cancel(); // stop the stream early
        }
    }
    Ok(())
}
```

## Retries (Unified Facade)

```rust
use siumai::prelude::*;
use siumai::retry_api::{retry_with, RetryOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Provider::openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .build()
        .await?;

    let text = client
        .ask_with_retry("Hello".to_string(), RetryOptions::backoff())
        .await?;
    println!("{}", text);
    Ok(())
}
```

## HTTP Interceptors (New)

Install custom interceptors globally via `LlmBuilder`, or per‑provider where supported. A built‑in `LoggingInterceptor` is available for lightweight debug (no sensitive data).

```rust
use std::sync::Arc;
use siumai::prelude::*;
use siumai::utils::http_interceptor::LoggingInterceptor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .with_http_interceptor(Arc::new(LoggingInterceptor::default()))
        .openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;
    let _ = client.ask("ping".into()).await?;
    Ok(())
}
```

Interceptors receive hooks: `on_before_send`, `on_response`, `on_error`, and `on_sse_event` for streaming. See `src/utils/http_interceptor.rs`.

## Providers & Features

Enable only what you need to reduce compile time and binary size:

```toml
[dependencies]
# One provider
siumai = { version = "0.10", features = ["openai"] }

# Multiple providers
siumai = { version = "0.10", features = ["openai", "anthropic", "google"] }

# All providers (default)
siumai = { version = "0.10", features = ["all-providers"] }
```

| Feature           | Description                                  |
|-------------------|----------------------------------------------|
| `openai`          | OpenAI and OpenAI‑compatible adapters         |
| `anthropic`       | Anthropic Claude                              |
| `google`          | Google Gemini (multimodal)                    |
| `ollama`          | Local models via Ollama                        |
| `xai`             | xAI Grok                                      |
| `groq`            | Groq (fast inference)                         |
| `all-providers`   | Include all supported providers               |

## Configuration (HTTP, Headers, Proxy)

```rust
use siumai::prelude::*;
use std::time::Duration;

let http = reqwest::Client::builder()
    .timeout(Duration::from_secs(30))
    .build()?;

let client = LlmBuilder::new()
    .with_http_client(http)
    .with_user_agent("my-app/1.0")
    .with_header("X-User-Project", "acme")
    .with_proxy("http://proxy.example.com:8080")
    .openai()
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .model("gpt-4o-mini")
    .build()
    .await?;
```

Environment variables vary by provider (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).

## Custom Provider

Implement `CustomProvider` and wrap it with `CustomProviderClient` to integrate any AI service:

```rust
use siumai::custom_provider::*;

pub struct MyProvider;

#[async_trait::async_trait]
impl CustomProvider for MyProvider {
    fn name(&self) -> &str { "my" }
    fn supported_models(&self) -> Vec<String> { vec!["my-model".into()] }
    fn capabilities(&self) -> siumai::ProviderCapabilities { siumai::ProviderCapabilities::new().with_chat().with_streaming() }
    async fn chat(&self, req: CustomChatRequest) -> Result<CustomChatResponse, siumai::LlmError> {
        Ok(CustomChatResponse::new(format!("hello {}", req.model)))
    }
    async fn chat_stream(&self, _req: CustomChatRequest) -> Result<siumai::ChatStream, siumai::LlmError> { unimplemented!() }
}
```

- Example: `examples/03_advanced_features/custom_provider.rs`
- Guide: `src/custom_provider/guide.rs`

## Advanced Topics (Pointers)

- Vertex AI (Bearer/ADC, publishers, billing headers): see `docs/` (enterprise details moved out of README)
- Files & Audio executors/transformers with consistent headers and tracing: see `docs/` and provider modules
- OpenAI‑compatible adapters and field mapping: see `docs/openai-compatible-architecture.md`
- Tracing and W3C `traceparent`: enable by `SIUMAI_W3C_TRACE=1`
- HTTP Interceptors best practices: see `docs/http-interceptor-best-practices.md`

## Examples

See the `examples/` directory for getting started, core features, providers, advanced features, and MCP integration.

Common commands:

```bash
cargo run --example quick_start
cargo run --example streaming_chat
cargo run --example custom_provider
```

## Changelog & Migration

See `CHANGELOG.md`. The 0.11.x line introduces a Transformers + Executors architecture and unified streaming events; migration notes are included.

## License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license

at your option.
