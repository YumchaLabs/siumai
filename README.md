# Siumai — Unified LLM Interface for Rust

[![Crates.io](https://img.shields.io/crates/v/siumai.svg)](https://crates.io/crates/siumai)
[![Documentation](https://docs.rs/siumai/badge.svg)](https://docs.rs/siumai)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Siumai (烧卖) is a type-safe Rust library that provides a single, consistent API over multiple LLM providers. It focuses on clear abstractions, predictable behavior, and practical extensibility.

This README keeps things straightforward: what you can do, how to customize, and short examples.

## What It Provides

- Unified clients for multiple providers (OpenAI, Anthropic, Google Gemini, Ollama, Groq, xAI, and OpenAI‑compatible vendors)
- Capability traits for chat, streaming, tools, vision, audio, files, embeddings, and rerank
- Streaming with start/delta/usage/end events and cancellation
- Tool calling and a lightweight orchestrator for multi‑step workflows
- Structured outputs (JSON/schema) with repair and validation helpers
- HTTP interceptors, middleware, and a simple retry facade
- Optional extras for telemetry, OpenTelemetry, schema validation, and server adapters

## Install

```toml
[dependencies]
siumai = "0.11.0-beta.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Feature flags (enable only what you need):

```toml
# One provider
siumai = { version = "0.11.0-beta.1", features = ["openai"] }

# Multiple providers
siumai = { version = "0.11.0-beta.1", features = ["openai", "anthropic", "google"] }

# All (default)
siumai = { version = "0.11.0-beta.1", features = ["all-providers"] }
```

Optional package for advanced utilities:

```toml
[dependencies]
siumai = "0.11.0-beta.1"
siumai-extras = { version = "0.11.0-beta.1", features = ["schema", "telemetry", "opentelemetry", "server", "mcp"] }
```

## Usage

### Registry (recommended)

Use the registry to resolve models via `provider:model` and get a handle with a uniform API.

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reg = registry::global();
    let model = reg.language_model("openai:gpt-4o-mini")?;
    let resp = model.chat(vec![user!("Hello")], None).await?;
    println!("{}", resp.content_text().unwrap_or_default());
    Ok(())
}
```

Supported examples of `provider:model`:
- `openai:gpt-4o`, `openai:gpt-4o-mini`
- `anthropic:claude-3-5-sonnet-20240620`
- `gemini:gemini-2.0-flash-exp`
- `groq:llama-3.1-70b-versatile`
- `xai:grok-beta`
- `ollama:llama3.2`

OpenAI‑compatible vendors follow the same pattern (API keys read as `{PROVIDER_ID}_API_KEY` when possible). See docs for details.

### Builder (unified or provider‑specific)

Provider‑specific client:

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Provider::openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .build()
        .await?;

    let resp = client.chat(vec![user!("Hi")]).await?;
    println!("{}", resp.content_text().unwrap_or_default());
    Ok(())
}
```

Unified interface (portable across providers):

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .anthropic()
        .api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .model("claude-3-5-sonnet")
        .build()
        .await?;

    let resp = client.chat(vec![user!("What's Rust?")]).await?;
    println!("{}", resp.content_text().unwrap_or_default());
    Ok(())
}
```

OpenAI‑compatible (custom base URL):

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vllm = LlmBuilder::new()
        .openai()
        .base_url("http://localhost:8000/v1")
        .model("meta-llama/Llama-3.1-8B-Instruct")
        .build()
        .await?;

    let resp = vllm.chat(vec![user!("Hello from vLLM")]).await?;
    println!("{}", resp.content_text().unwrap_or_default());
    Ok(())
}
```

### Streaming

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

    let mut stream = client.chat_stream(vec![user!("Stream a long answer")], None).await?;
    while let Some(ev) = stream.next().await {
        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = ev { print!("{}", delta); }
    }
    Ok(())
}
```

### Structured output

Provider‑agnostic high‑level helper for generating typed JSON:

```rust
use serde::Deserialize;
use siumai::prelude::*;

#[derive(Deserialize, Debug)]
struct Post { title: String }

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder().openai().model("gpt-4o-mini").build().await?;
    let (post, _resp) = siumai::highlevel::object::generate_object::<Post>(
        &client,
        vec![user!("Return JSON: {\"title\":\"hi\"}")],
        None,
        Default::default(),
    ).await?;
    println!("{}", post.title);
    Ok(())
}
```

Recommended: ChatRequestBuilder + ProviderOptions (example: OpenAI Responses API):

```rust
use siumai::prelude::*;
use serde_json::json;

let schema = json!({"type":"object","properties":{"title":{"type":"string"}},"required":["title"]});
let req = ChatRequestBuilder::new()
    .message(user!("Return an object with title"))
    .openai_options(
        OpenAiOptions::new().with_responses_api(
            ResponsesApiConfig::new().with_response_format(json!({
                "type": "json_object",
                "json_schema": { "schema": schema, "strict": true }
            }))
        )
    )
    .build();
let resp = client.chat_request(req).await?;
```

### Retries

```rust
use siumai::prelude::*;
use siumai::retry_api::{retry_with, RetryOptions};

let text = client
    .ask_with_retry("Hello".to_string(), RetryOptions::backoff())
    .await?;
```

## Customization

- HTTP client and headers
- Middleware chain (defaults, clamping, reasoning extraction)
- HTTP interceptors (request/response hooks, SSE observation)
- Retry options and backoff

HTTP configuration example:

```rust
use siumai::prelude::*;
let http = reqwest::Client::builder().build()?;
let client = LlmBuilder::new()
    .with_http_client(http)
    .with_user_agent("my-app/1.0")
    .with_header("X-User-Project", "acme")
    .openai()
    .model("gpt-4o-mini")
    .build()
    .await?;
```

Registry with custom middleware and interceptors:

```rust
use siumai::prelude::*;
use siumai::execution::middleware::samples::chain_default_and_clamp;
use siumai::execution::http::interceptor::LoggingInterceptor;
use siumai::registry::{create_provider_registry, RegistryOptions};
use std::collections::HashMap;
use std::sync::Arc;

let reg = create_provider_registry(
    HashMap::new(),
    Some(RegistryOptions {
        separator: ':',
        language_model_middleware: chain_default_and_clamp(),
        http_interceptors: vec![Arc::new(LoggingInterceptor)],
        retry_options: None,
        max_cache_entries: Some(128),
        client_ttl: None,
        auto_middleware: true,
    })
);
```

## Extras (`siumai-extras`)

- Telemetry subscribers and helpers
- OpenTelemetry middleware (W3C Trace Context)
- JSON schema validation
- Server adapters (Axum SSE)
- MCP utilities

See the `siumai-extras` crate for details and examples.

## Examples

Examples are under `siumai/examples/`:
- 01-quickstart — basic chat, streaming, provider switching
- 02-core-api — chat, streaming, tools, multimodal
- 03-advanced-features — middleware, retry, orchestrator, error types
- 04-provider-specific — provider‑unique capabilities
- 05-integrations — registry, MCP, telemetry
- 06-applications — chatbot, code assistant, API server

Typical commands:

```bash
cargo run --example basic-chat --features openai
cargo run --example streaming --features openai
cargo run --example basic-orchestrator --features openai
```

## Status and notes

- OpenAI Responses API web_search is not implemented yet and returns `UnsupportedOperation`.
- Several modules were reorganized in 0.11: HTTP helpers live under `execution::http::*`, Vertex helpers under `auth::vertex`. See CHANGELOG for migration notes.

API keys and environment variables:
- OpenAI: `.api_key(..)` or `OPENAI_API_KEY`
- Anthropic: `.api_key(..)` or `ANTHROPIC_API_KEY`
- Groq: `.api_key(..)` or `GROQ_API_KEY`
- Gemini: `.api_key(..)` or `GEMINI_API_KEY`
- xAI: `.api_key(..)` or `XAI_API_KEY`
- Ollama: no API key
- OpenAI‑compatible via Registry: reads `{PROVIDER_ID}_API_KEY` (e.g., `DEEPSEEK_API_KEY`)
- OpenAI‑compatible via Builder: `.api_key(..)` or `{PROVIDER_ID}_API_KEY`

## Changelog and license

See `CHANGELOG.md` for detailed changes and migration tips.

Licensed under either of:
- Apache License, Version 2.0, or
- MIT license

at your option.
