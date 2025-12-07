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
- Structured outputs:
  - Provider‑native structured outputs (OpenAI/Anthropic/Gemini, etc.)
  - Provider‑agnostic decoding helpers with JSON repair and validation (via `siumai-extras`)
- HTTP interceptors, middleware, and a simple retry facade
- Optional extras for telemetry, OpenTelemetry, schema validation, and server adapters

## Install

```toml
[dependencies]
siumai = "0.11.0-beta.4"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Feature flags (enable only what you need):

```toml
# One provider
siumai = { version = "0.11.0-beta.4", features = ["openai"] }

# Multiple providers
siumai = { version = "0.11.0-beta.4", features = ["openai", "anthropic", "google"] }

# All (default)
siumai = { version = "0.11.0-beta.4", features = ["all-providers"] }
```

Optional package for advanced utilities:

```toml
[dependencies]
siumai = "0.11.0-beta.4"
siumai-extras = { version = "0.11.0-beta.4", features = ["schema", "telemetry", "opentelemetry", "server", "mcp"] }
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
- `anthropic-vertex:claude-3-5-sonnet-20240620`
- `gemini:gemini-2.0-flash-exp`
- `groq:llama-3.1-70b-versatile`
- `xai:grok-beta`
- `ollama:llama3.2`
- `minimaxi:minimax-text-01`

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

#### 1) Provider‑agnostic decoding (recommended for cross‑provider flows)

Use `siumai-extras` to parse model text into typed JSON with optional schema validation and repair:

```rust
use serde::Deserialize;
use siumai::prelude::*;
use siumai_extras::highlevel::object::generate_object;

#[derive(Deserialize, Debug)]
struct Post { title: String }

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder().openai().model("gpt-4o-mini").build().await?;
    let (post, _resp) = generate_object::<Post>(
        &client,
        vec![user!("Return JSON: {\"title\":\"hi\"}")],
        None,
        Default::default(),
    ).await?;
    println!("{}", post.title);
    Ok(())
}
```

Under the hood this uses `siumai_extras::structured_output::OutputDecodeConfig` to:
- enforce shape hints (object/array/enum)
- optionally validate against a JSON Schema
- repair common issues (markdown fences, trailing commas, partial slices)

#### 2) Provider‑native structured outputs (example: OpenAI Responses API)

For providers that expose native structured outputs, configure them via provider options.
You still can combine them with the decoding helpers above if you want:

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
// Optionally: further validate/repair/deserialize using `siumai-extras` helpers.
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

### HTTP configuration

You have three practical ways to control HTTP behavior, from simple to advanced.

1) Per‑builder toggles (most common)

```rust
use siumai::prelude::*;

// Provider-specific builder (LlmBuilder)
let client = LlmBuilder::new()
    .with_timeout(std::time::Duration::from_secs(30))
    .with_connect_timeout(std::time::Duration::from_secs(10))
    .with_user_agent("my-app/1.0")
    .with_header("X-User-Project", "acme")
    .with_proxy("http://proxy.example.com:8080") // optional
    .openai()
    .model("gpt-4o-mini")
    .build()
    .await?;

// Unified builder (Siumai::builder) with SSE stability control
let client = Siumai::builder()
    .openai()
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .model("gpt-4o-mini")
    .http_timeout(std::time::Duration::from_secs(30))
    .http_connect_timeout(std::time::Duration::from_secs(10))
    .http_user_agent("my-app/1.0")
    .http_header("X-User-Project", "acme")
    .http_stream_disable_compression(true) // keep SSE stable; default can be controlled by env
    .build()
    .await?;
```

2) HttpConfig builder + shared client builder (centralized configuration)

```rust
use siumai::execution::http::client::build_http_client_from_config;
use siumai::types::HttpConfig;
use siumai::prelude::*;

// Construct a reusable HTTP config
let http_cfg = HttpConfig::builder()
    .timeout(Some(std::time::Duration::from_secs(30)))
    .connect_timeout(Some(std::time::Duration::from_secs(10)))
    .user_agent(Some("my-app/1.0"))
    .proxy(Some("http://proxy.example.com:8080"))
    .header("X-User-Project", "acme")
    .stream_disable_compression(true) // explicit SSE stability
    .build();

// Build reqwest client using the shared helper
let http = build_http_client_from_config(&http_cfg)?;

// Inject it into a builder (takes precedence over other HTTP settings)
let client = LlmBuilder::new()
    .with_http_client(http)
    .openai()
    .model("gpt-4o-mini")
    .build()
    .await?;
```

3) Fully custom reqwest client (maximum control)

```rust
use siumai::prelude::*;

let http = reqwest::Client::builder()
    .timeout(std::time::Duration::from_secs(30))
    // .danger_accept_invalid_certs(true) // if needed for dev
    .build()?;

let client = LlmBuilder::new()
    .with_http_client(http)
    .openai()
    .model("gpt-4o-mini")
    .build()
    .await?;
```

Notes:
- Streaming stability: By default, `stream_disable_compression` is derived from `SIUMAI_STREAM_DISABLE_COMPRESSION` (true unless set to `false|0|off|no`). You can override it per request via the unified builder method `http_stream_disable_compression`.
- When a custom `reqwest::Client` is provided via `.with_http_client(..)`, it takes precedence over any other HTTP settings on the builder.

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

## Acknowledgements

This project draws inspiration from:
- [Vercel AI SDK](https://github.com/vercel/ai) (adapter patterns)
- [Cherry Studio](https://github.com/CherryHQ/cherry-studio) (transformer design)

## Changelog and license

See `CHANGELOG.md` for detailed changes and migration tips.

Licensed under either of:
- Apache License, Version 2.0, or
- MIT license

at your option.
