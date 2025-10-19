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

## 通过 Registry 使用 OpenAI-Compatible Provider（实验）

以下示例展示两种方式来使用 Provider Registry（实验）按 `provider:model` 解析模型并自动挂载“模型级中间件”。

```rust,no_run
use siumai::registry::helpers::create_registry_with_defaults;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 方式 A：使用内置便捷函数（推荐）
    let reg = create_registry_with_defaults();

    // 2) 解析 "provider:model"
    // 例如：OpenAI 原生 -> "openai:gpt-4o"；OpenAI-Compatible -> "openrouter:openai/gpt-4o-mini"
    let lm = reg.language_model("openai:gpt-4o")?;

    // 3) 发送对话（此处仅示意，需设置对应环境变量，例如 OPENAI_API_KEY）
    let messages = vec![siumai::user!("Hello from registry!")];
    let _resp = lm.chat(messages, None).await?;
    Ok(())
}
```

方式 B：自定义中间件
```rust,no_run
use siumai::registry::entry::{create_provider_registry, RegistryOptions};
use siumai::middleware::samples::chain_default_and_clamp;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 自定义中间件与分隔符
    let reg = create_provider_registry(
        HashMap::new(),
        Some(RegistryOptions { separator: ':', language_model_middleware: chain_default_and_clamp() }),
    );
    let lm = reg.language_model("openai:gpt-4o")?;
    let _resp = lm.chat(vec![siumai::user!("Hello")], None).await?;
    Ok(())
}
```

示例（OpenAI-Compatible）
- DeepSeek：
  - 环境变量：`DEEPSEEK_API_KEY`
  - 代码：`let lm = reg.language_model("deepseek:deepseek-chat")?;`
- OpenRouter：
  - 环境变量：`OPENROUTER_API_KEY`
  - 代码：`let lm = reg.language_model("openrouter:openai/gpt-4o-mini")?;`
- SiliconFlow：
  - 环境变量：`SILICONFLOW_API_KEY`
  - 代码：`let lm = reg.language_model("siliconflow:deepseek-ai/DeepSeek-V3")?;`

注意：若 provider 归属 OpenAI-Compatible（在 `src/providers/openai_compatible/config.rs` 列表中），Registry 会使用 `{PROVIDER_ID}_API_KEY` 从环境变量读取 Key 并自动构建客户端。

环境变量命名规则与清单见 `docs/ENV_VARS.md`。可参考示例：`examples/registry_basic.rs`。

## Custom OpenAI-Compatible Providers

Siumai's `.base_url()` + `.model()` API provides full flexibility for using **any** OpenAI-compatible provider, equivalent to Vercel AI SDK's `createOpenAICompatible`. This is perfect for:

- Self-hosted OpenAI-compatible servers (vLLM, LocalAI, Ollama, etc.)
- Custom proxy or gateway configurations
- Any provider with an OpenAI-compatible API

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Self-hosted vLLM server
    let vllm_client = LlmBuilder::new()
        .openai()  // Use OpenAI-compatible interface
        .api_key("not-needed")  // vLLM may not require a key
        .base_url("http://localhost:8000/v1")  // Your vLLM server
        .model("meta-llama/Llama-3.1-8B-Instruct")  // Model served by vLLM
        .build()
        .await?;

    let resp = vllm_client.chat(vec![user!("Hello from vLLM!")]).await?;
    println!("{}", resp.content_text().unwrap_or_default());

    // Example 2: Custom OpenAI-compatible gateway
    let gateway_client = LlmBuilder::new()
        .openai()
        .api_key(std::env::var("GATEWAY_API_KEY")?)
        .base_url("https://my-gateway.example.com/v1")
        .model("custom-model-name")
        .temperature(0.7)
        .build()
        .await?;

    let resp2 = gateway_client.chat(vec![user!("Hello from gateway!")]).await?;
    println!("{}", resp2.content_text().unwrap_or_default());

    // Example 3: Use any provider with OpenAI-compatible API
    let together_client = LlmBuilder::new()
        .openai()
        .api_key(std::env::var("TOGETHER_API_KEY")?)
        .base_url("https://api.together.xyz/v1")
        .model("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        .build()
        .await?;

    let resp3 = together_client.chat(vec![user!("Hello from Together AI!")]).await?;
    println!("{}", resp3.content_text().unwrap_or_default());

    Ok(())
}
```

**Key Features:**
- ✅ **Full OpenAI API compatibility**: Works with any server implementing OpenAI's API spec
- ✅ **Flexible configuration**: Override base URL and model for any use case
- ✅ **All Siumai features**: Streaming, retries, middleware, interceptors all work seamlessly
- ✅ **Type-safe**: Full Rust type safety and error handling

**Supported out-of-the-box** (with built-in defaults):
- DeepSeek, OpenRouter, Groq, xAI, SiliconFlow, Fireworks, Together AI, and more

**Custom providers**: Use `.base_url()` + `.model()` for any other OpenAI-compatible service!

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

## Structured Output (JSON/Schema)

Siumai supports provider‑agnostic structured outputs. Pass a hint via `provider_params.structured_output` and the library maps it to each provider’s JSON mode:

```rust
use siumai::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder().openai()
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let schema = json!({
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"]
    });
    let mut pp = ProviderParams::new();
    pp = pp.with_param("structured_output", json!({"schema": schema}));

    let req = ChatRequestBuilder::new()
        .model("gpt-4o-mini")
        .message(user!("Return a JSON object with title"))
        .provider_params(pp)
        .build();

    let resp = client.chat_request(req).await?;
    println!("{}", resp.content_text().unwrap_or_default());
    Ok(())
}
```

Notes:
- OpenAI: uses `response_format` (json_object/json_schema + strict)
- Gemini: sets `generationConfig.responseMimeType/responseSchema`
- Anthropic: sets `response_format` for Messages API
- Groq/xAI: sets OpenAI‑compatible `response_format`
- Ollama: sets `format` (schema or "json")

See more patterns and high‑level helpers in `docs/OBJECT_API_AND_OPENAI_STRUCTURED_OUTPUT.md`.

## OpenAI‑Compatible (Adapters)

OpenAI‑compatible providers share one meta provider using adapters instead of separate provider types. Each adapter declares base URL, headers, and capability quirks (JSON mode, tools, streaming, reasoning fields). This keeps a single execution path (transformers + executors) while capturing provider differences.

- Build via convenience helpers, e.g. `LlmBuilder::new().openrouter()`, `deepseek()`, `together()`, or the generic `OpenAiCompatibleBuilder`.
- Environment: adapters read `{PROVIDER_ID}_API_KEY` (see `docs/ENV_VARS.md`).
- Structured output: pass `provider_params.structured_output` and it maps to `response_format` for all compat providers.
- Tools/streaming/thinking: handled by the adapter’s field mappings and compatibility flags.

Example (OpenRouter via adapter):

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .openrouter() // OpenAI-compatible adapter
        .model("openai/gpt-4o-mini")
        .build()
        .await?;

    let resp = client.chat(vec![user!("Hello from OpenRouter")]).await?;
    println!("{}", resp.content_text().unwrap_or_default());
    Ok(())
}
```

## Middleware

- Simulate streaming middleware and example: docs/developer/simulate_streaming_middleware.md
- Example code: examples/03_advanced_features/middleware_simulate_streaming.rs

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

## Provider Capabilities (Quick Matrix)

- OpenAI
  - Chat/Stream: SSE; Tools: function calls; Structured Output: `response_format` (json_object/json_schema + strict)
  - Images/Audio/Embeddings: supported via dedicated executors/transformers
- Google Gemini
  - Chat/Stream: SSE; Tools: function calls + code execution (models dependent)
  - Structured Output: `generationConfig.responseMimeType/responseSchema`
- Anthropic Claude
  - Chat/Stream: SSE; Tools: function calls; Thinking: supported（beta flags）
  - Structured Output: `response_format`（json_object/json_schema）
- Groq
  - Chat/Stream: SSE; Tools: function calls; Audio: TTS/STT helpers
  - Structured Output: OpenAI‑style `response_format`
- xAI Grok
  - Chat/Stream: SSE; Tools: function calls; Thinking: reasoning fields
  - Structured Output: OpenAI‑style `response_format`
- Ollama
  - Chat/Stream: JSON streaming; Tools: function calls（模型支持）
  - Structured Output: `format`（json or schema object）
- OpenAI‑Compatible（Meta Provider）
  - Chat/Stream: SSE（依适配器）；Tools/Thinking：由适配器能力决定
  - Structured Output: `response_format`（由 adapters 统一映射）

Notes
- 行为以各模型与供应商实际支持为准；以上为 Siumai 侧映射与执行路径能力概览。

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

## Server Adapters Examples

需要启用示例特性与 Provider：

```
cargo run --example axum_sse  --features "openai,server-adapters"
cargo run --example actix_sse --features "openai,server-adapters"
```
