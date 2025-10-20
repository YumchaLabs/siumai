# Siumai — Unified LLM Interface for Rust

[![Crates.io](https://img.shields.io/crates/v/siumai.svg)](https://crates.io/crates/siumai)
[![Documentation](https://docs.rs/siumai/badge.svg)](https://docs.rs/siumai)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Siumai (烧卖) is a **production-ready**, **type-safe** Rust library for working with multiple LLM providers through a unified API. Built with a clean **Transformers + Executors** architecture, it provides first-class support for streaming, tool calling, structured outputs, and advanced features like middleware, orchestration, and observability.

## ✨ Why Siumai

### 🎯 **Unified Multi-Provider Interface**
- **8+ Providers**: OpenAI, Anthropic, Google Gemini, Ollama, Groq, xAI, DeepSeek, OpenRouter, and any OpenAI-compatible service
- **Consistent API**: Write once, switch providers with a single line change
- **Provider-Specific Features**: Access unique capabilities (Anthropic thinking, Gemini code execution, etc.) through unified interfaces

### 🏗️ **Clean Architecture**
- **Transformers + Executors**: Modular design separating request/response transformation from HTTP execution
- **Capability Traits**: Type-safe capability discovery (Chat, Streaming, Vision, Embeddings, Audio, Files, Tools)
- **Easy to Extend**: Add new providers by implementing transformers—no core code changes needed

### 🚀 **Production Features**
- **First-Class Streaming**: SSE with multi-event emission (start/delta/usage/end), cancellation, and backpressure
- **Tool Calling & Orchestration**: Multi-step tool execution with automatic retry and approval workflows
- **Structured Outputs**: Provider-agnostic JSON schema validation and typed responses
- **Middleware System**: Transform requests/responses at model level (parameter clamping, default injection, etc.)
- **Observability**: Built-in tracing, telemetry exporters (Langfuse, Helicone), and performance metrics
- **HTTP Interceptors**: Custom request/response hooks for logging, auth, and debugging
- **Retry & Error Handling**: Unified retry facade with exponential backoff and structured error classification

### 📦 **Flexible & Lightweight**
- **Feature Flags**: Include only the providers you need to reduce compile time and binary size
- **Optional Extras**: Separate `siumai-extras` package for heavy dependencies (JSON schema, telemetry, server adapters)
- **Async-First**: Built on `tokio` with full async/await support

## 🏗️ Architecture Overview

Siumai is built on a clean, modular architecture that separates concerns and makes it easy to extend:

```
┌─────────────────────────────────────────────────────────────┐
│                     User API Layer                          │
│  (Siumai, Provider, LlmBuilder, ChatRequest, etc.)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Capability Traits                         │
│  (ChatCapability, EmbeddingCapability, AudioCapability...)  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Middleware Layer                           │
│  (Parameter transformation, defaults, validation)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Executors Layer                           │
│  (HttpChatExecutor, HttpEmbeddingExecutor, etc.)           │
│  - HTTP orchestration                                       │
│  - Header building, retry, tracing                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Transformers Layer                         │
│  (RequestTransformer, ResponseTransformer, etc.)           │
│  - Provider-specific request/response mapping              │
│  - Streaming event conversion                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Provider Clients                          │
│  (OpenAI, Anthropic, Gemini, Ollama, Groq, xAI...)        │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **Traits**: Define capabilities (chat, streaming, vision, etc.) that providers can implement
- **Transformers**: Convert between unified types and provider-specific formats
- **Executors**: Handle HTTP execution, headers, retry, and tracing
- **Middleware**: Transform requests/responses at the model level
- **Orchestrator**: Coordinate multi-step tool calling workflows
- **Registry**: Discover and instantiate providers dynamically

### Why This Architecture?

✅ **Separation of Concerns**: Each layer has a single responsibility
✅ **Easy to Extend**: Add new providers by implementing transformers
✅ **Testable**: Mock any layer for unit testing
✅ **Maintainable**: Changes to one provider don't affect others
✅ **Type-Safe**: Rust's type system ensures correctness at compile time

## 🚀 Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
siumai = "0.11"
tokio = { version = "1.0", features = ["rt-multi-thread", "macros"] }
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

## 📦 Providers & Features

### Core Package (`siumai`)

Enable only what you need to reduce compile time and binary size:

```toml
[dependencies]
# One provider
siumai = { version = "0.11", features = ["openai"] }

# Multiple providers
siumai = { version = "0.11", features = ["openai", "anthropic", "google"] }

# All providers (default)
siumai = { version = "0.11", features = ["all-providers"] }
```

| Feature           | Description                                  |
|-------------------|----------------------------------------------|
| `openai`          | OpenAI and OpenAI‑compatible adapters         |
| `anthropic`       | Anthropic Claude                              |
| `google`          | Google Gemini (multimodal)                    |
| `ollama`          | Local models via Ollama                        |
| `xai`             | xAI Grok                                      |
| `groq`            | Groq (fast inference)                         |
| `all-providers`   | Include all supported providers (default)     |
| `gcp`             | Google Cloud Platform authentication          |

### Extras Package (`siumai-extras`)

Optional features for advanced use cases (separate package to keep core lightweight):

```toml
[dependencies]
siumai = "0.11"
siumai-extras = { version = "0.11", features = ["schema", "telemetry", "server"] }
```

| Feature      | Description                                           |
|--------------|-------------------------------------------------------|
| `schema`     | JSON schema validation with `jsonschema` crate        |
| `telemetry`  | Advanced tracing subscriber configuration             |
| `server`     | Server adapters (Axum SSE, etc.)                      |
| `all`        | All extras features                                   |

**Why separate packages?**
- **Smaller binaries**: Core package has minimal dependencies
- **Faster compilation**: Only include heavy dependencies when needed
- **Flexibility**: Mix and match features based on your use case

See [Migration Guide](CHANGELOG.md#0110---2025-01-xx) for upgrading from 0.10.x.

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

## 🎯 Feature Highlights

### 🔧 **Tool Calling & Orchestration**
- **Multi-Step Execution**: Automatic tool calling with retry and approval workflows
- **Parallel Tools**: Execute multiple tools concurrently
- **Custom Tools**: Define your own tools with type-safe schemas
- **Streaming Tools**: Stream tool call results in real-time
- See: `examples/orchestrator_*.rs`, `src/orchestrator/`

### 📊 **Structured Outputs**
- **Provider-Agnostic**: Same API works across OpenAI, Anthropic, Gemini, etc.
- **JSON Schema**: Validate responses against JSON schemas
- **Typed Responses**: Generate Rust structs from LLM responses
- **Streaming Objects**: Stream structured data incrementally
- See: `docs/OBJECT_API_AND_OPENAI_STRUCTURED_OUTPUT.md`, `examples/highlevel/`

### 🎨 **Multimodal Support**
- **Vision**: Image understanding across providers
- **Audio**: Text-to-speech (TTS) and speech-to-text (STT)
- **Image Generation**: DALL-E, Imagen, and more
- **File Management**: Upload, list, retrieve, and delete files
- See: `examples/04_providers/*/vision_*.rs`, `examples/04_providers/*/audio_*.rs`

### 🔄 **Middleware & Customization**
- **Model-Level Middleware**: Transform requests/responses before execution
- **Parameter Clamping**: Automatically adjust parameters to provider limits
- **Default Injection**: Set default values for missing parameters
- **Custom Middleware**: Implement your own transformation logic
- See: `src/middleware/`, `examples/03_advanced_features/middleware_*.rs`

### 📡 **Observability**
- **Tracing**: Built-in `tracing` instrumentation for debugging
- **Telemetry**: Export events to Langfuse, Helicone, and other platforms
- **Performance Metrics**: Track latency, throughput, and error rates
- **W3C Trace Context**: Propagate trace IDs across services (`SIUMAI_W3C_TRACE=1`)
- See: `src/tracing/README.md`, `src/telemetry/README.md`, `docs/developer/performance_module.md`

### 🔌 **HTTP Interceptors**
- **Request/Response Hooks**: Intercept and modify HTTP traffic
- **Logging**: Built-in `LoggingInterceptor` for debugging
- **Custom Auth**: Implement custom authentication logic
- **SSE Events**: Hook into streaming events
- See: `src/utils/http_interceptor.rs`, `docs/http-interceptor-best-practices.md`

### 🌐 **Provider Registry**
- **Dynamic Discovery**: Discover and instantiate providers at runtime
- **Model Resolution**: Resolve `provider:model` strings to clients
- **Middleware Injection**: Attach middleware to specific models
- **Environment Variables**: Auto-configure from env vars
- See: `src/registry/`, `examples/registry_*.rs`, `docs/ENV_VARS.md`

## 📚 Advanced Topics

- **Vertex AI**: Bearer/ADC authentication, publishers, billing headers → `docs/`
- **Files & Audio**: Executors/transformers with consistent headers and tracing → `docs/` and provider modules
- **OpenAI-Compatible**: Adapter architecture and field mapping → `docs/openai-compatible-architecture.md`
- **Custom Providers**: Implement your own provider → `src/custom_provider/guide.rs`, `examples/03_advanced_features/custom_provider.rs`
- **HTTP Interceptors**: Best practices and examples → `docs/http-interceptor-best-practices.md`
- **Code Organization**: Module structure and patterns → `docs/developer/code_organization.md`

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

## 🌐 Server Adapters

Siumai provides server adapters for building LLM-powered APIs with popular Rust web frameworks.

### Axum SSE Example

```rust
use siumai::prelude::*;
use siumai_extras::server::axum::to_sse_response;
use axum::{routing::post, Router};

async fn chat_handler() -> impl axum::response::IntoResponse {
    let client = Siumai::builder()
        .openai()
        .api_key(std::env::var("OPENAI_API_KEY").unwrap())
        .model("gpt-4o-mini")
        .build()
        .await
        .unwrap();

    let stream = client
        .chat_stream(vec![user!("Hello!")], None)
        .await
        .unwrap();

    to_sse_response(stream)
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/chat", post(chat_handler));
    // ... run server
}
```

**Run examples:**

```bash
# Requires siumai-extras with server feature
cargo run --example axum_sse --features "openai"
```

See: `examples/server/`, `siumai-extras/src/server/`
