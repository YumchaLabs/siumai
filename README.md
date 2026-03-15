# Siumai — Unified LLM Interface for Rust





[![Crates.io](https://img.shields.io/crates/v/siumai.svg)](https://crates.io/crates/siumai)


[![Documentation](https://docs.rs/siumai/badge.svg)](https://docs.rs/siumai)


[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)


[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/YumchaLabs/siumai)





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


siumai = "0.11.0-beta.6"


tokio = { version = "1", features = ["rt-multi-thread", "macros"] }


```





## Migration (beta.6)





Upgrading from `0.11.0-beta.4` (or earlier)?





- See `docs/migration/migration-0.11.0-beta.6.md`


  - Note: legacy method-style entry points are treated as compatibility surface; the explicit module is `siumai::compat`.





Feature flags (enable only what you need):





```toml


# One provider


siumai = { version = "0.11.0-beta.6", features = ["openai"] }





# Multiple providers


siumai = { version = "0.11.0-beta.6", features = ["openai", "anthropic", "google"] }





# All


siumai = { version = "0.11.0-beta.6", features = ["all-providers"] }


```





Note: `siumai` enables `openai` by default. Disable defaults via `default-features = false`.





Optional package for advanced utilities:





```toml


[dependencies]


siumai = "0.11.0-beta.6"


siumai-extras = { version = "0.11.0-beta.6", features = ["schema", "telemetry", "opentelemetry", "server", "mcp"] }


```





## Usage





### Construction order





For new code, prefer construction modes in this order:





1. `registry-first` for application code and cross-provider routing


2. `config-first` for provider-specific setup and tests


3. `builder convenience` for quick setup, migration, and side-by-side comparison





Rule of thumb:



- reach for `registry::global().language_model("provider:model")?` in app code

- reach for `*Client::from_config(*Config::new(...))` in provider-specific code

- treat `Siumai::builder()` and provider builders as convenience wrappers, not the architectural center



### Public surface map



Use public surfaces by intent, not by habit:



- **App-level routing and default usage**: `registry::global()` + the six family APIs `text::{generate, stream}`, `embedding::embed`, `image::generate`, `rerank::rerank`, `speech::synthesize`, and `transcription::transcribe`

- **Provider-specific construction**: `siumai::providers::<provider>::*Config` + `*Client::from_config(...)`

- **Provider-specific typed escape hatches**: `siumai::provider_ext::<provider>::options::*`, request ext traits, and typed response metadata helpers

- **Migration and quick demos**: `siumai::compat::Siumai` and `Provider::*()` builders

- **Last-resort vendor knobs**: raw `with_provider_option(...)` when a typed provider extension does not exist yet



Policy for new features:



- no new capability should be builder-only

- typed provider knobs should live under `provider_ext::<provider>` before recommending raw provider-option maps

- docs and examples should present `registry-first -> config-first -> builder convenience` in that order

- docs should treat the six family modules `text`, `embedding`, `image`, `rerank`, `speech`, and `transcription` as the primary public entry points




### Registry (recommended)





Use the registry to resolve models via `provider:model` and get a handle with a uniform API.





```rust


use siumai::prelude::unified::*;





#[tokio::main]


async fn main() -> Result<(), Box<dyn std::error::Error>> {


    let reg = registry::global();


    let model = reg.language_model("openai:gpt-4o-mini")?;


    let resp = text::generate(


        &model,


        ChatRequest::new(vec![user!("Hello")]),


        text::GenerateOptions::default(),


    )


    .await?;


    println!("{}", resp.content_text().unwrap_or_default());


    Ok(())


}


```





Note: OpenAI routing via the registry uses the Responses API by default. If you specifically need


Chat Completions (`POST /chat/completions`) for a specific request, override it via provider options:





```rust


use siumai::prelude::unified::*;


use siumai::providers::openai::{OpenAiChatRequestExt, OpenAiOptions, ResponsesApiConfig};





let req = ChatRequest::new(vec![user!("Hello")]).with_openai_options(


    OpenAiOptions::new().with_responses_api(ResponsesApiConfig {


        enabled: false,


        ..Default::default()


    }),


);


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





### OpenAI-compatible vendors (config-first)

Typed vendor views such as `siumai::provider_ext::openrouter` and `siumai::provider_ext::perplexity`
are helper layers over the same compat runtime; they do not imply that every preset should grow
into a separate full provider package.





For OpenAI-compatible providers like Moonshot/OpenRouter/DeepSeek, you can use the built-in vendor registry:





```rust,no_run


use siumai::prelude::unified::*;


use siumai::providers::openai_compatible::OpenAiCompatibleClient;





#[tokio::main]


async fn main() -> Result<(), Box<dyn std::error::Error>> {


    // Reads `DEEPSEEK_API_KEY` by default.


    let client = OpenAiCompatibleClient::from_builtin_env("deepseek", Some("deepseek-chat")).await?;





    let resp = text::generate(


        &client,


        ChatRequest::new(vec![user!("hi")]),


        text::GenerateOptions::default(),


    )


    .await?;





    println!("{}", resp.content_text().unwrap_or_default());


    Ok(())


}


```





Notes:


- `OpenAiCompatibleClient::from_builtin_env` reads API keys from env using this precedence:


  1) `ProviderConfig.api_key_env` (when present)


  2) `ProviderConfig.api_key_env_aliases` (fallbacks)


  3) `${PROVIDER_ID}_API_KEY` (uppercased, `-` replaced with `_`)


- To discover built-in OpenAI-compatible provider ids, call


  `siumai::providers::openai_compatible::list_provider_ids()`.





### Provider clients (config-first)





Provider-specific client:





```rust


use siumai::prelude::unified::*;


use siumai::providers::openai::{OpenAiClient, OpenAiConfig};





#[tokio::main]


async fn main() -> Result<(), Box<dyn std::error::Error>> {


    let cfg = OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)


        .with_model("gpt-4o")


        .with_temperature(0.7);


    let client = OpenAiClient::from_config(cfg)?;





    let resp = text::generate(


        &client,


        ChatRequest::new(vec![user!("Hi")]),


        text::GenerateOptions::default(),


    )


    .await?;


    println!("{}", resp.content_text().unwrap_or_default());


    Ok(())


}


```





MiniMaxi (config-first):





```rust,no_run


use siumai::models;


use siumai::prelude::unified::*;


use siumai::providers::minimaxi::{MinimaxiClient, MinimaxiConfig};





#[tokio::main]


async fn main() -> Result<(), Box<dyn std::error::Error>> {


    let cfg = MinimaxiConfig::new(std::env::var("MINIMAXI_API_KEY")?)


        .with_model(models::minimaxi::MINIMAX_M2);


    let client = MinimaxiClient::from_config(cfg)?;





    let resp = text::generate(


        &client,


        ChatRequest::new(vec![user!("Hello MiniMaxi!")]),


        text::GenerateOptions::default(),


    )


    .await?;





    println!("{}", resp.content_text().unwrap_or_default());


    Ok(())


}


```





OpenAI‑compatible (custom base URL):





```rust


use siumai::prelude::unified::*;


use siumai::providers::openai::{OpenAiClient, OpenAiConfig};





#[tokio::main]


async fn main() -> Result<(), Box<dyn std::error::Error>> {


    // For OpenAI-compatible local endpoints, you can use any non-empty API key.


    let cfg = OpenAiConfig::new("dummy")


        .with_base_url("http://localhost:8000/v1")


        .with_model("meta-llama/Llama-3.1-8B-Instruct");


    let vllm = OpenAiClient::from_config(cfg)?;





    let resp = text::generate(


        &vllm,


        ChatRequest::new(vec![user!("Hello from vLLM")]),


        text::GenerateOptions::default(),


    )


    .await?;


    println!("{}", resp.content_text().unwrap_or_default());


    Ok(())


}


```





### Rerank (registry-first)

```rust,no_run
use siumai::prelude::unified::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry_id = "cohere:rerank-english-v3.0";
    let model = registry::global().reranking_model(registry_id)?;

    let response = rerank::rerank(
        &model,
        RerankRequest::new(
            "rerank-english-v3.0".to_string(),
            "Which document is about Rust SDK architecture?".to_string(),
            vec![
                "A Python crawler tutorial".to_string(),
                "A Rust SDK architecture guide".to_string(),
                "A dumpling recipe".to_string(),
            ],
        )
        .with_top_n(2),
        rerank::RerankOptions::default(),
    )
    .await?;

    println!("{:?}", response.sorted_indices());
    Ok(())
}
```

Provider-specific rerank setup should still prefer config-first clients plus typed request extensions.
Runnable references:

- `siumai/examples/05-integrations/registry/rerank.rs`
- `siumai/examples/04-provider-specific/cohere/rerank.rs`
- `siumai/examples/04-provider-specific/togetherai/rerank.rs`
- `siumai/examples/04-provider-specific/bedrock/rerank.rs`

#### OpenAI endpoint routing (Responses vs Chat Completions)





Siumai supports both OpenAI chat endpoints:





- **Responses API**: `POST /responses` (default)


- **Chat Completions**: `POST /chat/completions` (override via `providerOptions.openai.responsesApi.enabled = false`)





If you need to override the default on a per-request basis, set `providerOptions.openai.responsesApi.enabled`


explicitly on the `ChatRequest`.





### Builder convenience (compat)





Builder-style construction remains available as a **temporary** compatibility surface.


It is useful for quick demos and migration, but it is **not** the recommended default for new code.





Recommended order:





- first: registry-first for app-level code


- second: config-first for provider-specific code


- third: builder convenience for quick setup and comparison





If you still want the builder style, prefer importing it explicitly from `siumai::compat`:





```rust,ignore


use siumai::compat::Siumai;





let client = Siumai::builder()


    .openai()


    .api_key(std::env::var("OPENAI_API_KEY")?)


    .model("gpt-4o-mini")


    .build()


    .await?;


```





Compatibility note: planned removal target is **no earlier than `0.12.0`**.

For details, see `docs/migration/migration-0.11.0-beta.6.md`.



Builder policy note: builders are expected to converge on the same config-first construction path.

If a feature matters for real usage, it should also be reachable from provider config/client APIs and,

when appropriate, typed provider extensions under `provider_ext`.




### Streaming





```rust


use futures::StreamExt;


use siumai::prelude::unified::*;





#[tokio::main]


async fn main() -> Result<(), Box<dyn std::error::Error>> {


    let reg = registry::global();


    let model = reg.language_model("openai:gpt-4o-mini")?;





    let mut stream = text::stream(


        &model,


        ChatRequest::new(vec![user!("Stream a long answer")]),


        text::StreamOptions::default(),


    )


    .await?;


    while let Some(ev) = stream.next().await {


        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = ev { print!("{}", delta); }


    }


    Ok(())


}


```





#### Streaming cancellation





`chat_stream_with_cancel` returns a `ChatStreamHandle` with a first-class `CancelHandle`.


Cancellation is wakeable: it can stop a pending `next().await` immediately (useful for both SSE and WebSocket streams).





```rust,no_run


use futures::StreamExt;


use siumai::prelude::unified::*;





#[tokio::main]


async fn main() -> Result<(), Box<dyn std::error::Error>> {


    let reg = registry::global();


    let model = reg.language_model("openai:gpt-4o-mini")?;


    let handle = text::stream_with_cancel(


        &model,


        ChatRequest::new(vec![user!("Stream...")]),


        text::StreamOptions::default(),


    )


    .await?;





    let ChatStreamHandle { mut stream, cancel } = handle;


    let reader = tokio::spawn(async move { while stream.next().await.is_some() {} });





    cancel.cancel();


    reader.await?;


    Ok(())


}


```





### OpenAI WebSocket streaming (Responses API)





If you have many sequential streaming steps (e.g., tool loops), OpenAI's WebSocket mode can reduce


TTFB by reusing a persistent connection. Enable the feature and inject the transport:





Note: `base_url` must use `http://` or `https://` (it is converted to `ws://` / `wss://` internally).


WebSocket mode only applies to **Responses streaming** (`POST /responses`). It is not compatible with


Chat Completions (`POST /chat/completions`).





```toml


# Cargo.toml


siumai = { version = "0.11.0-beta.6", features = ["openai-websocket"] }


```





```rust,no_run


	use futures::StreamExt;


	use siumai::prelude::unified::*;


	use siumai::providers::openai::{OpenAiClient, OpenAiConfig, OpenAiWebSocketTransport};


	use std::sync::Arc;


	


	#[tokio::main]


	async fn main() -> Result<(), Box<dyn std::error::Error>> {


	    // OpenAI WebSocket connections are time-limited; by default we avoid reusing connections


	    // older than ~55 minutes. Customize or disable if needed.


	    let ws = OpenAiWebSocketTransport::default()


	        // Keep up to N idle connections for concurrent tool loops.


	        .with_max_idle_connections(2);


	    // let ws = ws.with_max_connection_age(std::time::Duration::from_secs(55 * 60));


	    // let ws = ws.without_max_connection_age();





    // Optional: connection-local incremental continuation (`previous_response_id`).


    // Note: OpenAI caches the most recent response per WebSocket connection, so this is


    // only unambiguous when `max_idle_connections == 1`.


    // let ws = ws.with_stateful_previous_response_id(true);





    let cfg = OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)


        .with_model("gpt-4o-mini")


        .with_http_transport(Arc::new(ws.clone()));


    let client = OpenAiClient::from_config(cfg)?;





    // Streaming `/responses` requests are routed through WebSocket; everything else uses HTTP.


    let mut stream = text::stream(


        &client,


        ChatRequest::new(vec![user!("Hello!")]),


        text::StreamOptions::default(),


    )


    .await?;


    while let Some(ev) = stream.next().await {


        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = ev {


            print!("{delta}");


        }


    }





    ws.close().await; // optional: close the cached connection


    Ok(())


}


```





#### OpenAI WebSocket session (warm-up + single connection)





For agentic workflows with many sequential streaming steps, prefer a single-connection session


so `previous_response_id` continuation stays unambiguous:





This session also includes a conservative recovery strategy:


- if WebSocket setup fails (transient/connectivity), it falls back to HTTP (SSE) streaming for that request


- for some WebSocket-specific OpenAI errors, it may rebuild the connection and retry once





Note: configuration errors (e.g. invalid `base_url`, unsupported URL scheme) are surfaced directly and do not fall back to HTTP.





You can customize it, e.g. disable all recovery:


`OpenAiWebSocketSession::from_config_default_http(cfg)?.with_recovery_config(OpenAiWebSocketRecoveryConfig { allow_http_fallback: false, max_ws_retries: 0 });`





Important: recovery may rebuild the WebSocket connection (or fall back to HTTP), which resets


connection-local continuation state (`previous_response_id`). If you strictly rely on continuation


via a single warm connection, consider disabling recovery.





When recovery happens, the session also emits `ChatStreamEvent::Custom` with `event_type="openai:ws-recovery"`.





`OpenAiWebSocketSession` also attempts best-effort remote cancellation when using `chat_stream_with_cancel(...)`


by calling `POST /responses/{id}/cancel` once the response id is observed. Disable via `session.with_remote_cancel(false)`.





```rust,no_run


use futures::StreamExt;


use siumai::prelude::unified::*;


use siumai::providers::openai::{OpenAiConfig, OpenAiWebSocketSession};





#[tokio::main]


async fn main() -> Result<(), Box<dyn std::error::Error>> {


    let cfg = OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)


        .with_model("gpt-4o-mini");


    let session = OpenAiWebSocketSession::from_config_default_http(cfg)?;





    session.warm_up_messages(vec![user!("Warm up with my toolset")], None).await?;





    let mut stream = text::stream(


        &session,


        ChatRequest::new(vec![user!("Hello!")]),


        text::StreamOptions::default(),


    )


    .await?;


    while let Some(ev) = stream.next().await {


        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = ev {


            print!("{delta}");


        }


    }





    session.close().await;


    Ok(())


}


```





### Structured output





#### 1) Provider‑agnostic decoding (recommended for cross‑provider flows)





Use `siumai-extras` to parse model text into typed JSON with optional schema validation and repair:





```rust


use serde::Deserialize;


use siumai::prelude::unified::*;


use siumai_extras::highlevel::object::generate_object;





#[derive(Deserialize, Debug)]


struct Post { title: String }





#[tokio::main]


async fn main() -> Result<(), Box<dyn std::error::Error>> {


    let cfg = siumai::providers::openai::OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)


        .with_model("gpt-4o-mini");


    let client = siumai::providers::openai::OpenAiClient::from_config(cfg)?;


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


use siumai::prelude::unified::*;


use siumai::provider_ext::openai::options::{OpenAiChatRequestExt, OpenAiOptions, ResponsesApiConfig};


use serde_json::json;





let schema = json!({"type":"object","properties":{"title":{"type":"string"}},"required":["title"]});


let req = ChatRequestBuilder::new()


    .message(user!("Return an object with title"))


    .build()


    .with_openai_options(OpenAiOptions::new().with_responses_api(


        ResponsesApiConfig::new().with_response_format(json!({


            "type": "json_object",


            "json_schema": { "schema": schema, "strict": true }


        }))


    ));


let resp = text::generate(&client, req, text::GenerateOptions::default()).await?;


// Optionally: further validate/repair/deserialize using `siumai-extras` helpers.


```





### Retries





```rust


use siumai::prelude::unified::*;


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





1) Provider config + `HttpConfig` (most common)





```rust


use siumai::prelude::unified::*;


use siumai::providers::openai::{OpenAiClient, OpenAiConfig};





let http_cfg = HttpConfig::builder()


    .timeout(Some(std::time::Duration::from_secs(30)))


    .connect_timeout(Some(std::time::Duration::from_secs(10)))


    .user_agent(Some("my-app/1.0"))


    .header("X-User-Project", "acme")


    .stream_disable_compression(true) // keep SSE stable; default can be controlled by env


    .build();





let cfg = OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)


    .with_model("gpt-4o-mini")


    .with_http_config(http_cfg);


let client = OpenAiClient::from_config(cfg)?;


```





2) `HttpConfig` builder + shared client builder (centralized configuration)





```rust


use siumai::experimental::execution::http::client::build_http_client_from_config;


use siumai::prelude::unified::*;


use siumai::providers::openai::{OpenAiClient, OpenAiConfig};





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





let cfg = OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)


    .with_model("gpt-4o-mini")


    .with_http_config(http_cfg);


let client = OpenAiClient::new(cfg, http);


```





3) Fully custom reqwest client (maximum control)





```rust


use siumai::prelude::unified::*;


use siumai::providers::openai::{OpenAiClient, OpenAiConfig};





let http = reqwest::Client::builder()


    .timeout(std::time::Duration::from_secs(30))


    // .danger_accept_invalid_certs(true) // if needed for dev


    .build()?;





let cfg = OpenAiConfig::new(std::env::var("OPENAI_API_KEY")?)


    .with_model("gpt-4o-mini");


let client = OpenAiClient::new(cfg, http);


```





Notes:


- Streaming stability: By default, `stream_disable_compression` is derived from `SIUMAI_STREAM_DISABLE_COMPRESSION` (true unless set to `false|0|off|no`). You can override it per client using `HttpConfig::builder().stream_disable_compression(...)`.


- Builder-style HTTP toggles remain available, but they are part of the builder compatibility surface. Prefer `HttpConfig` + registry/config-first clients for new code.





Registry with custom middleware and interceptors:





```rust


use siumai::prelude::unified::*;


use siumai::experimental::execution::middleware::samples::chain_default_and_clamp;


use siumai::experimental::execution::http::interceptor::LoggingInterceptor;


use siumai::prelude::unified::registry::{create_provider_registry, RegistryOptions};


use std::collections::HashMap;


use std::sync::Arc;





let reg = create_provider_registry(


    HashMap::new(),


    Some(RegistryOptions {


        separator: ':',


        language_model_middleware: chain_default_and_clamp(),


        http_interceptors: vec![Arc::new(LoggingInterceptor)],


        http_config: None,


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


cargo run --example bedrock-chat --features bedrock


```





## Status and notes





- OpenAI Responses API `web_search` is wired through `hosted_tools::openai::web_search` and the OpenAI Responses pipeline, but is still considered experimental and may change.


- Several modules were reorganized in 0.11: HTTP helpers live under `execution::http::*`, Vertex helpers under `auth::vertex`. See CHANGELOG for migration notes.





API keys and environment variables:


- OpenAI: `.api_key(..)` or `OPENAI_API_KEY`


- Anthropic: `.api_key(..)` or `ANTHROPIC_API_KEY`


- Groq: `.api_key(..)` or `GROQ_API_KEY`


- Gemini: `.api_key(..)` or `GEMINI_API_KEY`


- Bedrock: prefer `BedrockConfig::with_region(...)` + caller-supplied SigV4 headers in `HttpConfig.headers`; `BEDROCK_API_KEY` is available only for Bearer/proxy compatibility


- xAI: `.api_key(..)` or `XAI_API_KEY`


- Ollama: no API key


- OpenAI‑compatible via Registry: reads `{PROVIDER_ID}_API_KEY` (e.g., `DEEPSEEK_API_KEY`)


- OpenAI‑compatible via Builder: `.api_key(..)` or `{PROVIDER_ID}_API_KEY`





For Bedrock-specific guidance, see `siumai/examples/04-provider-specific/bedrock/README.md`.

Compatibility note: OpenAI-compatible builder entry points remain available, but they belong to the
builder compatibility surface and are not the recommended default for new code.





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
