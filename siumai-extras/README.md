# Siumai Extras

Optional utilities for the [siumai](https://github.com/YumchaLabs/siumai) LLM library.

## Features

This crate provides optional functionality that extends `siumai` without adding heavy dependencies to the core library:

- **`schema`** - JSON Schema validation for structured outputs
- **`telemetry`** - Advanced tracing and logging with `tracing-subscriber`
- **`server`** - Server adapters for Axum and other web frameworks
- **`mcp`** - MCP (Model Context Protocol) integration for dynamic tool discovery
- **`all`** - Enable all features

## Installation

```toml
[dependencies]
siumai = "0.11"
siumai-extras = { version = "0.11", features = ["schema", "telemetry", "mcp"] }
```

## Usage

> Orchestrator and high-level object helpers do **not** require any extra
> features. Schema validation and tracing are opt-in via the `schema` and
> `telemetry` features.

### High-level structured objects

Provider-agnostic helpers for generating typed JSON objects:

```rust
use serde::Deserialize;
use siumai::prelude::*;
use siumai_extras::highlevel::object::{generate_object, GenerateObjectOptions};

#[derive(Deserialize, Debug)]
struct Post { title: String }

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .model("gpt-4o-mini")
        .build()
        .await?;

    let (post, _resp) = generate_object::<Post>(
        &client,
        vec![user!("Return JSON: {\"title\":\"hi\"}")],
        None,
        GenerateObjectOptions::default(),
    )
    .await?;

    println!("{}", post.title);
    Ok(())
}
```

If you enable the `schema` feature, `GenerateObjectOptions::schema` is
validated via `siumai_extras::schema` before deserializing into `T`.

### Orchestrator & agents

Multi-step tool calling, agents, and stop conditions:

```rust
use serde_json::json;
use siumai::prelude::*;
use siumai_extras::orchestrator::{
    ToolLoopAgent, ToolResolver, ToolChoice, step_count_is,
};

struct SimpleResolver;

#[async_trait::async_trait]
impl ToolResolver for SimpleResolver {
    async fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, siumai::error::LlmError> {
        match name {
            "get_weather" => {
                let city = args.get("city").and_then(|v| v.as_str()).unwrap_or("Unknown");
                Ok(json!({ "city": city, "temperature": 72, "condition": "sunny" }))
            }
            _ => Err(siumai::error::LlmError::InternalError(format!(
                "Unknown tool: {}",
                name
            ))),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .model("gpt-4o-mini")
        .build()
        .await?;

    let weather_tool = Tool::function(
        "get_weather",
        "Get weather for a city",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    );

    let agent = ToolLoopAgent::new(client, vec![weather_tool], vec![step_count_is(10)])
        .with_system("You are a helpful assistant.")
        .with_tool_choice(ToolChoice::Required);

    let messages = vec![ChatMessage::user("What's the weather in Tokyo?").build()];
    let resolver = SimpleResolver;

    let result = agent.generate(messages, &resolver).await?;
    println!("Answer: {}", result.text().unwrap_or_default());
    Ok(())
}
```

You can attach telemetry to the agent or orchestrator using
`siumai::experimental::observability::telemetry::TelemetryConfig`:

```rust
use siumai::experimental::observability::telemetry::TelemetryConfig;
use siumai_extras::orchestrator::OrchestratorBuilder;

let telemetry = TelemetryConfig::builder()
    .record_inputs(false)
    .record_outputs(false)
    .record_usage(true)
    .build();

let builder = OrchestratorBuilder::new().telemetry(telemetry);
```

### Schema Validation

```rust
use siumai_extras::schema::SchemaValidator;

// Validate JSON against a schema
let validator = SchemaValidator::new(schema)?;
validator.validate(&json_value)?;
```

### Telemetry

```rust
use siumai_extras::telemetry::init_subscriber;

// Initialize tracing subscriber
init_subscriber(config)?;
```

### Server Adapters

```rust
use siumai_extras::server::axum::to_sse_response;

// Convert ChatStream to Axum SSE response
let sse = to_sse_response(stream, options);
```

If you are building an OpenAI-compatible gateway and need to output **OpenAI Responses SSE**,
`siumai-extras` also provides a helper that:
- bridges provider-specific `ChatStreamEvent::Custom` parts into `openai:*` stream parts, and
- serializes the stream into OpenAI Responses SSE frames.

```rust
use axum::response::Response;
use axum::body::Body;
use siumai_extras::server::axum::to_openai_responses_sse_response;
use siumai::prelude::unified::ChatStream;

fn handler(stream: ChatStream) -> Response<Body> {
    to_openai_responses_sse_response(stream)
}
```

See the runnable example: `siumai-extras/examples/openai-responses-gateway.rs` (streaming + non-streaming).
For custom conversion hooks, see: `siumai-extras/examples/gateway-custom-transform.rs`.

If you need to expose multiple downstream protocol surfaces from the same upstream stream,
use the transcoder helper:

```rust
use axum::{body::Body, response::Response};
use siumai::prelude::unified::ChatStream;
use siumai_extras::server::axum::{
    TargetSseFormat, TranscodeSseOptions, to_transcoded_sse_response,
};

fn handler(stream: ChatStream) -> Response<Body> {
    to_transcoded_sse_response(stream, TargetSseFormat::OpenAiResponses, TranscodeSseOptions::strict())
}
```

If you need to customize the conversion logic (redaction/rewrites/custom part mapping),
you can provide an event transform hook:

```rust
use axum::{body::Body, response::Response};
use siumai::prelude::unified::{ChatStream, ChatStreamEvent};
use siumai_extras::server::axum::{
    TargetSseFormat, TranscodeSseOptions, to_transcoded_sse_response_with_transform,
};

fn handler(stream: ChatStream) -> Response<Body> {
    to_transcoded_sse_response_with_transform(
        stream,
        TargetSseFormat::OpenAiResponses,
        TranscodeSseOptions::strict(),
        |ev: ChatStreamEvent| vec![ev],
    )
}
```

For non-streaming gateways, you can also transcode a `ChatResponse` into a provider-native
JSON response body:

```rust
use axum::{body::Body, response::Response};
use siumai::prelude::*;
use siumai_extras::server::axum::{
    TargetJsonFormat, TranscodeJsonOptions, to_transcoded_json_response,
};

fn handler(resp: ChatResponse) -> Response<Body> {
    to_transcoded_json_response(resp, TargetJsonFormat::OpenAiResponses, TranscodeJsonOptions::default())
}
```

If you want to customize conversion for non-streaming responses, prefer the response-level transform
hook (no JSON parse/round-trip):

```rust
use axum::{body::Body, response::Response};
use siumai::prelude::*;
use siumai_extras::server::axum::{
    TargetJsonFormat, TranscodeJsonOptions, to_transcoded_json_response_with_response_transform,
};

fn handler(resp: ChatResponse) -> Response<Body> {
    to_transcoded_json_response_with_response_transform(
        resp,
        TargetJsonFormat::OpenAiResponses,
        TranscodeJsonOptions::default(),
        |r| {
            r.content = MessageContent::Text("[REDACTED]".to_string());
        },
    )
}
```

### MCP Integration

```rust
use siumai_extras::mcp::mcp_tools_from_stdio;

// Connect to MCP server and get tools
let (tools, resolver) = mcp_tools_from_stdio("node mcp-server.js").await?;

// Use with Siumai orchestrator (from siumai-extras)
let (response, _) = siumai_extras::orchestrator::generate(
    &model,
    messages,
    Some(tools),
    Some(&resolver),
    vec![siumai_extras::orchestrator::step_count_is(10)],
    Default::default(),
).await?;
```

## Documentation

- [MCP Feature Guide](./docs/MCP_FEATURE.md)
- [Siumai MCP Integration Guide](../siumai/docs/guides/MCP_INTEGRATION.md)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
