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

### MCP Integration

```rust
use siumai_extras::mcp::mcp_tools_from_stdio;

// Connect to MCP server and get tools
let (tools, resolver) = mcp_tools_from_stdio("node mcp-server.js").await?;

// Use with Siumai orchestrator
let (response, _) = siumai::orchestrator::generate(
    &model,
    messages,
    Some(tools),
    Some(&resolver),
    vec![siumai::orchestrator::step_count_is(10)],
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

