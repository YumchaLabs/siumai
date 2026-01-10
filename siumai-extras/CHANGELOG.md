# Changelog - siumai-extras

All notable changes to the `siumai-extras` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0-beta.5] - Unreleased

### Added

- OpenAI Responses SSE gateway helpers (Axum)
  - `siumai_extras::server::axum::to_openai_responses_sse_stream(...)`
  - `siumai_extras::server::axum::to_openai_responses_sse_response(...)`
- Unified SSE transcoder helper (Axum)
  - `siumai_extras::server::axum::{to_transcoded_sse_response, TargetSseFormat, TranscodeSseOptions}`
- Runnable gateway example (Gemini backend, OpenAI Responses SSE output)
  - `siumai-extras/examples/openai-responses-gateway.rs`

### Changed

- Workspace version alignment for the `.5` split phase (no siumai-extras-specific behavior changes).

## [0.11.0-beta.4] - 2025-12-09

### Added

#### MCP Integration (NEW)

- **`mcp` feature**: MCP (Model Context Protocol) integration for dynamic tool discovery
  - **Core Implementation**:
    - `McpToolResolver`: Implements Siumai's `ToolResolver` trait for MCP tools
    - Support for stdio, SSE, and HTTP transports via `rmcp` library (v0.8)
    - Automatic tool discovery from MCP servers
    - Type-safe tool execution with compile-time guarantees
  - **Convenience Functions**:
    - `mcp_tools_from_stdio(command)`: Connect to local MCP servers via stdio
    - `mcp_tools_from_sse(url)`: Connect to remote MCP servers via SSE
    - `mcp_tools_from_http(url)`: Connect to MCP servers via HTTP
  - **Integration**:
    - Seamless integration with Siumai's orchestrator
    - Works with all Siumai-supported LLM providers (OpenAI, Anthropic, Google, etc.)
    - Compatible with `ToolLoopAgent` for reusable agent patterns
  - **Documentation**:
    - Complete integration guide: `siumai/docs/guides/MCP_INTEGRATION.md`
    - API reference: `siumai-extras/docs/MCP_FEATURE.md`
    - Examples: `siumai/examples/05-integrations/mcp/`
  - **Design Philosophy**:
    - External integration (not in core library) following Vercel AI SDK's pattern
    - Keeps core library lightweight and fast to compile
    - Optional feature that users can opt-in as needed
    - Based on industry-standard MCP protocol

#### Schema Validation

- **`schema` feature**: JSON schema validation using `jsonschema` crate
  - Moved from core `siumai` package to reduce core dependencies
  - Provides `validate_json_schema()` function for validating JSON against schemas
  - Optional feature that users can opt-in as needed

#### Telemetry

- **`telemetry` feature**: Advanced tracing subscriber configuration
  - Moved from core `siumai` package to reduce core dependencies
  - Provides initialization functions:
    - `init_default()`: Initialize with default configuration
    - `init_debug()`: Initialize with debug-level logging
    - `init_production()`: Initialize with production-level logging
    - `init_performance()`: Initialize with performance monitoring
    - `init_from_env()`: Initialize from environment variables
    - `init(config)`: Initialize with custom configuration
  - Supports multiple output formats: Pretty, Compact, JSON
  - Supports multiple output targets: Stdout, File, Both
  - Complete documentation in `siumai/src/telemetry/README.md`

#### Server Adapters

- **`server` feature**: Server adapters for web frameworks
  - Moved from core `siumai` package to reduce core dependencies
  - Axum adapter: `to_sse_response()` for converting Siumai streams to SSE responses
  - Optional feature that users can opt-in as needed

### Changed

- **Package Structure**: Created as a separate package in the workspace
  - Located in `siumai-extras/` directory
  - Shares workspace dependencies with core `siumai` package
  - Provides optional features that users can opt-in as needed

### Migration Guide

#### Enabling Features

Add `siumai-extras` to your `Cargo.toml` with the features you need:

```toml
[dependencies]
siumai = { version = "0.11", features = ["openai"] }
siumai-extras = { version = "0.11", features = ["mcp", "telemetry", "schema", "server"] }
```

Or enable all features:

```toml
[dependencies]
siumai-extras = { version = "0.11", features = ["all"] }
```

#### MCP Integration

```rust
use siumai::prelude::*;
use siumai_extras::mcp::mcp_tools_from_stdio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to MCP server
    let (tools, resolver) = mcp_tools_from_stdio("node mcp-server.js").await?;
    
    // Create model
    let model = Siumai::builder().openai().build().await?;
    
    // Use with orchestrator
    let messages = vec![user!("Use the available tools to help me")];
    let (response, _) = siumai_extras::orchestrator::generate(
        &model,
        messages,
        Some(tools),
        Some(&resolver),
        vec![siumai_extras::orchestrator::step_count_is(10)],
        Default::default(),
    ).await?;
    
    println!("Response: {}", response.content_text().unwrap());
    Ok(())
}
```

#### Telemetry

**Before (v0.10.3 and earlier):**
```rust
use siumai::tracing::{init_default_tracing, TracingConfig};
init_default_tracing()?;
```

**After (v0.11.0):**
```rust
use siumai_extras::telemetry;
telemetry::init_default()?;
```

#### Schema Validation

```rust
use siumai_extras::schema::validate_json_schema;

let schema = serde_json::json!({
    "type": "object",
    "properties": {
        "name": { "type": "string" }
    }
});

let data = serde_json::json!({
    "name": "Alice"
});

validate_json_schema(&data, &schema)?;
```

#### Server Adapters

**Before:**
```rust
use siumai::server_adapters::axum::to_sse_response;
```

**After:**
```rust
use siumai_extras::server::axum::to_sse_response;
```

## Features

- **`schema`**: JSON schema validation using `jsonschema` crate
- **`telemetry`**: Advanced tracing subscriber configuration with `tracing-subscriber`
- **`server`**: Server adapters for Axum and other web frameworks
- **`mcp`**: MCP (Model Context Protocol) integration for dynamic tool discovery
- **`all`**: Enable all features

## Dependencies

### Core Dependencies
- `siumai` (v0.11.0): Core Siumai library
- `serde`, `serde_json`: JSON serialization
- `thiserror`: Error handling

### Optional Dependencies
- `jsonschema` (feature: `schema`): JSON schema validation
- `tracing`, `tracing-subscriber`, `tracing-appender` (feature: `telemetry`): Tracing and logging
- `axum`, `futures` (feature: `server`): Server adapters
- `rmcp`, `async-trait` (feature: `mcp`): MCP integration

## Documentation

- **MCP Integration**:
  - Integration guide: `siumai/docs/guides/MCP_INTEGRATION.md`
  - API reference: `siumai-extras/docs/MCP_FEATURE.md`
  - Examples: `siumai/examples/05-integrations/mcp/`
- **Telemetry**: `siumai/src/telemetry/README.md`
- **API Documentation**: https://docs.rs/siumai-extras

## License

Licensed under the same license as the main `siumai` package.
