# Changelog - siumai-extras

All notable changes to the `siumai-extras` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Runnable loss-policy bridge example:
  - `siumai-extras/examples/gateway-loss-policy.rs`
  - demonstrates default strict rejection vs custom allowlisted / continue policies for lossy
    JSON and SSE bridge routes

### Changed

- Gateway bridge customization now recommends `GatewayBridgePolicy + BridgeOptions + typed bridge
  hooks` as the primary extension path for Axum SSE/JSON transcoders.
- Axum gateway transcode helpers now support partial bridge overrides and route-level bridge-mode
  override without rebuilding a full `BridgeOptions` value.
- Axum SSE transcode helpers now support source-aware inspected loss-policy rejection and warning /
  decision headers via `TranscodeSseOptions::with_bridge_source(...)`.
- Axum SSE transcode helpers now enforce policy-driven keepalive and idle-timeout behavior when
  configured.
- Axum server adapters now expose policy-aware request/upstream body read helpers for enforcing
  request-body and upstream-read limits at route/runtime level.
- Orchestrator/agent/workflow public model bounds now use `LanguageModel` instead of bare
  `ChatCapability`, so extras step/result surfaces can rely on stable provider/model metadata
  instead of heuristic response inspection.
- Orchestrator/agent/workflow now also expose one canonical AI SDK-style runtime `context`
  object across prepare-step, tool execution, step results, and finish callbacks.

### Fixed

- `stream_object` and the extras tool-loop gateway now consume the upgraded stable
  `ChatStreamEvent::Part` tool lifecycle directly instead of depending on legacy delta/custom-only
  accumulation.
- `stream_object`, tool-loop assistant-history accumulation, and streamed orchestrator fallback
  now also consume stable `Part(TextDelta)` directly instead of depending on legacy
  `ContentDelta` shadows.
- The extras tool-loop gateway now also emits stable runtime `Part(ToolResult)` between local
  tool-execution steps while keeping the legacy `gateway:tool-result` custom event for
  compatibility, so downstream protocol serializers can prefer the semantic lane.
- The Axum SSE helper now forwards stable runtime `Part` / `PartWithReplay` events as explicit
  `event: part` frames, and both now use one stable `{ part, replay }` JSON envelope (`replay:
  null` when absent), so the AI-SDK-aligned semantic stream lane is observable outside the core
  crate without event-kind-dependent payload shape drift.
- The Axum plain-text helper `to_text_stream()` now also reads stable `Part(TextDelta)` /
  `PartWithReplay(TextDelta)` events instead of depending only on legacy `ContentDelta`.
- Gateway bridge samples, Axum transcode tests, and bridge customization guidance now mutate
  stable `Part(TextDelta)` / `PartWithReplay(TextDelta)` directly instead of assuming the legacy
  `ContentDelta` shadow lane is always present.
- Orchestrator step-usage aggregation now follows AI SDK `totalUsage` semantics more closely:
  `StepResult::merge_usage()` / `AgentResult::total_usage()` still sum token/accounting fields
  across steps, but aggregated results no longer preserve per-step provider-native `Usage.raw`,
  including the single-step case.
- Orchestrator/agent/workflow finish callbacks now expose an explicit AI-SDK-style completion
  event carrying the final response, last step, all `steps`, and aggregated `total_usage`.
- `StreamOrchestration` now exposes aggregated `total_usage`, and the basic streaming path now
  records a real final step instead of resolving an empty `steps` list.
- `StepResult` now also carries stable `call_id`, stable `model { provider, model_id }`,
  explicit `step_number`, step-scoped `request` / `response`, telemetry `function_id` /
  `metadata`, and stable `raw_finish_reason` sourced from `ChatResponse.raw_finish_reason`,
  closing another structural gap against AI SDK step metadata without fabricating provider
  raw-finish values.
- `PrepareStepContext` now exposes the base `LanguageModel`, `PrepareStepResult` can swap the
  model for one step via `with_model(...)`, and both non-stream and stream orchestration now honor
  that override without leaking it into later steps.
- `StepResult.text()` now concatenates all top-level text parts from unified step content, and
  standardized projections now expose `tool_call_views()` / `tool_result_views()` plus
  `static_*` / `dynamic_*` splits with resolved tool inputs for results.
- Streaming orchestration now applies `prepare_step`, `tool_choice`, `active_tools`, and
  runtime `context` on the first streamed step too, and context-aware tool execution remains
  backward-compatible because `ToolResolver::{call_tool_with_context, call_tool_stream_with_context}`
  default back to the legacy methods.

## [0.11.0-beta.5] - 2026-01-15

### Added

- OpenAI Responses SSE gateway helpers (Axum)
  - `siumai_extras::server::axum::to_openai_responses_sse_stream(...)`
  - `siumai_extras::server::axum::to_openai_responses_sse_response(...)`
- Unified SSE transcoder helper (Axum)
  - `siumai_extras::server::axum::{to_transcoded_sse_response, TargetSseFormat, TranscodeSseOptions}`
- Unified SSE transcoder transform hook (Axum)
  - `siumai_extras::server::axum::to_transcoded_sse_response_with_transform(...)`
- Unified JSON (non-streaming) transcoder helper (Axum)
  - `siumai_extras::server::axum::{transcode_chat_response_to_json, to_transcoded_json_response, to_transcoded_json_response_with_transform, TargetJsonFormat, TranscodeJsonOptions}`
  - Backed by protocol-level encoders (`siumai::experimental::encoding::JsonResponseConverter`) for lower overhead than `serde_json::Value` round-trips.
- Runnable gateway example (Gemini backend, OpenAI Responses SSE output)
  - `siumai-extras/examples/openai-responses-gateway.rs`
    - Added non-streaming JSON endpoints (`*.json`)
- Runnable custom conversion example (stream + JSON)
  - `siumai-extras/examples/gateway-custom-transform.rs`
- Tool-loop gateway helper (keep one downstream stream open across tool calls)
  - `siumai_extras::server::tool_loop::tool_loop_chat_stream(...)`
- Runnable multi-protocol tool-loop gateway example
  - `siumai-extras/examples/tool-loop-gateway.rs`

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
use siumai::prelude::unified::*;
use siumai_extras::mcp::mcp_tools_from_stdio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to MCP server
    let (tools, resolver) = mcp_tools_from_stdio("node mcp-server.js").await?;
    
    // Create model
    let reg = registry::global();
    let model = reg.language_model("openai:gpt-4o-mini")?;
    
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
