# Siumai Extras Examples

These examples cover optional runtime utilities that live outside the core `siumai` crate:
orchestration, MCP tool discovery, OpenTelemetry, and Axum gateway helpers.

The package uses explicit example registration, so every public example has a `required-features`
entry in `siumai-extras/Cargo.toml`.

## Orchestrator

Run these with the default `openai` feature, or pass it explicitly:

```bash
cargo run -p siumai-extras --example basic-orchestrator --features openai
cargo run -p siumai-extras --example agent-pattern --features openai
cargo run -p siumai-extras --example stop-conditions --features openai
cargo run -p siumai-extras --example tool-approval --features openai
cargo run -p siumai-extras --example streaming-orchestrator --features openai
cargo run -p siumai-extras --example streaming-tool-execution --features openai
cargo run -p siumai-extras --example workflow_planner_coder --features openai
cargo run -p siumai-extras --example orchestrator_structured_output --features openai
cargo run -p siumai-extras --example orchestrator_advanced_agent_features --features openai
```

Recommended reading order:

1. `basic-orchestrator.rs`
2. `agent-pattern.rs`
3. `stop-conditions.rs`
4. `tool-approval.rs`
5. `streaming-orchestrator.rs`
6. `streaming-tool-execution.rs`
7. `workflow_planner_coder.rs`
8. `orchestrator/structured-output.rs`
9. `orchestrator/advanced-agent-features.rs`

## MCP

```bash
MCP_SERVER_COMMAND="node mcp-server.js" \
cargo run -p siumai-extras --example mcp-stdio-tools --features "mcp,openai"

MCP_URL="http://localhost:3000/mcp" \
cargo run -p siumai-extras --example mcp-streamable-http-tools --features "mcp,openai"
```

## Server Gateways

```bash
cargo run -p siumai-extras --example openai-responses-gateway --features "server,google,openai,anthropic"
cargo run -p siumai-extras --example tool-loop-gateway --features "server,google,openai,anthropic"
cargo run -p siumai-extras --example gateway-custom-transform --features "server,google,openai"
cargo run -p siumai-extras --example gateway-loss-policy --features "server,openai,anthropic"
cargo run -p siumai-extras --example anthropic-to-openai-responses-gateway --features "server,openai,anthropic"
cargo run -p siumai-extras --example openai-responses-to-anthropic-gateway --features "server,openai,anthropic"
```

## Observability

```bash
cargo run -p siumai-extras --example opentelemetry_tracing --features "opentelemetry,openai"
```
