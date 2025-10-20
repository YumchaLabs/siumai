# Tracing Module

## 📋 Purpose

The `tracing` module provides **logging and debugging instrumentation** for the Siumai library. It uses the `tracing` crate to emit structured log events that help developers understand what's happening inside the library during development and debugging.

## 🎯 Responsibilities

### 1. **Logging Instrumentation**
- Emit structured log events using the `tracing` crate
- Provide debug-level visibility into library operations
- Support different log levels (trace, debug, info, warn, error)

### 2. **Structured Logging**
- HTTP request/response logging
- LLM interaction logging (chat messages, tool calls)
- Error tracking and classification
- Performance timing (for debugging purposes)

### 3. **Log Configuration**
- Initialize tracing subscribers (console, JSON, file output)
- Configure log levels and filters
- Support environment variable configuration

## 🔄 Relationship with Other Modules

```
┌─────────────────────────────────────────────────────────────┐
│                    Observability Stack                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   tracing    │  │  telemetry   │  │ performance  │      │
│  │              │  │              │  │              │      │
│  │  Logging &   │  │  External    │  │  Metrics &   │      │
│  │  Debugging   │  │  Exporters   │  │  Monitoring  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            │                                │
│                    ┌───────▼────────┐                       │
│                    │  Application   │                       │
│                    │  (Your Code)   │                       │
│                    └────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### vs. `telemetry`
- **tracing**: Internal logging for developers (stdout, files)
- **telemetry**: External event export to platforms (Langfuse, Helicone)

### vs. `performance`
- **tracing**: Logs timing information for debugging
- **performance**: Collects and aggregates metrics for monitoring

## 📦 Module Structure

```
tracing/
├── mod.rs              # Module entry, utilities, context
├── config.rs           # TracingConfig, OutputFormat (for internal use)
├── events.rs           # Event types (TracingEvent, HttpEvent, etc.)
├── http.rs             # HTTP request/response tracing
├── llm.rs              # LLM interaction tracing
└── performance.rs      # Performance timing helpers
```

## 🚀 Usage

### Basic Setup

For subscriber initialization, use `siumai-extras::telemetry`:

```rust
use siumai_extras::telemetry;

// Initialize with default configuration (console output, INFO level)
telemetry::init_default()?;

// Or with custom configuration
let config = telemetry::SubscriberConfig::builder()
    .log_level_str("debug")?
    .output_format(telemetry::OutputFormat::Json)
    .build();

telemetry::init_subscriber(config)?;
```

For simple cases, use `tracing-subscriber` directly:

```rust
// Simple console logging
tracing_subscriber::fmt::init();
```

### Environment Variable Configuration

```bash
# Set log level
export RUST_LOG=siumai=debug

# Or use siumai-specific variables (with siumai-extras)
export SIUMAI_LOG_LEVEL=debug
export SIUMAI_LOG_FORMAT=json
```

### Using Tracing in Code

```rust
use tracing::{info, debug, error};

// Simple logging
info!("Starting chat request");
debug!(model = "gpt-4", "Using model");

// Structured logging with fields
info!(
    provider = "openai",
    model = "gpt-4",
    tokens = 150,
    "Request completed"
);

// Error logging
error!(error = ?err, "Request failed");
```

### Tracing HTTP Requests

```rust
use siumai::tracing::ProviderTracer;

let tracer = ProviderTracer::new("openai")
    .with_model("gpt-4");

tracer.trace_request_start("POST", "https://api.openai.com/v1/chat/completions");
// ... make request ...
tracer.trace_request_complete(start_time, response_length);
```

## ⚙️ Configuration Options

### Output Formats

- **Text**: Human-readable console output (default)
- **Json**: Structured JSON logs (for log aggregation)
- **JsonCompact**: Compact JSON (one line per event)

### Log Levels

- **trace**: Very detailed, includes all events
- **debug**: Detailed debugging information
- **info**: General informational messages (default)
- **warn**: Warning messages
- **error**: Error messages only

### Features

- **HTTP Tracing**: Log all HTTP requests/responses
- **Performance Monitoring**: Log timing information
- **Sensitive Value Masking**: Automatically mask API keys and tokens
- **W3C Trace Headers**: Add traceparent headers for distributed tracing

## 🔒 Security

### Automatic Sensitive Value Masking

The tracing module automatically masks sensitive values in logs:

```rust
// API keys are automatically masked
// Input:  "sk-1234567890abcdef1234567890abcdef"
// Output: "sk-12345...cdef"

// Bearer tokens are masked
// Input:  "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
// Output: "Bearer eyJh...VCJ9"
```

### Disable Masking (for debugging only)

```rust
use siumai::tracing::set_mask_sensitive_values;

// WARNING: Only use in secure development environments
set_mask_sensitive_values(false);
```

## 📊 Event Types

### HttpEvent
- Request start/end
- HTTP method, URL, headers
- Status code, response time

### LlmEvent
- Chat requests/responses
- Tool calls
- Streaming events

### ErrorEvent
- Error classification
- Stack traces
- Retry information

### PerformanceEvent
- Request latency
- Token throughput
- Resource usage

## 🔧 Advanced Usage

### Custom Tracing Spans

```rust
use tracing::{info_span, instrument};

#[instrument(skip(client))]
async fn make_request(client: &Client, prompt: &str) -> Result<String> {
    let span = info_span!("llm_request", provider = "openai");
    let _enter = span.enter();
    
    // Your code here
    Ok("response".to_string())
}
```

### Tracing Context

```rust
use siumai::tracing::TracingContext;

let ctx = TracingContext::new("openai")
    .with_model("gpt-4")
    .with_tag("user_id".to_string(), "123".to_string());

// Use context for correlated logging
```

## 📦 Subscriber Initialization

**Note**: Subscriber initialization has been moved to `siumai-extras` as of v0.11.1.

For advanced tracing subscriber configuration, use the `siumai-extras` package with the `telemetry` feature:

```rust
use siumai_extras::telemetry;

// Initialize with default configuration
telemetry::init_default()?;

// Or with custom configuration
let config = telemetry::SubscriberConfig::builder()
    .log_level_str("debug")?
    .output_format(telemetry::OutputFormat::Json)
    .build();

telemetry::init_subscriber(config)?;
```

For simple cases, use `tracing-subscriber` directly:

```rust
// Simple console logging
tracing_subscriber::fmt::init();
```

## 📚 Related Modules

- **`telemetry/`**: External event export (Langfuse, Helicone)
- **`performance.rs`**: Metrics collection and monitoring
- **`utils/http_interceptor.rs`**: HTTP request/response interception

## 🎓 Best Practices

1. **Use appropriate log levels**
   - `debug` for detailed debugging
   - `info` for important events
   - `warn` for recoverable issues
   - `error` for failures

2. **Add structured fields**
   ```rust
   // ✅ Good: structured fields
   info!(provider = "openai", model = "gpt-4", "Request started");
   
   // ❌ Avoid: string interpolation
   info!("Request started for openai with gpt-4");
   ```

3. **Don't log sensitive data**
   - API keys, tokens, passwords
   - User personal information
   - Use masking when necessary

4. **Use spans for context**
   ```rust
   let span = info_span!("chat_request", request_id = %uuid);
   let _enter = span.enter();
   // All logs within this scope will include request_id
   ```

## 🔍 Debugging Tips

### Enable Debug Logging

```bash
# All siumai logs at debug level
RUST_LOG=siumai=debug cargo run

# Specific module
RUST_LOG=siumai::providers::openai=debug cargo run

# Multiple modules
RUST_LOG=siumai::providers=debug,siumai::executors=trace cargo run
```

### JSON Output for Analysis

Use `siumai-extras::telemetry` for JSON output:

```rust
use siumai_extras::telemetry;

let config = telemetry::SubscriberConfig::builder()
    .output_format(telemetry::OutputFormat::Json)
    .build();

telemetry::init_subscriber(config)?;
```

Then pipe to `jq` for analysis:

```bash
cargo run 2>&1 | jq 'select(.target | startswith("siumai"))'
```

## 📖 Examples

See `examples/08_telemetry/` for complete examples of using the tracing module.
