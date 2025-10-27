# Telemetry Module (Shim)

## 📋 Purpose

This module is a thin re-export shim to `observability::telemetry`.

- All implementation now lives under `siumai/src/observability/telemetry/*`.
- The public API remains available under `siumai::telemetry::*` via re-exports.

The telemetry subsystem provides **external event export** capabilities for the Siumai library. It allows you to send structured telemetry events to external observability platforms like Langfuse and Helicone for analysis, monitoring, and debugging in production environments.

## 🎯 Responsibilities

### 1. **Event Export**
- Export structured events to external platforms
- Support multiple exporters simultaneously
- Async, non-blocking event emission

### 2. **Platform Integration**
- **Langfuse**: Full-featured LLM observability platform
- **Helicone**: Request tracking and analytics via HTTP headers
- Extensible exporter interface for custom platforms

### 3. **Event Types**
- Generation events (chat completions, embeddings)
- Tool execution events (function calls)
- Orchestrator events (multi-step workflows)
- Span events (hierarchical tracing)

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
│         │                  ▼                  │             │
│         │          ┌───────────────┐          │             │
│         │          │   Langfuse    │          │             │
│         │          │   Helicone    │          │             │
│         │          │   Custom...   │          │             │
│         │          └───────────────┘          │             │
│         └──────────────────┬──────────────────┘             │
│                            │                                │
│                    ┌───────▼────────┐                       │
│                    │  Application   │                       │
│                    │  (Your Code)   │                       │
│                    └────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### vs. `tracing`
- **tracing**: Internal logging for developers (stdout, files, local debugging)
- **telemetry**: External event export to platforms (production monitoring, analytics)

### vs. `performance`
- **telemetry**: Exports events to external platforms
- **performance**: Collects metrics internally for monitoring

## 📦 Module Structure

Effective source of truth (re-export target):

```
observability/telemetry/
├── mod.rs              # Module entry, global collector
├── config.rs           # TelemetryConfig
├── events.rs           # Event types (TelemetryEvent, GenerationEvent, etc.)
└── exporters/
    ├── mod.rs          # TelemetryExporter trait
    ├── langfuse.rs     # Langfuse integration
    └── helicone.rs     # Helicone integration
```

## 🚀 Usage

### Basic Setup

```rust
use siumai::telemetry::{self, TelemetryConfig};
use siumai::telemetry::exporters::langfuse::LangfuseExporter;

// Create and register an exporter
let exporter = LangfuseExporter::new(
    "your-public-key",
    "your-secret-key",
    None, // Use default Langfuse host
);

telemetry::add_exporter(Box::new(exporter)).await;
```

### With LLM Client

```rust
use siumai::prelude::*;
use siumai::telemetry::TelemetryConfig;

let config = TelemetryConfig::builder()
    .enabled(true)
    .record_inputs(true)
    .record_outputs(true)
    .build();

let client = LlmBuilder::new()
    .openai()
    .model("gpt-4")
    .telemetry(config)
    .build()
    .await?;
```

### Manual Event Emission

```rust
use siumai::telemetry::{self, TelemetryEvent, GenerationEvent};

let event = TelemetryEvent::Generation(GenerationEvent {
    id: "gen-123".to_string(),
    model: "gpt-4".to_string(),
    input: vec![/* messages */],
    output: "response".to_string(),
    // ... other fields
});

telemetry::emit(event).await;
```

## 🔌 Exporters

### Langfuse Exporter

Langfuse is a full-featured LLM observability platform that provides:
- Trace visualization
- Cost tracking
- Prompt management
- User feedback collection

```rust
use siumai::telemetry::exporters::langfuse::LangfuseExporter;

let exporter = LangfuseExporter::new(
    "pk-lf-...",           // Public key
    "sk-lf-...",           // Secret key
    Some("https://cloud.langfuse.com"), // Optional custom host
);

telemetry::add_exporter(Box::new(exporter)).await;
```

**Features**:
- Automatic trace creation
- Span hierarchy
- Token usage tracking
- Cost calculation
- User identification

### Helicone Exporter

Helicone provides request tracking and analytics via HTTP headers:

```rust
use siumai::telemetry::exporters::helicone::HeliconeExporter;

let exporter = HeliconeExporter::new("your-api-key");

telemetry::add_exporter(Box::new(exporter)).await;
```

**Features**:
- Request logging
- Cost tracking
- Rate limiting
- Caching
- User tracking

**Note**: Helicone works by adding headers to HTTP requests, so it requires integration at the HTTP client level.

### Custom Exporter

Implement the `TelemetryExporter` trait for custom platforms:

```rust
use siumai::telemetry::{TelemetryExporter, TelemetryEvent};
use siumai::error::LlmError;

struct MyCustomExporter {
    api_key: String,
}

#[async_trait::async_trait]
impl TelemetryExporter for MyCustomExporter {
    async fn export(&self, event: &TelemetryEvent) -> Result<(), LlmError> {
        // Send event to your platform
        println!("Exporting event: {:?}", event);
        Ok(())
    }
}

// Register your exporter
telemetry::add_exporter(Box::new(MyCustomExporter {
    api_key: "your-key".to_string(),
})).await;
```

## 📊 Event Types

### GenerationEvent

Represents a single LLM generation（chat/embedding 等）。

```rust
pub struct GenerationEvent {
    pub id: String,
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub timestamp: SystemTime,
    pub provider: String,
    pub model: String,
    pub input: Option<Vec<ChatMessage>>,   // 受 TelemetryConfig.record_inputs 控制
    pub output: Option<ChatResponse>,      // 受 TelemetryConfig.record_outputs 控制
    pub usage: Option<Usage>,
    pub finish_reason: Option<FinishReason>,
    pub duration: Option<Duration>,
    pub metadata: HashMap<String, String>,
    pub error: Option<String>,
}
```

### ToolExecutionEvent

表示一次工具调用（函数调用）：

```rust
pub struct ToolExecutionEvent {
    pub id: String,
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub timestamp: SystemTime,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub arguments: Option<serde_json::Value>,
    pub result: Option<String>,
    pub duration: Option<Duration>,
    pub error: Option<String>,
    pub metadata: HashMap<String, String>,
}
```

### OrchestratorEvent

代表多步管弦（编排）事件：

```rust
pub struct OrchestratorEvent {
    pub id: String,
    pub trace_id: String,
    pub timestamp: SystemTime,
    pub total_steps: usize,
    pub current_step: usize,
    pub step_type: OrchestratorStepType,
    pub total_usage: Option<Usage>,
    pub total_duration: Option<Duration>,
    pub metadata: HashMap<String, String>,
}
```

### SpanEvent

代表层级跨度（分布式追踪）：

```rust
pub struct SpanEvent {
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub trace_id: String,
    pub name: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub duration: Option<Duration>,
    pub attributes: HashMap<String, String>,
    pub status: SpanStatus,
    pub error: Option<String>,
}
```

## ⚙️ Configuration

### TelemetryConfig

```rust
use siumai::telemetry::TelemetryConfig;

let config = TelemetryConfig::builder()
    .enabled(true)              // Enable telemetry
    .record_inputs(true)        // Record input messages
    .record_outputs(true)       // Record output responses
    .record_metadata(true)      // Record metadata
    .sample_rate(1.0)           // Sample 100% of requests
    .build();
```

### Configuration Options

- enabled: Enable/disable telemetry globally
- record_inputs: Whether to record input messages
- record_outputs: Whether to record output responses
- record_tools: Whether to record tool calls
- record_usage: Whether to record token usage
- function_id: Optional grouping identifier
- metadata/tags/session_id/user_id: Optional contextual fields

## 🔒 Privacy and Security

### Sensitive Data Handling

**Important**: Telemetry events may contain sensitive data (user messages, API responses). Consider:

1. **Data Minimization**: Only record what you need
   ```rust
   let config = TelemetryConfig::builder()
       .enabled(true)
       .record_inputs(false)  // Don't record user messages
       .record_outputs(false) // Don't record AI responses
       .record_usage(true)
       .build();
   ```

2. **Sampling**: Reduce data volume
   ```rust
   // Sampling is not built-in; consider filtering in your exporter
   // or wrap telemetry::emit to downsample events.
   ```

3. **Data Scrubbing**: Implement custom exporters that scrub sensitive data

4. **Compliance**: Ensure your telemetry practices comply with:
   - GDPR (EU)
   - CCPA (California)
   - HIPAA (Healthcare)
   - Other relevant regulations

### API Key Security

- Store API keys in environment variables
- Use secrets management systems (AWS Secrets Manager, HashiCorp Vault)
- Never commit API keys to version control

```rust
use std::env;

let langfuse_key = env::var("LANGFUSE_PUBLIC_KEY")
    .expect("LANGFUSE_PUBLIC_KEY not set");
let langfuse_secret = env::var("LANGFUSE_SECRET_KEY")
    .expect("LANGFUSE_SECRET_KEY not set");

let exporter = LangfuseExporter::new(&langfuse_key, &langfuse_secret, None);
```

## 🎓 Best Practices

### 1. Use Trace IDs for Correlation

```rust
use uuid::Uuid;

let trace_id = Uuid::new_v4().to_string();

// Pass trace_id through your application
let event = GenerationEvent {
    trace_id: Some(trace_id.clone()),
    // ... other fields
};
```

### 2. Add Metadata for Context

```rust
let mut metadata = HashMap::new();
metadata.insert("user_id".to_string(), json!("user-123"));
metadata.insert("session_id".to_string(), json!("session-456"));
metadata.insert("environment".to_string(), json!("production"));

let event = GenerationEvent {
    metadata,
    // ... other fields
};
```

### 3. Handle Export Failures Gracefully

Telemetry export failures should not break your application:

```rust
// The telemetry module automatically handles failures
// and logs warnings without crashing your app
telemetry::emit(event).await; // Won't panic on failure
```

### 4. Use Sampling in Production

```rust
let config = TelemetryConfig::builder()
    .sample_rate(0.1)  // 10% sampling for high-volume apps
    .build();
```

## 📈 Monitoring and Debugging

### Check if Telemetry is Enabled

```rust
if telemetry::is_enabled().await {
    println!("Telemetry is active");
}
```

### Clear Exporters (for testing)

```rust
// Remove all exporters
telemetry::clear_exporters().await;
```

### Multiple Exporters

You can register multiple exporters simultaneously:

```rust
// Send to both Langfuse and Helicone
telemetry::add_exporter(Box::new(langfuse_exporter)).await;
telemetry::add_exporter(Box::new(helicone_exporter)).await;
telemetry::add_exporter(Box::new(custom_exporter)).await;
```

## 🔧 Advanced Usage

### Hierarchical Tracing

```rust
use siumai::telemetry::{SpanEvent, TelemetryEvent};

// Parent span
let parent_span = SpanEvent {
    id: "span-1".to_string(),
    trace_id: "trace-123".to_string(),
    parent_span_id: None,
    name: "orchestrator".to_string(),
    // ... other fields
};

telemetry::emit(TelemetryEvent::Span(parent_span)).await;

// Child span
let child_span = SpanEvent {
    id: "span-2".to_string(),
    trace_id: "trace-123".to_string(),
    parent_span_id: Some("span-1".to_string()),
    name: "llm_call".to_string(),
    // ... other fields
};

telemetry::emit(TelemetryEvent::Span(child_span)).await;
```

## 📚 Related Modules

- **`tracing/`**: Internal logging and debugging
- **`performance.rs`**: Metrics collection and monitoring
- **`orchestrator/`**: Multi-step workflows that emit telemetry events

## 📖 Examples

See `examples/08_telemetry/` for complete examples of using the telemetry module with Langfuse and Helicone.

## 🔗 External Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Helicone Documentation](https://docs.helicone.ai)
- [OpenTelemetry](https://opentelemetry.io) (for future integration)
