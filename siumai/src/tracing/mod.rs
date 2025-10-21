//! Tracing Module - Logging and Debugging Instrumentation
//!
//! ## 📋 Purpose
//!
//! This module provides **logging and debugging instrumentation** for the Siumai library.
//! It uses the `tracing` crate to emit structured log events that help developers understand
//! what's happening inside the library during development and debugging.
//!
//! ## 🎯 Responsibilities
//!
//! 1. **Logging Instrumentation**: Emit structured log events using the `tracing` crate
//! 2. **Structured Logging**: HTTP requests/responses, LLM interactions, errors, performance timing
//! 3. **Tracing Context**: Provide context for correlated logging across requests
//!
//! ## 🔄 Relationship with Other Modules
//!
//! - **vs. `telemetry/`**: `tracing` = internal logging for developers (stdout, files);
//!   `telemetry` = external event export to platforms (Langfuse, Helicone)
//! - **vs. `performance`**: `tracing` = logs timing info for debugging;
//!   `performance` = collects and aggregates metrics for monitoring
//!
//! ## 📚 Documentation
//!
//! See `src/tracing/README.md` for detailed documentation and examples.
//!
//! ## 🚀 Quick Start
//!
//! For subscriber initialization, use `siumai-extras::telemetry`:
//!
//! ```rust,ignore
//! use siumai_extras::telemetry;
//!
//! // Initialize with default configuration
//! telemetry::init_default()?;
//!
//! // Or with custom configuration
//! let config = telemetry::SubscriberConfig::builder()
//!     .log_level_str("debug")?
//!     .output_format(telemetry::OutputFormat::Json)
//!     .build();
//!
//! telemetry::init_subscriber(config)?;
//! ```
//!
//! For simple cases, use `tracing-subscriber` directly:
//!
//! ```rust,ignore
//! // Simple console logging (requires adding `tracing-subscriber` dependency)
//! // tracing_subscriber::fmt::init();
//! ```

pub mod config;
pub mod events;
pub mod http;
pub mod llm;
pub mod performance;

// Re-export main types
pub use config::{OutputFormat, TracingConfig, TracingConfigBuilder};
pub use events::{
    ChatEvent, ErrorEvent, HttpEvent, LlmEvent, PerformanceEvent, StreamEvent, ToolEvent,
    TracingEvent,
};
pub use http::{HttpTracer, RequestContext, ResponseContext};
pub use llm::{ChatTracer, LlmTracer, StreamTracer, ToolTracer};
pub use performance::{PerformanceTracer, TimingContext};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime};
use tracing::{Span, debug, error, info};
use uuid::Uuid;

/// Unique identifier for a tracing session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceId(pub Uuid);

impl TraceId {
    /// Generate a new trace ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Span identifier for hierarchical tracing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(pub Uuid);

impl SpanId {
    /// Generate a new span ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SpanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Context information for tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingContext {
    /// Unique trace identifier
    pub trace_id: TraceId,
    /// Current span identifier
    pub span_id: SpanId,
    /// Parent span identifier (if any)
    pub parent_span_id: Option<SpanId>,
    /// Provider name (e.g., "openai", "anthropic")
    pub provider: String,
    /// Model name
    pub model: Option<String>,
    /// User-defined tags
    pub tags: HashMap<String, String>,
    /// Session start time
    pub session_start: SystemTime,
}

impl TracingContext {
    /// Create a new tracing context
    pub fn new(provider: String) -> Self {
        Self {
            trace_id: TraceId::new(),
            span_id: SpanId::new(),
            parent_span_id: None,
            provider,
            model: None,
            tags: HashMap::new(),
            session_start: SystemTime::now(),
        }
    }

    /// Create a child context with a new span
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id,
            span_id: SpanId::new(),
            parent_span_id: Some(self.span_id),
            provider: self.provider.clone(),
            model: self.model.clone(),
            tags: self.tags.clone(),
            session_start: self.session_start,
        }
    }

    /// Set the model name
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Get session duration
    pub fn session_duration(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.session_start)
            .unwrap_or_default()
    }
}

/// Tracing utilities
pub struct TracingUtils;

impl TracingUtils {
    /// Create a tracing span with context
    pub fn create_span(_name: &'static str, _context: &TracingContext) -> Span {
        // For now, return a simple span - this would need proper implementation
        tracing::info_span!("siumai_operation")
    }

    /// Extract trace context from current span
    pub fn current_context() -> Option<TracingContext> {
        // This would extract context from the current tracing span
        // Implementation depends on how we store context in spans
        None
    }

    /// Format duration for human readability
    pub fn format_duration(duration: Duration) -> String {
        if duration.as_secs() > 0 {
            format!("{:.2}s", duration.as_secs_f64())
        } else if duration.as_millis() > 0 {
            format!("{}ms", duration.as_millis())
        } else {
            format!("{}μs", duration.as_micros())
        }
    }

    /// Format bytes for human readability
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }
}

/// Macro for creating traced HTTP requests
#[macro_export]
macro_rules! traced_http_request {
    ($tracer:expr, $method:expr, $url:expr, $body:expr) => {{
        let context = $tracer.start_request($method, $url);
        // Request execution would go here
        $tracer.end_request(context /* response data */);
    }};
}

/// Macro for creating traced LLM interactions
#[macro_export]
macro_rules! traced_llm_chat {
    ($tracer:expr, $messages:expr, $tools:expr) => {{
        let context = $tracer.start_chat($messages, $tools);
        // Chat execution would go here
        $tracer.end_chat(context /* response */);
    }};
}

/// Global flag for pretty JSON formatting in tracing
static PRETTY_JSON: AtomicBool = AtomicBool::new(false);

/// Global flag for masking sensitive values in tracing
static MASK_SENSITIVE_VALUES: AtomicBool = AtomicBool::new(true);

/// Global flag for enabling W3C trace headers (traceparent)
static W3C_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Set the global pretty JSON flag
pub fn set_pretty_json(pretty: bool) {
    PRETTY_JSON.store(pretty, Ordering::Relaxed);
}

/// Get the global pretty JSON flag
pub fn get_pretty_json() -> bool {
    PRETTY_JSON.load(Ordering::Relaxed)
}

/// Set the global mask sensitive values flag
pub fn set_mask_sensitive_values(mask: bool) {
    MASK_SENSITIVE_VALUES.store(mask, Ordering::Relaxed);
}

/// Enable or disable W3C trace headers at runtime
pub fn set_w3c_trace_enabled(enabled: bool) {
    W3C_TRACE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Check whether W3C trace headers are enabled
pub fn w3c_trace_enabled() -> bool {
    W3C_TRACE_ENABLED.load(Ordering::Relaxed)
}

/// Build a W3C traceparent header value using newly generated IDs
pub fn create_w3c_traceparent() -> String {
    // version 00 - 32 hex trace-id - 16 hex span-id - flags 01
    let trace_id = uuid_hex32(Uuid::new_v4());
    let span_id = uuid_hex16(Uuid::new_v4());
    format!("00-{}-{}-01", trace_id, span_id)
}

fn uuid_hex32(u: Uuid) -> String {
    // 16 bytes => 32 hex
    let b = u.as_bytes();
    bytes_to_hex(b)
}

fn uuid_hex16(u: Uuid) -> String {
    // take first 8 bytes => 16 hex
    let b = &u.as_bytes()[..8];
    bytes_to_hex(b)
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

/// Get the global mask sensitive values flag
pub fn get_mask_sensitive_values() -> bool {
    MASK_SENSITIVE_VALUES.load(Ordering::Relaxed)
}

/// Format JSON for logging based on global configuration
pub fn format_json_for_logging(value: &serde_json::Value) -> String {
    if get_pretty_json() {
        serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
    } else {
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
    }
}

/// Mask sensitive values in strings for security
pub fn mask_sensitive_value(value: &str) -> String {
    if !get_mask_sensitive_values() {
        return value.to_string();
    }

    // Check if this looks like an API key or token
    if let Some(token) = value.strip_prefix("Bearer ")
        && token.len() > 8
    {
        return format!("Bearer {}...{}", &token[..4], &token[token.len() - 4..]);
    }

    // Check for API key patterns
    if (value.starts_with("sk-") || value.starts_with("sk-ant-") || value.starts_with("gsk-"))
        && value.len() > 12
    {
        return format!("{}...{}", &value[..8], &value[value.len() - 4..]);
    }

    // For other potentially sensitive values
    if value.len() > 16 {
        format!("{}...{}", &value[..6], &value[value.len() - 4..])
    } else {
        value.to_string()
    }
}

/// Format headers for logging with sensitive value masking
pub fn format_headers_for_logging(headers: &reqwest::header::HeaderMap) -> String {
    let header_map: std::collections::HashMap<&str, String> = headers
        .iter()
        .map(|(k, v)| {
            let value = v.to_str().unwrap_or("<invalid>");
            let masked_value = if k.as_str().to_lowercase().contains("authorization")
                || k.as_str().to_lowercase().contains("key")
                || k.as_str().to_lowercase().contains("token")
            {
                mask_sensitive_value(value)
            } else {
                value.to_string()
            };
            (k.as_str(), masked_value)
        })
        .collect();

    if get_pretty_json() {
        serde_json::to_string_pretty(&header_map).unwrap_or_else(|_| format!("{header_map:?}"))
    } else {
        serde_json::to_string(&header_map).unwrap_or_else(|_| format!("{header_map:?}"))
    }
}

/// Unified provider tracing utility
///
/// This provides a consistent tracing interface across all providers,
/// ensuring uniform logging format and behavior.
pub struct ProviderTracer {
    provider: String,
    model: Option<String>,
}

impl ProviderTracer {
    /// Create a new provider tracer
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: None,
        }
    }

    /// Set the model name for this tracer
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Trace the start of a request
    pub fn trace_request_start(&self, method: &str, url: &str) {
        info!(
            provider = %self.provider,
            model = ?self.model,
            method = %method,
            url = %url,
            "Request started"
        );
    }

    /// Trace request details (debug level)
    pub fn trace_request_details(
        &self,
        headers: &reqwest::header::HeaderMap,
        body: &serde_json::Value,
    ) {
        debug!(
            provider = %self.provider,
            model = ?self.model,
            request_headers = %format_headers_for_logging(headers),
            request_body = %format_json_for_logging(body),
            "Request details"
        );
    }

    /// Trace successful response
    pub fn trace_response_success(
        &self,
        status_code: u16,
        duration: Instant,
        headers: &reqwest::header::HeaderMap,
    ) {
        let duration_ms = duration.elapsed().as_millis();
        debug!(
            provider = %self.provider,
            model = ?self.model,
            status_code = status_code,
            duration_ms = duration_ms,
            response_headers = %format_headers_for_logging(headers),
            "Request completed successfully"
        );
    }

    /// Trace response body (debug level)
    pub fn trace_response_body(&self, body: &str) {
        debug!(
            provider = %self.provider,
            model = ?self.model,
            response_body = %body,
            "Response body"
        );
    }

    /// Trace request completion
    pub fn trace_request_complete(&self, duration: Instant, response_length: usize) {
        let duration_ms = duration.elapsed().as_millis();
        info!(
            provider = %self.provider,
            model = ?self.model,
            duration_ms = duration_ms,
            response_length = response_length,
            "Request completed"
        );
    }

    /// Trace request failure
    pub fn trace_request_error(&self, status_code: u16, error_text: &str, duration: Instant) {
        let duration_ms = duration.elapsed().as_millis();
        error!(
            provider = %self.provider,
            model = ?self.model,
            status_code = status_code,
            error_text = %error_text,
            duration_ms = duration_ms,
            "Request failed"
        );
    }
}
