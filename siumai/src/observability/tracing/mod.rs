//! Tracing Module - Logging and Debugging Instrumentation
//!
//! Rehomes the tracing subsystem under `crate::observability::tracing`.
//! This module provides structured logging utilities, span/context helpers,
//! and provider-level tracing convenience utilities.

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
    pub fn new() -> Self { Self(Uuid::new_v4()) }
    pub fn as_uuid(&self) -> Uuid { self.0 }
}
impl Default for TraceId { fn default() -> Self { Self::new() } }
impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

/// Span identifier for hierarchical tracing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(pub Uuid);

impl SpanId { pub fn new() -> Self { Self(Uuid::new_v4()) } pub fn as_uuid(&self) -> Uuid { self.0 } }
impl Default for SpanId { fn default() -> Self { Self::new() } }
impl std::fmt::Display for SpanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

/// Context information for tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingContext {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub provider: String,
    pub model: Option<String>,
    pub tags: HashMap<String, String>,
    pub session_start: SystemTime,
}

impl TracingContext {
    pub fn new(provider: String) -> Self {
        Self { trace_id: TraceId::new(), span_id: SpanId::new(), parent_span_id: None, provider, model: None, tags: HashMap::new(), session_start: SystemTime::now() }
    }
    pub fn child(&self) -> Self {
        Self { trace_id: self.trace_id, span_id: SpanId::new(), parent_span_id: Some(self.span_id), provider: self.provider.clone(), model: self.model.clone(), tags: self.tags.clone(), session_start: self.session_start }
    }
    pub fn with_model(mut self, model: String) -> Self { self.model = Some(model); self }
    pub fn with_tag(mut self, key: String, value: String) -> Self { self.tags.insert(key, value); self }
    pub fn session_duration(&self) -> Duration { SystemTime::now().duration_since(self.session_start).unwrap_or_default() }
}

/// Tracing utilities
pub struct TracingUtils;
impl TracingUtils {
    pub fn create_span(_name: &'static str, _context: &TracingContext) -> Span { tracing::info_span!("siumai_operation") }
    pub fn current_context() -> Option<TracingContext> { None }
    pub fn format_duration(duration: Duration) -> String {
        if duration.as_secs() > 0 { format!("{:.2}s", duration.as_secs_f64()) }
        else if duration.as_millis() > 0 { format!("{}ms", duration.as_millis()) }
        else { format!("{}μs", duration.as_micros()) }
    }
    pub fn format_bytes(bytes: u64) -> String {
        const U: &[&str] = &["B", "KB", "MB", "GB"]; let mut size = bytes as f64; let mut i = 0;
        while size >= 1024.0 && i < U.len() - 1 { size /= 1024.0; i += 1; }
        if i == 0 { format!("{} {}", bytes, U[i]) } else { format!("{:.2} {}", size, U[i]) }
    }
}

// Global flags
static PRETTY_JSON: AtomicBool = AtomicBool::new(false);
static MASK_SENSITIVE_VALUES: AtomicBool = AtomicBool::new(true);
pub fn set_pretty_json(pretty: bool) { PRETTY_JSON.store(pretty, Ordering::Relaxed); }
pub fn get_pretty_json() -> bool { PRETTY_JSON.load(Ordering::Relaxed) }
pub fn set_mask_sensitive_values(mask: bool) { MASK_SENSITIVE_VALUES.store(mask, Ordering::Relaxed); }
pub fn get_mask_sensitive_values() -> bool { MASK_SENSITIVE_VALUES.load(Ordering::Relaxed) }

pub fn format_json_for_logging(value: &serde_json::Value) -> String {
    if get_pretty_json() { serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()) }
    else { serde_json::to_string(value).unwrap_or_else(|_| value.to_string()) }
}

pub fn mask_sensitive_value(value: &str) -> String {
    if !get_mask_sensitive_values() { return value.to_string(); }
    if let Some(token) = value.strip_prefix("Bearer ") && token.len() > 8 {
        return format!("Bearer {}...{}", &token[..4], &token[token.len()-4..]);
    }
    if (value.starts_with("sk-") || value.starts_with("sk-ant-") || value.starts_with("gsk-")) && value.len() > 12 {
        return format!("{}...{}", &value[..8], &value[value.len()-4..]);
    }
    if value.len() > 16 { format!("{}...{}", &value[..6], &value[value.len()-4..]) } else { value.to_string() }
}

pub fn format_headers_for_logging(headers: &reqwest::header::HeaderMap) -> String {
    let map: std::collections::HashMap<&str, String> = headers.iter().map(|(k,v)|{
        let value = v.to_str().unwrap_or("<invalid>");
        let masked = if k.as_str().to_lowercase().contains("authorization") || k.as_str().to_lowercase().contains("key") || k.as_str().to_lowercase().contains("token") {
            mask_sensitive_value(value)
        } else { value.to_string() };
        (k.as_str(), masked)
    }).collect();
    if get_pretty_json() { serde_json::to_string_pretty(&map).unwrap_or_else(|_| format!("{map:?}")) }
    else { serde_json::to_string(&map).unwrap_or_else(|_| format!("{map:?}")) }
}

/// Unified provider tracing utility
pub struct ProviderTracer { provider: String, model: Option<String> }
impl ProviderTracer {
    pub fn new(provider: impl Into<String>) -> Self { Self { provider: provider.into(), model: None } }
    pub fn with_model(mut self, model: impl Into<String>) -> Self { self.model = Some(model.into()); self }
    pub fn trace_request_start(&self, method: &str, url: &str) {
        info!(provider=%self.provider, model=?self.model, method=%method, url=%url, "Request started");
    }
    pub fn trace_request_details(&self, headers: &reqwest::header::HeaderMap, body: &serde_json::Value) {
        debug!(provider=%self.provider, model=?self.model, request_headers=%format_headers_for_logging(headers), request_body=%format_json_for_logging(body), "Request details");
    }
    pub fn trace_response_success(&self, status_code: u16, duration: Instant, headers: &reqwest::header::HeaderMap) {
        let duration_ms = duration.elapsed().as_millis();
        debug!(provider=%self.provider, model=?self.model, status_code=status_code, duration_ms=duration_ms, response_headers=%format_headers_for_logging(headers), "Request completed successfully");
    }
    pub fn trace_response_body(&self, body: &str) {
        debug!(provider=%self.provider, model=?self.model, response_body=%body, "Response body");
    }
    pub fn trace_request_complete(&self, duration: Instant, response_length: usize) {
        let duration_ms = duration.elapsed().as_millis();
        info!(provider=%self.provider, model=?self.model, duration_ms=duration_ms, response_length=response_length, "Request completed");
    }
    pub fn trace_request_error(&self, status_code: u16, error_text: &str, duration: Instant) {
        let duration_ms = duration.elapsed().as_millis();
        error!(provider=%self.provider, model=?self.model, status_code=status_code, error_text=%error_text, duration_ms=duration_ms, "Request failed");
    }
}

// Macros kept for compatibility if needed
#[macro_export]
macro_rules! traced_http_request { ($tracer:expr, $method:expr, $url:expr, $body:expr) => {{ let context = $tracer.start_request($method, $url); $tracer.end_request(context); }}; }
#[macro_export]
macro_rules! traced_llm_chat { ($tracer:expr, $messages:expr, $tools:expr) => {{ let context = $tracer.start_chat($messages, $tools); $tracer.end_chat(context); }}; }

