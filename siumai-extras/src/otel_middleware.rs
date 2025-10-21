//! OpenTelemetry middleware for automatic LLM request tracing
//!
//! This module provides middleware that automatically creates spans and records metrics
//! for all LLM requests and responses using the LanguageModelMiddleware trait.
//!
//! ## Example
//!
//! ```rust,ignore
//! use siumai_extras::otel_middleware::OpenTelemetryMiddleware;
//! use siumai::providers::openai::OpenAiClient;
//! use std::sync::Arc;
//!
//! let client = OpenAiClient::builder()
//!     .api_key("your-api-key")
//!     .with_middleware(Arc::new(OpenTelemetryMiddleware::new()))
//!     .build()?;
//! ```

use crate::metrics::LlmMetrics;
use opentelemetry::{
    KeyValue, global,
    trace::{Span, Status, Tracer},
};
use siumai::{
    error::LlmError,
    middleware::LanguageModelMiddleware,
    types::{ChatRequest, ChatResponse},
};

/// OpenTelemetry middleware for automatic tracing and metrics
#[derive(Clone)]
pub struct OpenTelemetryMiddleware {
    metrics: LlmMetrics,
}

impl OpenTelemetryMiddleware {
    /// Create a new OpenTelemetry middleware
    pub fn new() -> Self {
        Self {
            metrics: LlmMetrics::new(),
        }
    }

    /// Create a new OpenTelemetry middleware with custom metrics
    pub fn with_metrics(metrics: LlmMetrics) -> Self {
        Self { metrics }
    }

    /// Create span attributes from request
    fn create_span_attributes(req: &ChatRequest) -> Vec<KeyValue> {
        let mut attributes = vec![KeyValue::new("llm.model", req.common_params.model.clone())];

        if let Some(max_tokens) = req.common_params.max_tokens {
            attributes.push(KeyValue::new("llm.max_tokens", max_tokens as i64));
        }

        if let Some(temperature) = req.common_params.temperature {
            attributes.push(KeyValue::new("llm.temperature", temperature as f64));
        }

        if let Some(top_p) = req.common_params.top_p {
            attributes.push(KeyValue::new("llm.top_p", top_p as f64));
        }

        // Add message count
        attributes.push(KeyValue::new(
            "llm.message_count",
            req.messages.len() as i64,
        ));

        // Add tool count if present
        if let Some(ref tools) = req.tools {
            attributes.push(KeyValue::new("llm.tool_count", tools.len() as i64));
        }

        attributes
    }

    /// Add response attributes to span
    fn add_response_attributes<S: Span>(span: &mut S, response: &ChatResponse) {
        if let Some(ref usage) = response.usage {
            span.set_attribute(KeyValue::new(
                "llm.usage.prompt_tokens",
                usage.prompt_tokens as i64,
            ));
            span.set_attribute(KeyValue::new(
                "llm.usage.completion_tokens",
                usage.completion_tokens as i64,
            ));
            span.set_attribute(KeyValue::new(
                "llm.usage.total_tokens",
                usage.total_tokens as i64,
            ));
        }

        if let Some(ref model) = response.model {
            span.set_attribute(KeyValue::new("llm.response.model", model.clone()));
        }

        if let Some(ref id) = response.id {
            span.set_attribute(KeyValue::new("llm.response.id", id.clone()));
        }

        // Add finish reason if available
        if let Some(ref finish_reason) = response.finish_reason {
            span.set_attribute(KeyValue::new(
                "llm.finish_reason",
                format!("{:?}", finish_reason),
            ));
        }
    }
}

impl Default for OpenTelemetryMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageModelMiddleware for OpenTelemetryMiddleware {
    fn post_generate(
        &self,
        req: &ChatRequest,
        resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        // Create a span for this request
        let tracer = global::tracer("siumai");
        let mut span = tracer.start(format!("llm.chat {}", req.common_params.model));

        // Add request attributes
        for attr in Self::create_span_attributes(req) {
            span.set_attribute(attr);
        }

        // Add response attributes
        Self::add_response_attributes(&mut span, &resp);
        span.set_status(Status::Ok);
        span.end();

        Ok(resp)
    }
}
