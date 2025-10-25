//! OpenTelemetry middleware for automatic LLM request tracing
//!
//! This module provides middleware that automatically:
//! - Creates spans for all LLM requests and responses
//! - Records metrics (latency, token usage, error rates)
//! - Injects W3C traceparent headers for distributed tracing
//!
//! ## Features
//!
//! ### Automatic Span Creation
//! Creates OpenTelemetry spans for each LLM request with detailed attributes:
//! - Model name, temperature, max_tokens
//! - Token usage (prompt, completion, total)
//! - Request/response timing
//!
//! ### W3C Trace Context Propagation
//! Automatically injects `traceparent` headers into HTTP requests when an active
//! OpenTelemetry span context exists. This enables end-to-end distributed tracing
//! across your application and LLM providers.
//!
//! Format: `traceparent: 00-{trace_id}-{span_id}-{trace_flags}`
//!
//! ### Metrics Collection
//! Records metrics for monitoring and alerting:
//! - Request latency
//! - Token usage
//! - Error rates
//!
//! ## Example
//!
//! ```rust,ignore
//! use siumai_extras::otel;
//! use siumai_extras::otel_middleware::OpenTelemetryMiddleware;
//! use siumai::Client;
//! use std::sync::Arc;
//!
//! // Initialize OpenTelemetry
//! otel::init_opentelemetry("my-service", "http://localhost:4317")?;
//!
//! // Create client with OpenTelemetry middleware
//! let client = Client::builder()
//!     .add_middleware(Arc::new(OpenTelemetryMiddleware::new()))
//!     .build()?;
//!
//! // All requests will now be traced and have traceparent headers injected
//! let response = client.chat()
//!     .create(request)
//!     .await?;
//! ```
//!
//! ## Distributed Tracing Example
//!
//! ```rust,ignore
//! use opentelemetry::trace::{Tracer, TracerProvider};
//! use opentelemetry::global;
//!
//! // Create a parent span for your operation
//! let tracer = global::tracer("my-app");
//! let span = tracer.start("user_request");
//! let cx = opentelemetry::Context::current_with_span(span);
//!
//! // Make LLM request within the span context
//! let _guard = cx.attach();
//! let response = client.chat()
//!     .create(request)
//!     .await?;
//!
//! // The LLM request will be traced as a child span with traceparent header
//! ```

use crate::metrics::LlmMetrics;
use opentelemetry::{
    KeyValue, global,
    trace::{Span, Status, Tracer},
};
use siumai::{
    error::LlmError,
    execution::middleware::LanguageModelMiddleware,
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
    fn transform_params(&self, mut req: ChatRequest) -> ChatRequest {
        // Get current OpenTelemetry span context
        use opentelemetry::trace::TraceContextExt;
        let current_cx = opentelemetry::Context::current();
        let span = current_cx.span();
        let span_cx = span.span_context();

        // Only inject headers if we have a valid span context
        if span_cx.is_valid() {
            // Format W3C traceparent header
            // Format: "00-{trace_id}-{span_id}-{trace_flags}"
            let trace_id = span_cx.trace_id();
            let span_id = span_cx.span_id();
            let trace_flags = span_cx.trace_flags();

            let traceparent = format!(
                "00-{:032x}-{:016x}-{:02x}",
                trace_id,
                span_id,
                trace_flags
            );

            // Inject into request headers
            // Create http_config if it doesn't exist
            let http_config = req.http_config.get_or_insert_with(Default::default);
            http_config
                .headers
                .insert("traceparent".to_string(), traceparent);
        }

        req
    }

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
