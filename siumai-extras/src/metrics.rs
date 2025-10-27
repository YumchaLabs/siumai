//! Metrics collection for LLM requests
//!
//! This module provides utilities for collecting metrics about LLM requests,
//! including request duration, token usage, and error rates.
//!
//! ## Example
//!
//! ```rust,ignore
//! use siumai_extras::metrics::LlmMetrics;
//!
//! let metrics = LlmMetrics::new();
//!
//! // Record a successful request
//! metrics.record_request(
//!     "openai",
//!     "gpt-4",
//!     Duration::from_millis(1500),
//!     Some(1000),
//!     true,
//! );
//! ```

use opentelemetry::{KeyValue, global, metrics::*};
use std::time::Duration;

/// LLM metrics collector
#[derive(Clone)]
pub struct LlmMetrics {
    /// Request duration histogram (in milliseconds)
    request_duration: Histogram<f64>,
    /// Token usage counter
    token_usage: Counter<u64>,
    /// Request counter
    request_count: Counter<u64>,
    /// Error counter
    error_count: Counter<u64>,
}

impl LlmMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let meter = global::meter("siumai");

        let request_duration = meter
            .f64_histogram("llm.request.duration")
            .with_description("Duration of LLM requests in milliseconds")
            .build();

        let token_usage = meter
            .u64_counter("llm.tokens.usage")
            .with_description("Total tokens used in LLM requests")
            .build();

        let request_count = meter
            .u64_counter("llm.requests.count")
            .with_description("Total number of LLM requests")
            .build();

        let error_count = meter
            .u64_counter("llm.errors.count")
            .with_description("Total number of LLM request errors")
            .build();

        Self {
            request_duration,
            token_usage,
            request_count,
            error_count,
        }
    }

    /// Record a request
    ///
    /// ## Arguments
    ///
    /// - `provider`: Provider name (e.g., "openai", "anthropic")
    /// - `model`: Model name (e.g., "gpt-4", "claude-3-opus")
    /// - `duration`: Request duration
    /// - `tokens`: Total tokens used (optional)
    /// - `success`: Whether the request was successful
    pub fn record_request(
        &self,
        provider: &str,
        model: &str,
        duration: Duration,
        tokens: Option<u64>,
        success: bool,
    ) {
        let attributes = vec![
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("model", model.to_string()),
            KeyValue::new("success", success),
        ];

        // Record duration
        self.request_duration
            .record(duration.as_millis() as f64, &attributes);

        // Record request count
        self.request_count.add(1, &attributes);

        // Record token usage if available
        if let Some(token_count) = tokens {
            self.token_usage.add(token_count, &attributes);
        }

        // Record error if not successful
        if !success {
            self.error_count.add(1, &attributes);
        }
    }

    /// Record a streaming request
    ///
    /// ## Arguments
    ///
    /// - `provider`: Provider name
    /// - `model`: Model name
    /// - `duration`: Total streaming duration
    /// - `tokens`: Total tokens used (optional)
    /// - `success`: Whether the stream completed successfully
    pub fn record_stream(
        &self,
        provider: &str,
        model: &str,
        duration: Duration,
        tokens: Option<u64>,
        success: bool,
    ) {
        let attributes = vec![
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("model", model.to_string()),
            KeyValue::new("streaming", true),
            KeyValue::new("success", success),
        ];

        // Record duration
        self.request_duration
            .record(duration.as_millis() as f64, &attributes);

        // Record request count
        self.request_count.add(1, &attributes);

        // Record token usage if available
        if let Some(token_count) = tokens {
            self.token_usage.add(token_count, &attributes);
        }

        // Record error if not successful
        if !success {
            self.error_count.add(1, &attributes);
        }
    }

    /// Record an error
    ///
    /// ## Arguments
    ///
    /// - `provider`: Provider name
    /// - `model`: Model name (optional)
    /// - `error_type`: Error type (e.g., "timeout", "rate_limit", "api_error")
    pub fn record_error(&self, provider: &str, model: Option<&str>, error_type: &str) {
        let mut attributes = vec![
            KeyValue::new("provider", provider.to_string()),
            KeyValue::new("error_type", error_type.to_string()),
        ];

        if let Some(m) = model {
            attributes.push(KeyValue::new("model", m.to_string()));
        }

        self.error_count.add(1, &attributes);
    }
}

impl Default for LlmMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Global metrics instance
static GLOBAL_METRICS: once_cell::sync::Lazy<LlmMetrics> =
    once_cell::sync::Lazy::new(LlmMetrics::new);

/// Get the global metrics instance
pub fn global_metrics() -> &'static LlmMetrics {
    &GLOBAL_METRICS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = LlmMetrics::new();
        metrics.record_request(
            "openai",
            "gpt-4",
            Duration::from_millis(1500),
            Some(1000),
            true,
        );
    }

    #[test]
    fn test_global_metrics() {
        let metrics = global_metrics();
        metrics.record_request(
            "anthropic",
            "claude-3-opus",
            Duration::from_millis(2000),
            Some(1500),
            true,
        );
    }
}
