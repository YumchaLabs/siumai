#![allow(clippy::collapsible_if)]
//! Execution-level telemetry helpers to avoid duplication in executors.

pub mod chat {
    use std::time::SystemTime;

    use crate::observability::telemetry::{
        self,
        events::{GenerationEvent, TelemetryEvent},
    };
    use crate::types::{ChatRequest, ChatResponse};

    /// Emit a span start for chat execution if telemetry is enabled.
    pub async fn span_start(
        telemetry_config: Option<&crate::observability::telemetry::TelemetryConfig>,
        trace_id: &str,
        span_id: &str,
        provider_id: &str,
        model: &str,
        stream: bool,
    ) {
        if let Some(cfg) = telemetry_config {
            if cfg.enabled {
                let span = crate::observability::telemetry::events::SpanEvent::start(
                    span_id.to_string(),
                    None,
                    trace_id.to_string(),
                    "ai.executor.chat.execute".to_string(),
                )
                .with_attribute("provider_id", provider_id.to_string())
                .with_attribute("model", model.to_string())
                .with_attribute("stream", if stream { "true" } else { "false" });
                telemetry::emit(TelemetryEvent::SpanStart(span)).await;
            }
        }
    }

    /// Emit a successful span end. Optionally annotate short_circuit or finish_reason.
    pub async fn span_end_ok(
        telemetry_config: Option<&crate::observability::telemetry::TelemetryConfig>,
        trace_id: &str,
        span_id: &str,
        short_circuit: bool,
        finish_reason: Option<&crate::types::FinishReason>,
    ) {
        if let Some(cfg) = telemetry_config {
            if cfg.enabled {
                let mut span = crate::observability::telemetry::events::SpanEvent::start(
                    span_id.to_string(),
                    None,
                    trace_id.to_string(),
                    "ai.executor.chat.execute".to_string(),
                )
                .end_ok();
                if short_circuit {
                    span = span.with_attribute("short_circuit", "true");
                }
                if let Some(reason) = finish_reason {
                    span = span.with_attribute("finish_reason", format!("{:?}", reason));
                }
                telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
            }
        }
    }

    /// Emit an error span end when chat execution fails.
    pub async fn span_end_err(
        telemetry_config: Option<&crate::observability::telemetry::TelemetryConfig>,
        trace_id: &str,
        span_id: &str,
        error: &crate::error::LlmError,
    ) {
        if let Some(cfg) = telemetry_config {
            if cfg.enabled {
                let span = crate::observability::telemetry::events::SpanEvent::start(
                    span_id.to_string(),
                    None,
                    trace_id.to_string(),
                    "ai.executor.chat.execute".to_string(),
                )
                .end_error(format!("{}", error));
                telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
            }
        }
    }

    // === Streaming variants ===

    /// Emit a span start for chat streaming execution.
    pub async fn span_start_stream(
        telemetry_config: Option<&crate::observability::telemetry::TelemetryConfig>,
        trace_id: &str,
        span_id: &str,
        provider_id: &str,
        model: &str,
    ) {
        if let Some(cfg) = telemetry_config {
            if cfg.enabled {
                let span = crate::observability::telemetry::events::SpanEvent::start(
                    span_id.to_string(),
                    None,
                    trace_id.to_string(),
                    "ai.executor.chat.execute_stream".to_string(),
                )
                .with_attribute("provider_id", provider_id.to_string())
                .with_attribute("model", model.to_string())
                .with_attribute("stream", "true");
                telemetry::emit(TelemetryEvent::SpanStart(span)).await;
            }
        }
    }

    /// Emit an OK span end for streaming execution.
    pub async fn span_end_ok_stream(
        telemetry_config: Option<&crate::observability::telemetry::TelemetryConfig>,
        trace_id: &str,
        span_id: &str,
        short_circuit: bool,
        stream_created: bool,
    ) {
        if let Some(cfg) = telemetry_config {
            if cfg.enabled {
                let mut span = crate::observability::telemetry::events::SpanEvent::start(
                    span_id.to_string(),
                    None,
                    trace_id.to_string(),
                    "ai.executor.chat.execute_stream".to_string(),
                )
                .end_ok();
                if short_circuit {
                    span = span.with_attribute("short_circuit", "true");
                }
                if stream_created {
                    span = span.with_attribute("stream_created", "true");
                }
                telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
            }
        }
    }

    /// Emit an error span end for streaming execution.
    pub async fn span_end_err_stream(
        telemetry_config: Option<&crate::observability::telemetry::TelemetryConfig>,
        trace_id: &str,
        span_id: &str,
        error: &crate::error::LlmError,
    ) {
        if let Some(cfg) = telemetry_config {
            if cfg.enabled {
                let span = crate::observability::telemetry::events::SpanEvent::start(
                    span_id.to_string(),
                    None,
                    trace_id.to_string(),
                    "ai.executor.chat.execute_stream".to_string(),
                )
                .end_error(error.to_string());
                telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
            }
        }
    }

    /// Emit a generation event for a finished chat response.
    pub async fn generation(
        telemetry_config: Option<&crate::observability::telemetry::TelemetryConfig>,
        trace_id: &str,
        provider_id: &str,
        model: &str,
        req: &ChatRequest,
        resp: &ChatResponse,
        started_at: SystemTime,
    ) {
        if let Some(cfg) = telemetry_config {
            if cfg.enabled {
                let mut ge = GenerationEvent::new(
                    uuid::Uuid::new_v4().to_string(),
                    trace_id.to_string(),
                    provider_id.to_string(),
                    model.to_string(),
                );
                if cfg.record_inputs {
                    ge = ge.with_input(req.messages.clone());
                }
                if cfg.record_outputs {
                    ge = ge.with_output(resp.clone());
                }
                if cfg.record_usage
                    && let Some(usage) = &resp.usage
                {
                    ge = ge.with_usage(usage.clone());
                }
                if let Ok(dur) = SystemTime::now().duration_since(started_at) {
                    ge = ge.with_duration(dur);
                }
                if let Some(reason) = &resp.finish_reason {
                    ge = ge.with_finish_reason(reason.clone());
                }
                telemetry::emit(TelemetryEvent::Generation(ge)).await;
            }
        }
    }
}
