#![allow(clippy::map_flatten)]
//! Langfuse Exporter
//!
//! Export telemetry events to Langfuse for observability and analysis.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use siumai::experimental::observability::telemetry::exporters::langfuse::LangfuseExporter;
//! use siumai::experimental::observability::telemetry;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create Langfuse exporter
//! let exporter = LangfuseExporter::new(
//!     "https://cloud.langfuse.com",
//!     "your-public-key",
//!     "your-secret-key",
//! );
//!
//! // Register exporter
//! telemetry::add_exporter(Box::new(exporter)).await;
//! # Ok(())
//! # }
//! ```

use crate::error::LlmError;
use crate::observability::telemetry::events::{GenerationEvent, SpanEvent, TelemetryEvent};
use crate::observability::telemetry::exporters::TelemetryExporter;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Langfuse exporter
pub struct LangfuseExporter {
    client: Client,
    endpoint: String,
    public_key: String,
    secret_key: String,
}

impl LangfuseExporter {
    /// Create a new Langfuse exporter
    pub fn new(
        endpoint: impl Into<String>,
        public_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            endpoint: endpoint.into(),
            public_key: public_key.into(),
            secret_key: secret_key.into(),
        }
    }

    /// Export a span event
    async fn export_span(&self, span: &SpanEvent) -> Result<(), LlmError> {
        let payload = LangfuseSpan {
            id: span.span_id.clone(),
            trace_id: span.trace_id.clone(),
            parent_observation_id: span.parent_span_id.clone(),
            name: span.name.clone(),
            start_time: span.start_time,
            end_time: span.end_time,
            metadata: span.attributes.clone(),
            level: match span.status {
                crate::observability::telemetry::events::SpanStatus::Ok => "DEFAULT".to_string(),
                crate::observability::telemetry::events::SpanStatus::Error => "ERROR".to_string(),
                crate::observability::telemetry::events::SpanStatus::InProgress => {
                    "DEFAULT".to_string()
                }
            },
            status_message: span.error.clone(),
        };

        self.send_event("spans", &payload).await
    }

    /// Export a generation event
    async fn export_generation(&self, generation: &GenerationEvent) -> Result<(), LlmError> {
        let payload = LangfuseGeneration {
            id: generation.id.clone(),
            trace_id: generation.trace_id.clone(),
            parent_observation_id: generation.parent_span_id.clone(),
            name: format!("{}/{}", generation.provider, generation.model),
            start_time: generation.timestamp,
            end_time: generation.timestamp,
            model: generation.model.clone(),
            model_parameters: HashMap::new(),
            input: generation
                .input
                .as_ref()
                .map(|i| serde_json::to_value(i).ok())
                .flatten(),
            output: generation
                .output
                .as_ref()
                .map(|o| serde_json::to_value(o).ok())
                .flatten(),
            usage: generation.usage.as_ref().map(|u| LangfuseUsage {
                prompt_tokens: u.prompt_tokens as i64,
                completion_tokens: u.completion_tokens as i64,
                total_tokens: u.total_tokens as i64,
            }),
            metadata: generation.metadata.clone(),
            level: if generation.error.is_some() {
                "ERROR"
            } else {
                "DEFAULT"
            }
            .to_string(),
            status_message: generation.error.clone(),
        };

        self.send_event("generations", &payload).await
    }

    /// Send an event to Langfuse
    async fn send_event<T: Serialize>(&self, endpoint: &str, payload: &T) -> Result<(), LlmError> {
        let url = format!("{}/api/public/{}", self.endpoint, endpoint);

        let response = self
            .client
            .post(&url)
            .basic_auth(&self.public_key, Some(&self.secret_key))
            .json(payload)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Langfuse API error: {}", body),
                details: None,
            });
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl TelemetryExporter for LangfuseExporter {
    async fn export(&self, event: &TelemetryEvent) -> Result<(), LlmError> {
        match event {
            TelemetryEvent::SpanStart(span) | TelemetryEvent::SpanEnd(span) => {
                self.export_span(span).await
            }
            TelemetryEvent::Generation(generation) => self.export_generation(generation).await,
            TelemetryEvent::ToolExecution(_) => {
                // Tool executions can be exported as spans
                Ok(())
            }
            TelemetryEvent::Orchestrator(_) => {
                // Orchestrator events can be exported as traces
                Ok(())
            }
        }
    }
}

/// Langfuse span payload
#[derive(Debug, Serialize, Deserialize)]
struct LangfuseSpan {
    id: String,
    trace_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_observation_id: Option<String>,
    name: String,
    start_time: std::time::SystemTime,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_time: Option<std::time::SystemTime>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    metadata: HashMap<String, String>,
    level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    status_message: Option<String>,
}

/// Langfuse generation payload
#[derive(Debug, Serialize, Deserialize)]
struct LangfuseGeneration {
    id: String,
    trace_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_observation_id: Option<String>,
    name: String,
    start_time: std::time::SystemTime,
    end_time: std::time::SystemTime,
    model: String,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    model_parameters: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<LangfuseUsage>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    metadata: HashMap<String, String>,
    level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    status_message: Option<String>,
}

/// Langfuse usage information
#[derive(Debug, Serialize, Deserialize)]
struct LangfuseUsage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langfuse_exporter_creation() {
        let exporter = LangfuseExporter::new("https://cloud.langfuse.com", "pk-test", "sk-test");
        assert_eq!(exporter.endpoint, "https://cloud.langfuse.com");
        assert_eq!(exporter.public_key, "pk-test");
        assert_eq!(exporter.secret_key, "sk-test");
    }
}
