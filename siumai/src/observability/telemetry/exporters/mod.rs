//! Telemetry Exporters
//!
//! Exporters for sending telemetry data to external observability platforms.

pub mod helicone;
pub mod langfuse;

use crate::error::LlmError;
use crate::telemetry::events::TelemetryEvent;

/// Trait for telemetry exporters
#[async_trait::async_trait]
pub trait TelemetryExporter: Send + Sync {
    /// Export a telemetry event
    async fn export(&self, event: &TelemetryEvent) -> Result<(), LlmError>;

    /// Flush any buffered events
    async fn flush(&self) -> Result<(), LlmError> {
        Ok(())
    }

    /// Shutdown the exporter
    async fn shutdown(&self) -> Result<(), LlmError> {
        Ok(())
    }
}
