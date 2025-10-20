//! Telemetry Module - External Event Export
//!
//! ## 📋 Purpose
//!
//! This module provides **external event export** capabilities for the Siumai library.
//! It allows you to send structured telemetry events to external observability platforms
//! like Langfuse and Helicone for analysis, monitoring, and debugging in production.
//!
//! ## 🎯 Responsibilities
//!
//! 1. **Event Export**: Export structured events to external platforms
//! 2. **Platform Integration**: Support for Langfuse, Helicone, and custom exporters
//! 3. **Event Types**: Generation events, tool execution, orchestrator events, spans
//!
//! ## 🔄 Relationship with Other Modules
//!
//! - **vs. `tracing/`**: `tracing` = internal logging for developers (stdout, files);
//!   `telemetry` = external event export to platforms (production monitoring)
//! - **vs. `performance`**: `telemetry` = exports events to external platforms;
//!   `performance` = collects metrics internally
//!
//! ## 📚 Documentation
//!
//! See `src/telemetry/README.md` for detailed documentation and examples.
//!
//! ## 🚀 Quick Start
//!
//! ```rust,no_run
//! use siumai::telemetry::{TelemetryConfig, TelemetryEvent};
//! use siumai::prelude::*;
//!
//! // Enable telemetry
//! let config = TelemetryConfig::builder()
//!     .enabled(true)
//!     .record_inputs(true)
//!     .record_outputs(true)
//!     .build();
//!
//! // Use with LLM client
//! let client = LlmBuilder::new()
//!     .openai()
//!     .model("gpt-4")
//!     .telemetry(config)
//!     .build()
//!     .await?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod config;
pub mod events;
pub mod exporters;

pub use config::{TelemetryConfig, TelemetryConfigBuilder};
pub use events::{
    GenerationEvent, OrchestratorEvent, SpanEvent, TelemetryEvent, ToolExecutionEvent,
};
pub use exporters::{TelemetryExporter, helicone::HeliconeExporter, langfuse::LangfuseExporter};

use std::sync::{Arc, OnceLock};
use tokio::sync::RwLock;

/// Global telemetry collector
static TELEMETRY_COLLECTOR: OnceLock<Arc<RwLock<TelemetryCollector>>> = OnceLock::new();

fn get_telemetry_collector() -> Arc<RwLock<TelemetryCollector>> {
    TELEMETRY_COLLECTOR
        .get_or_init(|| Arc::new(RwLock::new(TelemetryCollector::new())))
        .clone()
}

/// Telemetry collector that manages exporters
pub struct TelemetryCollector {
    exporters: Vec<Box<dyn TelemetryExporter>>,
    enabled: bool,
}

impl TelemetryCollector {
    /// Create a new telemetry collector
    pub fn new() -> Self {
        Self {
            exporters: Vec::new(),
            enabled: false,
        }
    }

    /// Add an exporter
    pub fn add_exporter(&mut self, exporter: Box<dyn TelemetryExporter>) {
        self.exporters.push(exporter);
        self.enabled = true;
    }

    /// Remove all exporters
    pub fn clear_exporters(&mut self) {
        self.exporters.clear();
        self.enabled = false;
    }

    /// Check if telemetry is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Emit a telemetry event
    pub async fn emit(&self, event: TelemetryEvent) {
        if !self.enabled {
            return;
        }

        for exporter in &self.exporters {
            if let Err(e) = exporter.export(&event).await {
                tracing::warn!("Failed to export telemetry event: {}", e);
            }
        }
    }
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the global telemetry collector
pub async fn collector() -> Arc<RwLock<TelemetryCollector>> {
    get_telemetry_collector()
}

/// Emit a telemetry event to all registered exporters
pub async fn emit(event: TelemetryEvent) {
    let collector_arc = get_telemetry_collector();
    let collector = collector_arc.read().await;
    collector.emit(event).await;
}

/// Check if telemetry is enabled
pub async fn is_enabled() -> bool {
    let collector_arc = get_telemetry_collector();
    let collector = collector_arc.read().await;
    collector.is_enabled()
}

/// Add a telemetry exporter
pub async fn add_exporter(exporter: Box<dyn TelemetryExporter>) {
    let collector_arc = get_telemetry_collector();
    let mut collector = collector_arc.write().await;
    collector.add_exporter(exporter);
}

/// Clear all telemetry exporters
pub async fn clear_exporters() {
    let collector_arc = get_telemetry_collector();
    let mut collector = collector_arc.write().await;
    collector.clear_exporters();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telemetry_collector() {
        let mut collector = TelemetryCollector::new();
        assert!(!collector.is_enabled());

        // Add a mock exporter
        struct MockExporter;
        #[async_trait::async_trait]
        impl TelemetryExporter for MockExporter {
            async fn export(&self, _event: &TelemetryEvent) -> Result<(), crate::error::LlmError> {
                Ok(())
            }
        }

        collector.add_exporter(Box::new(MockExporter));
        assert!(collector.is_enabled());

        collector.clear_exporters();
        assert!(!collector.is_enabled());
    }
}
