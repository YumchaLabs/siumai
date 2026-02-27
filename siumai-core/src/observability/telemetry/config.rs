//! Telemetry configuration (re-export).
//!
//! The canonical `TelemetryConfig` types live in `siumai-spec` so spec-level request
//! structs (e.g. `ChatRequest`) can reference them without depending on runtime code.

pub use siumai_spec::observability::telemetry::{TelemetryConfig, TelemetryConfigBuilder};
