//! # Siumai Extras
//!
//! Optional utilities for the `siumai` LLM library, including:
//!
//! - **Schema Validation** (`schema` feature): JSON Schema validation for structured outputs
//! - **Telemetry** (`telemetry` feature): Advanced tracing and logging with `tracing-subscriber`
//! - **Server Adapters** (`server` feature): Axum integration for streaming responses
//!
//! ## Features
//!
//! - `schema` - Enable JSON Schema validation utilities
//! - `telemetry` - Enable tracing subscriber and logging utilities
//! - `server` - Enable server adapter utilities (Axum)
//! - `all` - Enable all features
//!
//! ## Example
//!
//! ```toml
//! [dependencies]
//! siumai = "0.10"
//! siumai-extras = { version = "0.10", features = ["schema", "telemetry"] }
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

// Re-export core siumai types that are commonly used with extras
pub use siumai;

/// Schema validation utilities
#[cfg(feature = "schema")]
pub mod schema;

/// Telemetry and tracing utilities
#[cfg(feature = "telemetry")]
pub mod telemetry;

/// Server adapter utilities
#[cfg(feature = "server")]
pub mod server;

/// Error types for siumai-extras
pub mod error;
