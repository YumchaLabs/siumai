//! Error types for siumai-extras

use thiserror::Error;

/// Errors that can occur in siumai-extras
#[derive(Error, Debug)]
pub enum ExtrasError {
    /// Schema validation error
    #[cfg(feature = "schema")]
    #[error("Schema validation error: {0}")]
    SchemaValidation(String),

    /// Schema compilation error
    #[cfg(feature = "schema")]
    #[error("Schema compilation error: {0}")]
    SchemaCompilation(String),

    /// Telemetry initialization error
    #[cfg(feature = "telemetry")]
    #[error("Telemetry initialization error: {0}")]
    TelemetryInit(String),

    /// Server adapter error
    #[cfg(feature = "server")]
    #[error("Server adapter error: {0}")]
    ServerAdapter(String),

    /// Generic error
    #[error("{0}")]
    Generic(String),
}

/// Result type for siumai-extras operations
pub type Result<T> = std::result::Result<T, ExtrasError>;
