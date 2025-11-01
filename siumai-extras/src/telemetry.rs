//! Telemetry and tracing utilities
//!
//! This module provides utilities for initializing and configuring tracing subscribers.
//!
//! ## Example
//!
//! ```rust,ignore
//! use siumai_extras::telemetry::{init_subscriber, SubscriberConfig, OutputFormat};
//!
//! // Initialize with default configuration
//! init_subscriber(SubscriberConfig::default())?;
//!
//! // Initialize with custom configuration
//! let config = SubscriberConfig::builder()
//!     .log_level(tracing::Level::DEBUG)
//!     .output_format(OutputFormat::Json)
//!     .build();
//! init_subscriber(config)?;
//! ```

use crate::error::{ExtrasError, Result};
use tracing_appender::non_blocking::WorkerGuard;

/// Output format for tracing logs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Human-readable text format
    #[default]
    Text,
    /// JSON format with pretty printing
    Json,
    /// Compact JSON format
    JsonCompact,
}

/// Configuration for tracing subscriber
#[derive(Debug, Clone)]
pub struct SubscriberConfig {
    /// Log level
    pub log_level: tracing::Level,
    /// Output format
    pub output_format: OutputFormat,
    /// Enable console output
    pub enable_console: bool,
    /// Log file path (optional)
    pub log_file: Option<std::path::PathBuf>,
}

impl Default for SubscriberConfig {
    fn default() -> Self {
        Self {
            log_level: tracing::Level::INFO,
            output_format: OutputFormat::Text,
            enable_console: true,
            log_file: None,
        }
    }
}

impl SubscriberConfig {
    /// Create a new builder for SubscriberConfig
    pub fn builder() -> SubscriberConfigBuilder {
        SubscriberConfigBuilder::default()
    }

    /// Create a debug configuration
    pub fn debug() -> Self {
        Self {
            log_level: tracing::Level::DEBUG,
            output_format: OutputFormat::Text,
            enable_console: true,
            log_file: None,
        }
    }

    /// Create a production configuration
    pub fn production(log_file: std::path::PathBuf) -> Self {
        Self {
            log_level: tracing::Level::WARN,
            output_format: OutputFormat::Json,
            enable_console: false,
            log_file: Some(log_file),
        }
    }
}

/// Builder for SubscriberConfig
#[derive(Debug, Default)]
pub struct SubscriberConfigBuilder {
    log_level: Option<tracing::Level>,
    output_format: Option<OutputFormat>,
    enable_console: Option<bool>,
    log_file: Option<std::path::PathBuf>,
}

impl SubscriberConfigBuilder {
    /// Set the log level
    pub fn log_level(mut self, level: tracing::Level) -> Self {
        self.log_level = Some(level);
        self
    }

    /// Set the log level from a string
    pub fn log_level_str(mut self, level: &str) -> Result<Self> {
        let level = match level.to_lowercase().as_str() {
            "trace" => tracing::Level::TRACE,
            "debug" => tracing::Level::DEBUG,
            "info" => tracing::Level::INFO,
            "warn" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => {
                return Err(ExtrasError::TelemetryInit(format!(
                    "Invalid log level: {}. Valid options: trace, debug, info, warn, error",
                    level
                )));
            }
        };
        self.log_level = Some(level);
        Ok(self)
    }

    /// Set the output format
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    /// Enable or disable console output
    pub fn enable_console(mut self, enable: bool) -> Self {
        self.enable_console = Some(enable);
        self
    }

    /// Set the log file path
    pub fn log_file(mut self, path: std::path::PathBuf) -> Self {
        self.log_file = Some(path);
        self
    }

    /// Build the configuration
    pub fn build(self) -> SubscriberConfig {
        SubscriberConfig {
            log_level: self.log_level.unwrap_or(tracing::Level::INFO),
            output_format: self.output_format.unwrap_or_default(),
            enable_console: self.enable_console.unwrap_or(true),
            log_file: self.log_file,
        }
    }
}

/// Initialize tracing subscriber with the given configuration
///
/// ## Arguments
///
/// - `config`: The subscriber configuration
///
/// ## Returns
///
/// - `Ok(Option<WorkerGuard>)` if initialization succeeds. The guard must be kept alive
///   for the duration of the program if file logging is enabled.
/// - `Err(ExtrasError::TelemetryInit)` if initialization fails
///
/// ## Example
///
/// ```rust,ignore
/// use siumai_extras::telemetry::{init_subscriber, SubscriberConfig};
///
/// let config = SubscriberConfig::default();
/// let _guard = init_subscriber(config)?;
/// ```
pub fn init_subscriber(config: SubscriberConfig) -> Result<Option<WorkerGuard>> {
    let level_str = match config.log_level {
        tracing::Level::TRACE => "trace",
        tracing::Level::DEBUG => "debug",
        tracing::Level::INFO => "info",
        tracing::Level::WARN => "warn",
        tracing::Level::ERROR => "error",
    };

    // Create filter
    let filter = format!("siumai={level_str},siumai_extras={level_str}");

    // Apply output format
    let init_result = match config.output_format {
        OutputFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .json()
            .try_init(),
        OutputFormat::JsonCompact => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .json()
            .compact()
            .try_init(),
        OutputFormat::Text => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(false)
            .with_thread_names(false)
            .try_init(),
    };

    // Handle the case where tracing is already initialized
    match init_result {
        Ok(()) => Ok(None),
        Err(e) => {
            let error_msg = e.to_string();
            if error_msg.contains("global default trace dispatcher has already been set") {
                // Tracing is already initialized, which is fine
                Ok(None)
            } else {
                Err(ExtrasError::TelemetryInit(format!(
                    "Failed to initialize tracing: {}",
                    e
                )))
            }
        }
    }
}

/// Initialize tracing subscriber with default configuration
pub fn init_default() -> Result<Option<WorkerGuard>> {
    init_subscriber(SubscriberConfig::default())
}

/// Initialize tracing subscriber for debugging
pub fn init_debug() -> Result<Option<WorkerGuard>> {
    init_subscriber(SubscriberConfig::debug())
}

/// Initialize tracing subscriber for production
pub fn init_production(log_file: std::path::PathBuf) -> Result<Option<WorkerGuard>> {
    init_subscriber(SubscriberConfig::production(log_file))
}

/// Initialize tracing subscriber from environment variables
///
/// Supported environment variables:
/// - `SIUMAI_LOG_LEVEL`: Log level (trace, debug, info, warn, error)
/// - `SIUMAI_LOG_FORMAT`: Output format (text, json, json-compact)
/// - `SIUMAI_LOG_FILE`: Log file path
pub fn init_from_env() -> Result<Option<WorkerGuard>> {
    let mut builder = SubscriberConfig::builder();

    if let Ok(level) = std::env::var("SIUMAI_LOG_LEVEL") {
        builder = builder.log_level_str(&level)?;
    }

    if let Ok(format) = std::env::var("SIUMAI_LOG_FORMAT") {
        let output_format = match format.to_lowercase().as_str() {
            "json" => OutputFormat::Json,
            "json-compact" => OutputFormat::JsonCompact,
            "text" => OutputFormat::Text,
            _ => {
                return Err(ExtrasError::TelemetryInit(format!(
                    "Invalid log format: {}. Valid options: text, json, json-compact",
                    format
                )));
            }
        };
        builder = builder.output_format(output_format);
    }

    if let Ok(file_path) = std::env::var("SIUMAI_LOG_FILE") {
        builder = builder.log_file(std::path::PathBuf::from(file_path));
    }

    let config = builder.build();
    init_subscriber(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_subscriber() {
        let config = SubscriberConfig::default();
        let _result = init_subscriber(config);
    }

    #[test]
    fn test_init_default() {
        let _guard = init_default();
    }
}
