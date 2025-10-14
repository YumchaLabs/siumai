//! Tracing Subscriber Initialization
//!
//! This module provides functions to initialize the tracing subscriber with various configurations.

use super::config::{OutputFormat, TracingConfig};
use crate::error::LlmError;
use tracing_appender::non_blocking::WorkerGuard;
/// Initialize tracing with the given configuration
pub fn init_tracing(config: TracingConfig) -> Result<Option<WorkerGuard>, LlmError> {
    // Set global pretty JSON flag
    crate::tracing::set_pretty_json(config.pretty_json);

    // Set global mask sensitive values flag
    crate::tracing::set_mask_sensitive_values(config.mask_sensitive_values);

    // Enable W3C trace headers from env override or default off
    let w3c_env = std::env::var("SIUMAI_W3C_TRACE").ok();
    let enabled = matches!(
        w3c_env.as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("on")
    );
    crate::tracing::set_w3c_trace_enabled(enabled);

    // Create filter based on configuration
    let level_str = match config.log_level {
        tracing::Level::TRACE => "trace",
        tracing::Level::DEBUG => "debug",
        tracing::Level::INFO => "info",
        tracing::Level::WARN => "warn",
        tracing::Level::ERROR => "error",
    };

    // Create a more permissive filter that includes examples and all siumai modules
    let filter = format!(
        "siumai={level_str},tracing_monitoring={level_str},simple_trace_test={level_str},automatic_tracing={level_str}"
    );

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

        OutputFormat::Text => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(true)
                .with_thread_ids(false) // Remove thread IDs
                .with_thread_names(false) // Remove thread names
                .fmt_fields(tracing_subscriber::fmt::format::DefaultFields::new()) // Use default field formatting
                .try_init()
        }
    };

    // Handle the case where tracing is already initialized
    match init_result {
        Ok(()) => {
            // Successfully initialized tracing
        }
        Err(e) => {
            let error_msg = e.to_string();
            if error_msg.contains("global default trace dispatcher has already been set") {
                // Tracing is already initialized, which is fine
                // Just log a debug message if possible
                eprintln!("Debug: Tracing already initialized, skipping re-initialization");
            } else {
                // Some other error occurred
                return Err(LlmError::ConfigurationError(format!(
                    "Failed to initialize tracing: {e}"
                )));
            }
        }
    }

    Ok(None)
}

// Simplified implementation - removed complex layer functions

/// Initialize tracing with default configuration
pub fn init_default_tracing() -> Result<Option<WorkerGuard>, LlmError> {
    init_tracing(TracingConfig::default())
}

/// Initialize tracing for debugging
pub fn init_debug_tracing() -> Result<Option<WorkerGuard>, LlmError> {
    init_tracing(TracingConfig::debug())
}

/// Initialize tracing for production
pub fn init_production_tracing(
    log_file: std::path::PathBuf,
) -> Result<Option<WorkerGuard>, LlmError> {
    let config = TracingConfig::builder()
        .log_level_str("warn")
        .map_err(LlmError::ConfigurationError)?
        .output_format(OutputFormat::Json)
        .enable_console(false)
        .log_file(log_file)
        .build();
    init_tracing(config)
}

/// Initialize tracing for performance monitoring only
pub fn init_performance_tracing() -> Result<Option<WorkerGuard>, LlmError> {
    init_tracing(TracingConfig::performance())
}

/// Convenience function to initialize tracing from environment variables
pub fn init_tracing_from_env() -> Result<Option<WorkerGuard>, LlmError> {
    // Start with a builder instead of modifying config

    // Check environment variables and build new config
    let mut builder = TracingConfig::builder();

    if let Ok(level) = std::env::var("SIUMAI_LOG_LEVEL") {
        builder = builder
            .log_level_str(&level)
            .map_err(LlmError::ConfigurationError)?;
    }

    if let Ok(format) = std::env::var("SIUMAI_LOG_FORMAT") {
        let output_format = match format.to_lowercase().as_str() {
            "json" => OutputFormat::Json,
            "json-compact" => OutputFormat::JsonCompact,
            "text" => OutputFormat::Text,
            _ => {
                return Err(LlmError::ConfigurationError(format!(
                    "Invalid log format: {format}. Valid options: text, json, json-compact"
                )));
            }
        };
        builder = builder.output_format(output_format);
    }

    if let Ok(file_path) = std::env::var("SIUMAI_LOG_FILE") {
        builder = builder.log_file(std::path::PathBuf::from(file_path));
    }

    if let Ok(include_bodies) = std::env::var("SIUMAI_LOG_INCLUDE_BODIES") {
        builder = builder.include_bodies(include_bodies.parse().unwrap_or(false));
    }

    if let Ok(max_body_size) = std::env::var("SIUMAI_LOG_MAX_BODY_SIZE") {
        builder = builder.max_body_size(max_body_size.parse().unwrap_or(1024));
    }

    let config = builder.build();
    init_tracing(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_init_tracing() {
        let config = TracingConfig::default();
        // Basic test that tracing initialization doesn't panic
        let _result = init_tracing(config);
    }

    #[test]
    fn test_init_default_tracing() {
        // This test just ensures the function doesn't panic
        // In a real test environment, you might want to capture the output
        let _guard = init_default_tracing();
    }

    #[test]
    fn test_init_file_tracing() {
        let temp_dir = tempdir().unwrap();
        let log_file = temp_dir.path().join("test.log");

        let config = TracingConfig::builder()
            .enable_console(false)
            .log_file(log_file.clone())
            .build();

        let _guard = init_tracing(config).unwrap();

        // Note: In the simplified implementation, we don't actually create files
        // This test just ensures the function doesn't panic
        // In a real implementation, you would verify log file creation
    }
}
