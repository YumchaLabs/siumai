//! Tracing Configuration
//!
//! This module provides configuration options for the tracing system.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::Level;

/// Output format for tracing logs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Human-readable text format
    Text,
    /// JSON format for structured logging (single line)
    Json,
    /// Compact JSON format (minimal whitespace, no pretty printing)
    JsonCompact,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Text
    }
}

/// Tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Enable HTTP request/response tracing
    pub enable_http_tracing: bool,
    /// Enable LLM interaction tracing
    pub enable_llm_tracing: bool,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Enable error tracking
    pub enable_error_tracking: bool,
    /// Enable streaming event tracing
    pub enable_stream_tracing: bool,
    /// Enable tool call tracing
    pub enable_tool_tracing: bool,
    /// Log level
    pub log_level: Level,
    /// Output format
    pub output_format: OutputFormat,
    /// Log file path (None for stdout)
    pub log_file: Option<PathBuf>,
    /// Maximum log file size in bytes
    pub max_log_file_size: Option<u64>,
    /// Number of log files to keep in rotation
    pub log_file_rotation_count: Option<usize>,
    /// Include request/response bodies in logs
    pub include_bodies: bool,
    /// Maximum body size to log (in bytes)
    pub max_body_size: usize,
    /// Use pretty-printed formatting for JSON bodies and headers
    pub pretty_json: bool,
    /// Include sensitive headers (Authorization, etc.)
    pub include_sensitive_headers: bool,
    /// Mask sensitive values (API keys, tokens) in logs for security
    pub mask_sensitive_values: bool,
    /// Custom fields to include in all log entries
    pub custom_fields: std::collections::HashMap<String, String>,
    /// Enable console output
    pub enable_console: bool,
    /// Enable file output
    pub enable_file: bool,
    /// Sampling rate (0.0 to 1.0, 1.0 = log everything)
    pub sampling_rate: f64,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enable_http_tracing: true,
            enable_llm_tracing: true,
            enable_performance_monitoring: true,
            enable_error_tracking: true,
            enable_stream_tracing: false, // Can be verbose
            enable_tool_tracing: true,
            log_level: Level::INFO,
            output_format: OutputFormat::Text,
            log_file: None,
            max_log_file_size: Some(100 * 1024 * 1024), // 100MB
            log_file_rotation_count: Some(5),
            include_bodies: false, // Privacy by default
            max_body_size: 1024,   // 1KB
            pretty_json: false,
            include_sensitive_headers: false,
            mask_sensitive_values: true, // Security by default
            custom_fields: std::collections::HashMap::new(),
            enable_console: true,
            enable_file: false,
            sampling_rate: 1.0,
        }
    }
}

impl TracingConfig {
    /// Create a new builder
    pub fn builder() -> TracingConfigBuilder {
        TracingConfigBuilder::default()
    }

    /// Create a debug configuration with verbose logging
    pub fn debug() -> Self {
        Self {
            log_level: Level::DEBUG,
            include_bodies: true,
            max_body_size: 10 * 1024, // 10KB
            enable_stream_tracing: true,
            ..Default::default()
        }
    }

    /// Create a production configuration with minimal logging
    pub fn production() -> Self {
        Self {
            log_level: Level::WARN,
            output_format: OutputFormat::Json,
            include_bodies: false,
            include_sensitive_headers: false,
            enable_stream_tracing: false,
            enable_console: false,
            enable_file: true,
            sampling_rate: 0.1, // Sample 10% of requests
            ..Default::default()
        }
    }

    /// Create a performance monitoring configuration
    pub fn performance() -> Self {
        Self {
            enable_http_tracing: false,
            enable_llm_tracing: false,
            enable_performance_monitoring: true,
            enable_error_tracking: true,
            enable_stream_tracing: false,
            enable_tool_tracing: false,
            log_level: Level::INFO,
            output_format: OutputFormat::Json,
            ..Default::default()
        }
    }

    /// Create a development-friendly tracing configuration
    pub fn development() -> Self {
        Self {
            log_level: Level::DEBUG,
            output_format: OutputFormat::Text,
            enable_console: true,
            enable_file: false,
            enable_http_tracing: true,
            enable_llm_tracing: true,
            enable_performance_monitoring: true,
            enable_error_tracking: true,
            enable_stream_tracing: false,
            enable_tool_tracing: true,
            include_bodies: true,
            max_body_size: 4096,
            pretty_json: true, // Enable pretty formatting for development
            include_sensitive_headers: false,
            sampling_rate: 1.0,
            ..Default::default()
        }
    }

    /// Create a minimal tracing configuration (info level, LLM only)
    pub fn minimal() -> Self {
        Self {
            log_level: Level::INFO,
            output_format: OutputFormat::Text,
            enable_console: true,
            enable_file: false,
            enable_http_tracing: false,
            enable_llm_tracing: true,
            enable_performance_monitoring: false,
            enable_error_tracking: true,
            enable_stream_tracing: false,
            enable_tool_tracing: false,
            include_bodies: false,
            max_body_size: 1024,
            include_sensitive_headers: false,
            sampling_rate: 1.0,
            ..Default::default()
        }
    }

    /// Create a production-ready JSON tracing configuration
    pub fn json_production() -> Self {
        Self {
            log_level: Level::WARN,
            output_format: OutputFormat::Json,
            enable_console: false,
            enable_file: true,
            enable_http_tracing: false,
            enable_llm_tracing: true,
            enable_performance_monitoring: true,
            enable_error_tracking: true,
            enable_stream_tracing: false,
            enable_tool_tracing: false,
            include_bodies: false,
            max_body_size: 1024,
            include_sensitive_headers: false,
            sampling_rate: 0.1, // Sample 10% of requests
            ..Default::default()
        }
    }

    /// Create a disabled tracing configuration
    pub fn disabled() -> Self {
        Self {
            log_level: Level::ERROR,
            output_format: OutputFormat::Text,
            enable_console: false,
            enable_file: false,
            enable_http_tracing: false,
            enable_llm_tracing: false,
            enable_performance_monitoring: false,
            enable_error_tracking: false,
            enable_stream_tracing: false,
            enable_tool_tracing: false,
            include_bodies: false,
            max_body_size: 0,
            include_sensitive_headers: false,
            sampling_rate: 0.0,
            ..Default::default()
        }
    }

    /// Enable pretty-printed formatting for JSON bodies and headers
    pub fn with_pretty_json(mut self, pretty: bool) -> Self {
        self.pretty_json = pretty;
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in logs
    pub fn with_mask_sensitive_values(mut self, mask: bool) -> Self {
        self.mask_sensitive_values = mask;
        self
    }
}

/// Builder for tracing configuration
#[derive(Debug, Default, Clone)]
pub struct TracingConfigBuilder {
    config: TracingConfig,
}

impl TracingConfigBuilder {
    /// Create a builder from an existing configuration
    pub fn from_config(config: TracingConfig) -> Self {
        Self { config }
    }

    /// Enable or disable HTTP tracing
    pub fn enable_http_tracing(mut self, enable: bool) -> Self {
        self.config.enable_http_tracing = enable;
        self
    }

    /// Enable or disable LLM tracing
    pub fn enable_llm_tracing(mut self, enable: bool) -> Self {
        self.config.enable_llm_tracing = enable;
        self
    }

    /// Enable or disable performance monitoring
    pub fn enable_performance_monitoring(mut self, enable: bool) -> Self {
        self.config.enable_performance_monitoring = enable;
        self
    }

    /// Enable or disable error tracking
    pub fn enable_error_tracking(mut self, enable: bool) -> Self {
        self.config.enable_error_tracking = enable;
        self
    }

    /// Enable or disable stream tracing
    pub fn enable_stream_tracing(mut self, enable: bool) -> Self {
        self.config.enable_stream_tracing = enable;
        self
    }

    /// Enable or disable tool tracing
    pub fn enable_tool_tracing(mut self, enable: bool) -> Self {
        self.config.enable_tool_tracing = enable;
        self
    }

    /// Set log level
    pub fn log_level<L: Into<Level>>(mut self, level: L) -> Self {
        self.config.log_level = level.into();
        self
    }

    /// Set log level from string
    pub fn log_level_str(mut self, level: &str) -> Result<Self, String> {
        let level = match level.to_lowercase().as_str() {
            "trace" => Level::TRACE,
            "debug" => Level::DEBUG,
            "info" => Level::INFO,
            "warn" => Level::WARN,
            "error" => Level::ERROR,
            _ => return Err(format!("Invalid log level: {level}")),
        };
        self.config.log_level = level;
        Ok(self)
    }

    /// Set output format
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output_format = format;
        self
    }

    /// Set log file path
    pub fn log_file<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config.log_file = Some(path.into());
        self.config.enable_file = true;
        self
    }

    /// Set maximum log file size
    pub fn max_log_file_size(mut self, size: u64) -> Self {
        self.config.max_log_file_size = Some(size);
        self
    }

    /// Set log file rotation count
    pub fn log_file_rotation_count(mut self, count: usize) -> Self {
        self.config.log_file_rotation_count = Some(count);
        self
    }

    /// Include request/response bodies
    pub fn include_bodies(mut self, include: bool) -> Self {
        self.config.include_bodies = include;
        self
    }

    /// Set maximum body size to log
    pub fn max_body_size(mut self, size: usize) -> Self {
        self.config.max_body_size = size;
        self
    }

    /// Use pretty-printed formatting for JSON bodies and headers
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        self.config.pretty_json = pretty;
        self
    }

    /// Include sensitive headers
    pub fn include_sensitive_headers(mut self, include: bool) -> Self {
        self.config.include_sensitive_headers = include;
        self
    }

    /// Mask sensitive values (API keys, tokens) in logs for security
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        self.config.mask_sensitive_values = mask;
        self
    }

    /// Add custom field
    pub fn custom_field<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.config.custom_fields.insert(key.into(), value.into());
        self
    }

    /// Enable console output
    pub fn enable_console(mut self, enable: bool) -> Self {
        self.config.enable_console = enable;
        self
    }

    /// Enable file output
    pub fn enable_file(mut self, enable: bool) -> Self {
        self.config.enable_file = enable;
        self
    }

    /// Set sampling rate
    pub fn sampling_rate(mut self, rate: f64) -> Self {
        self.config.sampling_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Build the configuration
    pub fn build(self) -> TracingConfig {
        self.config
    }
}
