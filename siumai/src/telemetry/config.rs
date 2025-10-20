//! Telemetry Configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Telemetry configuration
///
/// Controls what data is recorded and exported to observability platforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable or disable telemetry
    pub enabled: bool,

    /// Record input messages and prompts
    ///
    /// You might want to disable this to avoid recording sensitive information,
    /// to reduce data transfers, or to increase performance.
    pub record_inputs: bool,

    /// Record output messages and completions
    ///
    /// You might want to disable this to avoid recording sensitive information,
    /// to reduce data transfers, or to increase performance.
    pub record_outputs: bool,

    /// Record tool calls and their arguments
    pub record_tools: bool,

    /// Record usage/token information
    pub record_usage: bool,

    /// Function identifier for grouping telemetry data
    pub function_id: Option<String>,

    /// Additional metadata to include in telemetry events
    pub metadata: HashMap<String, String>,

    /// Session ID for grouping related requests
    pub session_id: Option<String>,

    /// User ID for tracking user-specific metrics
    pub user_id: Option<String>,

    /// Tags for categorizing telemetry data
    pub tags: Vec<String>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            record_inputs: true,
            record_outputs: true,
            record_tools: true,
            record_usage: true,
            function_id: None,
            metadata: HashMap::new(),
            session_id: None,
            user_id: None,
            tags: Vec::new(),
        }
    }
}

impl TelemetryConfig {
    /// Create a new builder
    pub fn builder() -> TelemetryConfigBuilder {
        TelemetryConfigBuilder::default()
    }

    /// Create a configuration for development (all recording enabled)
    pub fn development() -> Self {
        Self {
            enabled: true,
            record_inputs: true,
            record_outputs: true,
            record_tools: true,
            record_usage: true,
            ..Default::default()
        }
    }

    /// Create a configuration for production (sensitive data disabled)
    pub fn production() -> Self {
        Self {
            enabled: true,
            record_inputs: false,  // Don't record inputs in production
            record_outputs: false, // Don't record outputs in production
            record_tools: true,
            record_usage: true,
            ..Default::default()
        }
    }

    /// Create a minimal configuration (only usage tracking)
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            record_inputs: false,
            record_outputs: false,
            record_tools: false,
            record_usage: true,
            ..Default::default()
        }
    }
}

/// Builder for TelemetryConfig
#[derive(Debug, Clone, Default)]
pub struct TelemetryConfigBuilder {
    enabled: bool,
    record_inputs: bool,
    record_outputs: bool,
    record_tools: bool,
    record_usage: bool,
    function_id: Option<String>,
    metadata: HashMap<String, String>,
    session_id: Option<String>,
    user_id: Option<String>,
    tags: Vec<String>,
}

impl TelemetryConfigBuilder {
    /// Enable or disable telemetry
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Record input messages and prompts
    pub fn record_inputs(mut self, record: bool) -> Self {
        self.record_inputs = record;
        self
    }

    /// Record output messages and completions
    pub fn record_outputs(mut self, record: bool) -> Self {
        self.record_outputs = record;
        self
    }

    /// Record tool calls and their arguments
    pub fn record_tools(mut self, record: bool) -> Self {
        self.record_tools = record;
        self
    }

    /// Record usage/token information
    pub fn record_usage(mut self, record: bool) -> Self {
        self.record_usage = record;
        self
    }

    /// Set function identifier
    pub fn function_id(mut self, id: impl Into<String>) -> Self {
        self.function_id = Some(id.into());
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set session ID
    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    /// Set user ID
    pub fn user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }

    /// Add a tag
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags.extend(tags);
        self
    }

    /// Build the configuration
    pub fn build(self) -> TelemetryConfig {
        TelemetryConfig {
            enabled: self.enabled,
            record_inputs: self.record_inputs,
            record_outputs: self.record_outputs,
            record_tools: self.record_tools,
            record_usage: self.record_usage,
            function_id: self.function_id,
            metadata: self.metadata,
            session_id: self.session_id,
            user_id: self.user_id,
            tags: self.tags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TelemetryConfig::default();
        assert!(!config.enabled);
        assert!(config.record_inputs);
        assert!(config.record_outputs);
    }

    #[test]
    fn test_development_config() {
        let config = TelemetryConfig::development();
        assert!(config.enabled);
        assert!(config.record_inputs);
        assert!(config.record_outputs);
    }

    #[test]
    fn test_production_config() {
        let config = TelemetryConfig::production();
        assert!(config.enabled);
        assert!(!config.record_inputs);
        assert!(!config.record_outputs);
        assert!(config.record_usage);
    }

    #[test]
    fn test_builder() {
        let config = TelemetryConfig::builder()
            .enabled(true)
            .record_inputs(false)
            .function_id("test-function")
            .metadata("key", "value")
            .session_id("session-123")
            .user_id("user-456")
            .tag("production")
            .build();

        assert!(config.enabled);
        assert!(!config.record_inputs);
        assert_eq!(config.function_id, Some("test-function".to_string()));
        assert_eq!(config.metadata.get("key"), Some(&"value".to_string()));
        assert_eq!(config.session_id, Some("session-123".to_string()));
        assert_eq!(config.user_id, Some("user-456".to_string()));
        assert_eq!(config.tags, vec!["production"]);
    }
}
