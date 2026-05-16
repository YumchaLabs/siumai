use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Structured timeout configuration details for request-facing controls.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TimeoutConfigurationSettings {
    /// Total timeout in milliseconds.
    #[serde(rename = "totalMs", skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<u64>,
    /// Per-step timeout in milliseconds.
    #[serde(rename = "stepMs", skip_serializing_if = "Option::is_none")]
    pub step_ms: Option<u64>,
    /// Timeout between stream chunks in milliseconds.
    #[serde(rename = "chunkMs", skip_serializing_if = "Option::is_none")]
    pub chunk_ms: Option<u64>,
    /// Default timeout for all tools in milliseconds.
    #[serde(rename = "toolMs", skip_serializing_if = "Option::is_none")]
    pub tool_ms: Option<u64>,
    /// Per-tool timeout overrides keyed by the AI SDK-style `{toolName}Ms` suffix form.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tools: HashMap<String, u64>,
}

impl TimeoutConfigurationSettings {
    /// Create empty timeout settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set total timeout in milliseconds.
    pub const fn with_total_ms(mut self, total_ms: u64) -> Self {
        self.total_ms = Some(total_ms);
        self
    }

    /// Set per-step timeout in milliseconds.
    pub const fn with_step_ms(mut self, step_ms: u64) -> Self {
        self.step_ms = Some(step_ms);
        self
    }

    /// Set between-chunk timeout in milliseconds.
    pub const fn with_chunk_ms(mut self, chunk_ms: u64) -> Self {
        self.chunk_ms = Some(chunk_ms);
        self
    }

    /// Set default tool timeout in milliseconds.
    pub const fn with_tool_ms(mut self, tool_ms: u64) -> Self {
        self.tool_ms = Some(tool_ms);
        self
    }

    /// Set a tool-specific timeout in milliseconds.
    pub fn with_tool_timeout_ms(mut self, tool_name: impl AsRef<str>, timeout_ms: u64) -> Self {
        self.tools
            .insert(format!("{}Ms", tool_name.as_ref()), timeout_ms);
        self
    }

    /// Resolve the tool timeout for the given tool name.
    pub fn tool_timeout_ms(&self, tool_name: &str) -> Option<u64> {
        self.tools
            .get(&format!("{tool_name}Ms"))
            .copied()
            .or(self.tool_ms)
    }
}

/// AI SDK-style timeout configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TimeoutConfiguration {
    /// A single total timeout in milliseconds.
    Millis(u64),
    /// Structured timeout configuration.
    Settings(TimeoutConfigurationSettings),
}

impl TimeoutConfiguration {
    /// Create a total-timeout configuration from milliseconds.
    pub const fn from_millis(total_ms: u64) -> Self {
        Self::Millis(total_ms)
    }

    /// Create a structured timeout configuration.
    pub fn settings(settings: TimeoutConfigurationSettings) -> Self {
        Self::Settings(settings)
    }

    /// Total timeout in milliseconds.
    pub const fn total_timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Millis(total_ms) => Some(*total_ms),
            Self::Settings(settings) => settings.total_ms,
        }
    }

    /// Per-step timeout in milliseconds.
    pub const fn step_timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Millis(_) => None,
            Self::Settings(settings) => settings.step_ms,
        }
    }

    /// Per-chunk timeout in milliseconds.
    pub const fn chunk_timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Millis(_) => None,
            Self::Settings(settings) => settings.chunk_ms,
        }
    }

    /// Per-tool timeout in milliseconds.
    pub fn tool_timeout_ms(&self, tool_name: &str) -> Option<u64> {
        match self {
            Self::Millis(_) => None,
            Self::Settings(settings) => settings.tool_timeout_ms(tool_name),
        }
    }

    /// Total timeout as a `Duration`.
    pub fn total_timeout(&self) -> Option<Duration> {
        self.total_timeout_ms().map(Duration::from_millis)
    }

    /// Step timeout as a `Duration`.
    pub fn step_timeout(&self) -> Option<Duration> {
        self.step_timeout_ms().map(Duration::from_millis)
    }

    /// Chunk timeout as a `Duration`.
    pub fn chunk_timeout(&self) -> Option<Duration> {
        self.chunk_timeout_ms().map(Duration::from_millis)
    }
}

/// AI SDK-style helper for extracting the total timeout in milliseconds.
pub const fn get_total_timeout_ms(timeout: Option<&TimeoutConfiguration>) -> Option<u64> {
    match timeout {
        Some(timeout) => timeout.total_timeout_ms(),
        None => None,
    }
}

/// AI SDK-style helper for extracting the step timeout in milliseconds.
pub const fn get_step_timeout_ms(timeout: Option<&TimeoutConfiguration>) -> Option<u64> {
    match timeout {
        Some(timeout) => timeout.step_timeout_ms(),
        None => None,
    }
}

/// AI SDK-style helper for extracting the chunk timeout in milliseconds.
pub const fn get_chunk_timeout_ms(timeout: Option<&TimeoutConfiguration>) -> Option<u64> {
    match timeout {
        Some(timeout) => timeout.chunk_timeout_ms(),
        None => None,
    }
}

/// AI SDK-style helper for extracting one tool timeout in milliseconds.
pub fn get_tool_timeout_ms(timeout: Option<&TimeoutConfiguration>, tool_name: &str) -> Option<u64> {
    timeout.and_then(|timeout| timeout.tool_timeout_ms(tool_name))
}
