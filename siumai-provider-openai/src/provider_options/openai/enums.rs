//! Enum types for OpenAI provider options

use serde::{Deserialize, Serialize};

/// Reasoning effort level for o1/o3 models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// No reasoning
    None,
    /// Minimal reasoning
    Minimal,
    /// Low reasoning effort
    Low,
    /// Medium reasoning effort (default)
    #[default]
    Medium,
    /// High reasoning effort
    High,
    /// Extra-high reasoning effort
    Xhigh,
}

/// Service tier preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    /// Automatic tier selection (default)
    #[default]
    Auto,
    /// Flex tier (cheaper, higher latency; limited model support)
    Flex,
    /// Priority tier (lower latency; requires Enterprise access; limited model support)
    Priority,
    /// Default tier (standard latency)
    Default,
}

/// Prompt cache retention policy (Responses API)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheRetention {
    /// Default caching behavior.
    #[default]
    InMemory,
    /// Keep cached prefixes active for up to 24 hours (limited model support).
    #[serde(rename = "24h")]
    H24,
}

/// Text verbosity level for Responses API
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TextVerbosity {
    /// Low verbosity
    Low,
    /// Medium verbosity (default)
    #[default]
    Medium,
    /// High verbosity
    High,
}

/// Truncation strategy for Responses API
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Truncation {
    /// Automatically truncate to fit within limits
    Auto,
    /// Error if exceeding context window (default)
    Disabled,
}
