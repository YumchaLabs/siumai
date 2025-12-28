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
    /// Default tier (standard latency)
    Default,
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

