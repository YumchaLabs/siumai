//! Enum types for OpenAI provider options

use serde::{Deserialize, Serialize};

/// Reasoning effort level for o1/o3 models
///
/// Controls how much effort the model puts into reasoning.
/// Higher effort levels may produce better results but take longer.
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
///
/// Specifies the latency tier to use for processing the request.
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
///
/// Controls the verbosity of text output in responses.
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
///
/// Controls how the model handles context that exceeds the maximum length.
/// - `auto`: If the context exceeds the model's context window, the model will
///   truncate the response to fit by dropping input items in the middle.
/// - `disabled` (default): If a model response will exceed the context window,
///   the request will fail with a 400 error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Truncation {
    /// Automatically truncate to fit within limits
    Auto,
    /// Error if exceeding context window (default)
    Disabled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_effort_serialization() {
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::None).unwrap(),
            r#""none""#
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Minimal).unwrap(),
            r#""minimal""#
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Low).unwrap(),
            r#""low""#
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Medium).unwrap(),
            r#""medium""#
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::High).unwrap(),
            r#""high""#
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Xhigh).unwrap(),
            r#""xhigh""#
        );
    }

    #[test]
    fn test_service_tier_serialization() {
        assert_eq!(
            serde_json::to_string(&ServiceTier::Auto).unwrap(),
            r#""auto""#
        );
        assert_eq!(
            serde_json::to_string(&ServiceTier::Default).unwrap(),
            r#""default""#
        );
    }

    #[test]
    fn test_truncation_serialization() {
        assert_eq!(
            serde_json::to_string(&Truncation::Auto).unwrap(),
            r#""auto""#
        );
        assert_eq!(
            serde_json::to_string(&Truncation::Disabled).unwrap(),
            r#""disabled""#
        );
    }
}
