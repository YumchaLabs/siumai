//! Groq provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["groq"]`.

use serde::{Deserialize, Serialize};

/// Groq reasoning output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GroqReasoningFormat {
    /// Return parsed reasoning.
    Parsed,
    /// Return raw reasoning.
    Raw,
    /// Hide reasoning output.
    Hidden,
}

/// Groq reasoning-effort level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GroqReasoningEffort {
    /// Disable reasoning.
    None,
    /// Provider default reasoning effort.
    Default,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
}

/// Groq service tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroqServiceTier {
    /// Use on-demand limits, then fall back to flex when exceeded.
    #[serde(rename = "auto")]
    Auto,
    /// Default tier with consistent fairness.
    #[serde(rename = "on_demand")]
    OnDemand,
    /// Prioritized tier for latency-sensitive workloads.
    #[serde(rename = "performance")]
    Performance,
    /// Higher-throughput tier.
    #[serde(rename = "flex")]
    Flex,
}

/// Typed Groq chat/language-model options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroqLanguageModelOptions {
    /// Groq reasoning output format.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reasoning_format"
    )]
    pub reasoning_format: Option<GroqReasoningFormat>,
    /// Groq reasoning-effort level.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reasoning_effort"
    )]
    pub reasoning_effort: Option<GroqReasoningEffort>,
    /// Whether to enable parallel tool calls.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "parallel_tool_calls"
    )]
    pub parallel_tool_calls: Option<bool>,
    /// End-user identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Whether to use structured outputs.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "structured_outputs"
    )]
    pub structured_outputs: Option<bool>,
    /// Whether to use strict JSON schema validation.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "strict_json_schema"
    )]
    pub strict_json_schema: Option<bool>,
    /// Groq service tier.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "service_tier"
    )]
    pub service_tier: Option<GroqServiceTier>,
}

impl GroqLanguageModelOptions {
    /// Create empty Groq language-model options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reasoning format.
    pub const fn with_reasoning_format(mut self, format: GroqReasoningFormat) -> Self {
        self.reasoning_format = Some(format);
        self
    }

    /// Set the reasoning effort.
    pub const fn with_reasoning_effort(mut self, effort: GroqReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Control parallel tool calls.
    pub const fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }

    /// Set the end-user identifier.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Control structured outputs.
    pub const fn with_structured_outputs(mut self, structured_outputs: bool) -> Self {
        self.structured_outputs = Some(structured_outputs);
        self
    }

    /// Control strict JSON schema validation.
    pub const fn with_strict_json_schema(mut self, strict_json_schema: bool) -> Self {
        self.strict_json_schema = Some(strict_json_schema);
        self
    }

    /// Set the service tier.
    pub const fn with_service_tier(mut self, service_tier: GroqServiceTier) -> Self {
        self.service_tier = Some(service_tier);
        self
    }
}

/// Deprecated AI SDK-compatible alias for Groq language-model options.
#[deprecated(note = "Use GroqLanguageModelOptions instead.")]
pub type GroqChatOptions = GroqLanguageModelOptions;

/// Deprecated AI SDK-compatible alias for Groq language-model options.
#[deprecated(note = "Use GroqLanguageModelOptions instead.")]
pub type GroqProviderOptions = GroqLanguageModelOptions;

/// Typed Groq transcription-model options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroqTranscriptionModelOptions {
    /// Optional input language.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional prompting text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Response format.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "response_format"
    )]
    pub response_format: Option<String>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Timestamp granularity list.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "timestamp_granularities"
    )]
    pub timestamp_granularities: Option<Vec<String>>,
}

impl GroqTranscriptionModelOptions {
    /// Create empty Groq transcription options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set an optional prompt.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set the response format.
    pub fn with_response_format(mut self, response_format: impl Into<String>) -> Self {
        self.response_format = Some(response_format.into());
        self
    }

    /// Set the sampling temperature.
    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Replace the timestamp granularity list.
    pub fn with_timestamp_granularities<I, S>(mut self, granularities: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.timestamp_granularities = Some(granularities.into_iter().map(Into::into).collect());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groq_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            GroqLanguageModelOptions::new()
                .with_reasoning_format(GroqReasoningFormat::Parsed)
                .with_reasoning_effort(GroqReasoningEffort::High)
                .with_parallel_tool_calls(false)
                .with_user("user-123")
                .with_structured_outputs(false)
                .with_strict_json_schema(false)
                .with_service_tier(GroqServiceTier::Performance),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "reasoningFormat": "parsed",
                "reasoningEffort": "high",
                "parallelToolCalls": false,
                "user": "user-123",
                "structuredOutputs": false,
                "strictJsonSchema": false,
                "serviceTier": "performance"
            })
        );
    }

    #[test]
    fn groq_options_accept_snake_case_aliases() {
        let options: GroqLanguageModelOptions = serde_json::from_value(serde_json::json!({
            "reasoning_format": "hidden",
            "reasoning_effort": "medium",
            "parallel_tool_calls": true,
            "structured_outputs": true,
            "strict_json_schema": false,
            "service_tier": "flex"
        }))
        .expect("options deserialize");

        assert_eq!(options.reasoning_format, Some(GroqReasoningFormat::Hidden));
        assert_eq!(options.reasoning_effort, Some(GroqReasoningEffort::Medium));
        assert_eq!(options.parallel_tool_calls, Some(true));
        assert_eq!(options.structured_outputs, Some(true));
        assert_eq!(options.strict_json_schema, Some(false));
        assert_eq!(options.service_tier, Some(GroqServiceTier::Flex));
    }

    #[test]
    fn groq_transcription_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            GroqTranscriptionModelOptions::new()
                .with_language("en")
                .with_prompt("plain text")
                .with_response_format("verbose_json")
                .with_temperature(0.2)
                .with_timestamp_granularities(["word", "segment"]),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "language": "en",
                "prompt": "plain text",
                "responseFormat": "verbose_json",
                "temperature": 0.2,
                "timestampGranularities": ["word", "segment"]
            })
        );
    }

    #[test]
    #[allow(deprecated)]
    fn groq_option_aliases_remain_available() {
        let _: GroqChatOptions = GroqLanguageModelOptions::new();
        let _: GroqProviderOptions = GroqLanguageModelOptions::new();
    }
}
