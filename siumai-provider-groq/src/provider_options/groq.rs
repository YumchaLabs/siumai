//! Groq provider options.
//!
//! These typed option structs are owned by the Groq provider crate and are serialized into
//! `providerOptions["groq"]` (Vercel-aligned open options map).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Groq service tier hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroqServiceTier {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "on_demand")]
    OnDemand,
    #[serde(rename = "performance")]
    Performance,
    #[serde(rename = "flex")]
    Flex,
}

/// Groq reasoning effort hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroqReasoningEffort {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

/// Groq reasoning output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroqReasoningFormat {
    #[serde(rename = "hidden")]
    Hidden,
    #[serde(rename = "raw")]
    Raw,
    #[serde(rename = "parsed")]
    Parsed,
}

/// Groq-specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroqOptions {
    /// Additional Groq-specific parameters.
    #[serde(flatten)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl GroqOptions {
    /// Create new Groq options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable logprobs in the response.
    pub fn with_logprobs(mut self, enabled: bool) -> Self {
        self.extra_params
            .insert("logprobs".to_string(), serde_json::Value::Bool(enabled));
        self
    }

    /// Request the number of top logprobs for each output token.
    pub fn with_top_logprobs(mut self, count: u32) -> Self {
        self.extra_params
            .insert("topLogprobs".to_string(), serde_json::json!(count));
        self
    }

    /// Set Groq service tier.
    pub fn with_service_tier(mut self, tier: GroqServiceTier) -> Self {
        self.insert_json("serviceTier", tier);
        self
    }

    /// Set Groq reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: GroqReasoningEffort) -> Self {
        self.insert_json("reasoningEffort", effort);
        self
    }

    /// Set Groq reasoning format.
    pub fn with_reasoning_format(mut self, format: GroqReasoningFormat) -> Self {
        self.insert_json("reasoningFormat", format);
        self
    }

    /// Control parallel tool calls.
    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.extra_params.insert(
            "parallelToolCalls".to_string(),
            serde_json::Value::Bool(enabled),
        );
        self
    }

    /// Set the end-user identifier.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.extra_params
            .insert("user".to_string(), serde_json::Value::String(user.into()));
        self
    }

    /// Control structured outputs for JSON schema response formats.
    pub fn with_structured_outputs(mut self, enabled: bool) -> Self {
        self.extra_params.insert(
            "structuredOutputs".to_string(),
            serde_json::Value::Bool(enabled),
        );
        self
    }

    /// Control strict JSON schema lowering.
    pub fn with_strict_json_schema(mut self, enabled: bool) -> Self {
        self.extra_params.insert(
            "strictJsonSchema".to_string(),
            serde_json::Value::Bool(enabled),
        );
        self
    }

    /// Add a custom parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }

    fn insert_json<T: serde::Serialize>(&mut self, key: &str, value: T) {
        let json = serde_json::to_value(value).expect("GroqOptions value should serialize");
        self.extra_params.insert(key.to_string(), json);
    }
}

/// AI SDK-style alias for Groq language-model options.
pub type GroqLanguageModelOptions = GroqOptions;

/// Deprecated AI SDK compatibility alias for Groq chat options.
#[deprecated(note = "Use `GroqLanguageModelOptions` instead.")]
pub type GroqProviderOptions = GroqLanguageModelOptions;

/// AI SDK-style alias for Groq transcription-model options.
pub type GroqTranscriptionModelOptions = crate::providers::groq::ext::audio_options::GroqSttOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groq_options_typed_builders_serialize_known_fields() {
        let value = serde_json::to_value(
            GroqOptions::new()
                .with_logprobs(true)
                .with_top_logprobs(3)
                .with_service_tier(GroqServiceTier::Performance)
                .with_reasoning_effort(GroqReasoningEffort::High)
                .with_reasoning_format(GroqReasoningFormat::Parsed)
                .with_parallel_tool_calls(false)
                .with_user("groq-user")
                .with_structured_outputs(false)
                .with_strict_json_schema(false)
                .with_param("vendor_extra", serde_json::json!(true)),
        )
        .expect("options serialize");

        assert_eq!(value["logprobs"], serde_json::json!(true));
        assert_eq!(value["topLogprobs"], serde_json::json!(3));
        assert_eq!(value["serviceTier"], serde_json::json!("performance"));
        assert_eq!(value["reasoningEffort"], serde_json::json!("high"));
        assert_eq!(value["reasoningFormat"], serde_json::json!("parsed"));
        assert_eq!(value["parallelToolCalls"], serde_json::json!(false));
        assert_eq!(value["user"], serde_json::json!("groq-user"));
        assert_eq!(value["structuredOutputs"], serde_json::json!(false));
        assert_eq!(value["strictJsonSchema"], serde_json::json!(false));
        assert_eq!(value["vendor_extra"], serde_json::json!(true));
    }

    #[test]
    fn groq_option_enums_cover_ai_sdk_variant_sets() {
        let service_tiers = [
            (GroqServiceTier::Auto, "auto"),
            (GroqServiceTier::OnDemand, "on_demand"),
            (GroqServiceTier::Performance, "performance"),
            (GroqServiceTier::Flex, "flex"),
        ];
        for (tier, expected) in service_tiers {
            assert_eq!(
                serde_json::to_value(tier).unwrap(),
                serde_json::json!(expected)
            );
        }

        let reasoning_efforts = [
            (GroqReasoningEffort::None, "none"),
            (GroqReasoningEffort::Default, "default"),
            (GroqReasoningEffort::Low, "low"),
            (GroqReasoningEffort::Medium, "medium"),
            (GroqReasoningEffort::High, "high"),
        ];
        for (effort, expected) in reasoning_efforts {
            assert_eq!(
                serde_json::to_value(effort).unwrap(),
                serde_json::json!(expected)
            );
        }
    }
}
