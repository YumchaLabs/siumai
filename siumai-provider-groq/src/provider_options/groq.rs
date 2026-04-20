//! Groq provider options.
//!
//! These typed option structs are owned by the Groq provider crate and are serialized into
//! `providerOptions["groq"]` (Vercel-aligned open options map).

use crate::error::LlmError;
use crate::types::CustomProviderOptions;
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

/// AI SDK-style typed options for Groq transcription models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GroqTranscriptionModelOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_granularities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl GroqTranscriptionModelOptions {
    /// Create new Groq transcription options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the response format.
    pub fn with_response_format(mut self, response_format: impl Into<String>) -> Self {
        self.response_format = Some(response_format.into());
        self
    }

    /// Set an optional prompt.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the input language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the timestamp granularity list.
    pub fn with_timestamp_granularities(mut self, granularities: Vec<String>) -> Self {
        self.timestamp_granularities = Some(granularities);
        self
    }

    /// Control streaming mode.
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Convert to a `(provider_id, json)` entry for `providerOptions`.
    pub fn into_provider_options_map_entry(self) -> Result<(String, serde_json::Value), LlmError> {
        self.to_provider_options_map_entry()
    }
}

impl CustomProviderOptions for GroqTranscriptionModelOptions {
    fn provider_id(&self) -> &str {
        "groq"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut obj = serde_json::Map::new();
        if let Some(v) = self.response_format.as_deref() {
            obj.insert(
                "responseFormat".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.prompt.as_deref() {
            obj.insert(
                "prompt".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.temperature {
            obj.insert("temperature".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.language.as_deref() {
            obj.insert(
                "language".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.timestamp_granularities.as_ref() {
            obj.insert("timestampGranularities".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.stream {
            obj.insert("stream".to_string(), serde_json::Value::Bool(v));
        }
        Ok(serde_json::Value::Object(obj))
    }
}

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

    #[test]
    fn groq_transcription_options_serialize_ai_sdk_style_keys() {
        let value = GroqTranscriptionModelOptions::new()
            .with_response_format("verbose_json")
            .with_prompt("prompt")
            .with_temperature(0.2)
            .with_language("en")
            .with_timestamp_granularities(vec!["segment".to_string()])
            .to_json()
            .expect("transcription options serialize");

        assert_eq!(value["responseFormat"], serde_json::json!("verbose_json"));
        assert_eq!(value["prompt"], serde_json::json!("prompt"));
        assert_eq!(value["temperature"], serde_json::json!(0.2));
        assert_eq!(value["language"], serde_json::json!("en"));
        assert_eq!(
            value["timestampGranularities"],
            serde_json::json!(["segment"])
        );
    }
}
