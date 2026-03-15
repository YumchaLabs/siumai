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
            .insert("top_logprobs".to_string(), serde_json::json!(count));
        self
    }

    /// Set Groq service tier.
    pub fn with_service_tier(mut self, tier: GroqServiceTier) -> Self {
        self.insert_json("service_tier", tier);
        self
    }

    /// Set Groq reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: GroqReasoningEffort) -> Self {
        self.insert_json("reasoning_effort", effort);
        self
    }

    /// Set Groq reasoning format.
    pub fn with_reasoning_format(mut self, format: GroqReasoningFormat) -> Self {
        self.insert_json("reasoning_format", format);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groq_options_typed_builders_serialize_known_fields() {
        let value = serde_json::to_value(
            GroqOptions::new()
                .with_logprobs(true)
                .with_top_logprobs(3)
                .with_service_tier(GroqServiceTier::Flex)
                .with_reasoning_effort(GroqReasoningEffort::Default)
                .with_reasoning_format(GroqReasoningFormat::Parsed)
                .with_param("vendor_extra", serde_json::json!(true)),
        )
        .expect("options serialize");

        assert_eq!(value["logprobs"], serde_json::json!(true));
        assert_eq!(value["top_logprobs"], serde_json::json!(3));
        assert_eq!(value["service_tier"], serde_json::json!("flex"));
        assert_eq!(value["reasoning_effort"], serde_json::json!("default"));
        assert_eq!(value["reasoning_format"], serde_json::json!("parsed"));
        assert_eq!(value["vendor_extra"], serde_json::json!(true));
    }
}
