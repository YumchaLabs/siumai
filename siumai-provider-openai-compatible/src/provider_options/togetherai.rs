//! TogetherAI provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["togetherai"]`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Typed TogetherAI image-model options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TogetherAIImageModelOptions {
    /// Number of generation steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// Guidance scale for image generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance: Option<f64>,
    /// Negative prompt to guide what to avoid.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "negativePrompt"
    )]
    pub negative_prompt: Option<String>,
    /// Whether to disable the safety checker.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "disableSafetyChecker"
    )]
    pub disable_safety_checker: Option<bool>,
    /// Additional TogetherAI image parameters.
    #[serde(default, flatten, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl TogetherAIImageModelOptions {
    /// Create empty TogetherAI image options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the generation step count.
    pub const fn with_steps(mut self, steps: u32) -> Self {
        self.steps = Some(steps);
        self
    }

    /// Set the guidance scale.
    pub const fn with_guidance(mut self, guidance: f64) -> Self {
        self.guidance = Some(guidance);
        self
    }

    /// Set the negative prompt.
    pub fn with_negative_prompt(mut self, negative_prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative_prompt.into());
        self
    }

    /// Control the TogetherAI safety checker.
    pub const fn with_disable_safety_checker(mut self, disable_safety_checker: bool) -> Self {
        self.disable_safety_checker = Some(disable_safety_checker);
        self
    }

    /// Add an extra provider-owned image field.
    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }
}

/// Deprecated AI SDK-compatible alias for TogetherAI image options.
#[deprecated(note = "Use TogetherAIImageModelOptions instead.")]
pub type TogetherAIImageProviderOptions = TogetherAIImageModelOptions;

/// Historical Rust-style alias for TogetherAI image options.
pub type TogetherAiImageModelOptions = TogetherAIImageModelOptions;

/// Deprecated historical Rust-style alias for TogetherAI image options.
#[deprecated(note = "Use TogetherAIImageModelOptions instead.")]
pub type TogetherAiImageProviderOptions = TogetherAIImageModelOptions;

/// Typed TogetherAI reranking-model options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TogetherAIRerankingModelOptions {
    /// Document fields to rank by.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "rank_fields"
    )]
    pub rank_fields: Option<Vec<String>>,
}

impl TogetherAIRerankingModelOptions {
    /// Create empty TogetherAI reranking options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the rank-fields list.
    pub fn with_rank_fields<I, S>(mut self, rank_fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.rank_fields = Some(rank_fields.into_iter().map(Into::into).collect());
        self
    }
}

/// Deprecated AI SDK-compatible alias for TogetherAI reranking options.
#[deprecated(note = "Use TogetherAIRerankingModelOptions instead.")]
pub type TogetherAIRerankingOptions = TogetherAIRerankingModelOptions;

/// Historical Rust-style alias for TogetherAI reranking options.
pub type TogetherAiRerankingModelOptions = TogetherAIRerankingModelOptions;

/// Deprecated historical Rust-style alias for TogetherAI reranking options.
#[deprecated(note = "Use TogetherAIRerankingModelOptions instead.")]
pub type TogetherAiRerankingOptions = TogetherAIRerankingModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn togetherai_image_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            TogetherAIImageModelOptions::new()
                .with_steps(28)
                .with_guidance(3.5)
                .with_negative_prompt("blurry")
                .with_disable_safety_checker(true)
                .with_extra_field("style", serde_json::json!("anime")),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "steps": 28,
                "guidance": 3.5,
                "negative_prompt": "blurry",
                "disable_safety_checker": true,
                "style": "anime"
            })
        );
    }

    #[test]
    fn togetherai_image_options_accept_camel_case_aliases() {
        let options: TogetherAIImageModelOptions = serde_json::from_value(serde_json::json!({
            "negativePrompt": "grainy",
            "disableSafetyChecker": true
        }))
        .expect("options deserialize");

        assert_eq!(options.negative_prompt.as_deref(), Some("grainy"));
        assert_eq!(options.disable_safety_checker, Some(true));
    }

    #[test]
    fn togetherai_reranking_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            TogetherAIRerankingModelOptions::new().with_rank_fields(["title", "text"]),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "rankFields": ["title", "text"]
            })
        );
    }

    #[test]
    #[allow(deprecated)]
    fn togetherai_option_aliases_remain_available() {
        let _: TogetherAIImageProviderOptions = TogetherAIImageModelOptions::new();
        let _: TogetherAiImageModelOptions = TogetherAIImageModelOptions::new();
        let _: TogetherAiImageProviderOptions = TogetherAIImageModelOptions::new();
        let _: TogetherAIRerankingOptions = TogetherAIRerankingModelOptions::new();
        let _: TogetherAiRerankingModelOptions = TogetherAIRerankingModelOptions::new();
        let _: TogetherAiRerankingOptions = TogetherAIRerankingModelOptions::new();
    }
}
