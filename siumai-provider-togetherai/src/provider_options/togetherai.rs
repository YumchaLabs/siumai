//! Typed provider options for TogetherAI reranking and image generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Typed options stored under `provider_options_map["togetherai"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TogetherAiRerankOptions {
    /// Rerank fields used for structured documents.
    #[serde(skip_serializing_if = "Option::is_none", rename = "rankFields")]
    pub rank_fields: Option<Vec<String>>,
}

impl TogetherAiRerankOptions {
    /// Create empty TogetherAI rerank options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `rankFields`.
    pub fn with_rank_fields(mut self, rank_fields: Vec<String>) -> Self {
        self.rank_fields = Some(rank_fields);
        self
    }
}

/// Typed TogetherAI image-model options stored under `provider_options_map["togetherai"]`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TogetherAiImageOptions {
    /// Number of image-generation steps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// Guidance scale used by the TogetherAI image model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance: Option<f64>,
    /// Negative prompt passed through to the provider-owned image endpoint.
    #[serde(skip_serializing_if = "Option::is_none", alias = "negativePrompt")]
    pub negative_prompt: Option<String>,
    /// Whether to disable the safety checker.
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "disableSafetyChecker"
    )]
    pub disable_safety_checker: Option<bool>,
    /// Additional TogetherAI image parameters.
    #[serde(flatten)]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl TogetherAiImageOptions {
    /// Create empty TogetherAI image options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the step count.
    pub fn with_steps(mut self, steps: u32) -> Self {
        self.steps = Some(steps);
        self
    }

    /// Set the guidance scale.
    pub fn with_guidance(mut self, guidance: f64) -> Self {
        self.guidance = Some(guidance);
        self
    }

    /// Set the negative prompt.
    pub fn with_negative_prompt(mut self, negative_prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative_prompt.into());
        self
    }

    /// Control the TogetherAI safety checker.
    pub fn with_disable_safety_checker(mut self, disable_safety_checker: bool) -> Self {
        self.disable_safety_checker = Some(disable_safety_checker);
        self
    }

    /// Add an extra provider-owned image field.
    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_fields.insert(key.into(), value);
        self
    }
}

/// AI SDK-aligned alias for TogetherAI image-model options.
pub type TogetherAiImageModelOptions = TogetherAiImageOptions;

/// Deprecated AI SDK compatibility alias.
#[deprecated(
    since = "0.11.0-beta.6",
    note = "Use `TogetherAiImageModelOptions` instead."
)]
pub type TogetherAiImageProviderOptions = TogetherAiImageOptions;

/// AI SDK-aligned alias for TogetherAI reranking-model options.
pub type TogetherAiRerankingModelOptions = TogetherAiRerankOptions;

/// Deprecated AI SDK compatibility alias.
#[deprecated(
    since = "0.11.0-beta.6",
    note = "Use `TogetherAiRerankingModelOptions` instead."
)]
pub type TogetherAiRerankingOptions = TogetherAiRerankOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_matches_expected_shape() {
        let value = serde_json::to_value(
            TogetherAiRerankOptions::new().with_rank_fields(vec!["example".to_string()]),
        )
        .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "rankFields": ["example"]
            })
        );
    }

    #[test]
    fn image_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            TogetherAiImageOptions::new()
                .with_steps(28)
                .with_guidance(3.5)
                .with_negative_prompt("blurry")
                .with_disable_safety_checker(true)
                .with_extra_field("style", serde_json::json!("anime")),
        )
        .expect("serialize image options");

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
    fn image_options_accept_camel_case_aliases() {
        let options: TogetherAiImageOptions = serde_json::from_value(serde_json::json!({
            "negativePrompt": "grainy",
            "disableSafetyChecker": true
        }))
        .expect("deserialize image options");

        assert_eq!(options.negative_prompt.as_deref(), Some("grainy"));
        assert_eq!(options.disable_safety_checker, Some(true));
    }

    #[test]
    #[allow(deprecated)]
    fn ai_sdk_style_aliases_resolve_to_same_types() {
        let image: TogetherAiImageModelOptions = TogetherAiImageOptions::new().with_steps(8);
        let image_provider: TogetherAiImageProviderOptions = image.clone();
        let rerank: TogetherAiRerankingModelOptions =
            TogetherAiRerankOptions::new().with_rank_fields(vec!["title".to_string()]);
        let reranking: TogetherAiRerankingOptions = rerank.clone();

        assert_eq!(image.steps, image_provider.steps);
        assert_eq!(rerank.rank_fields, reranking.rank_fields);
    }
}
