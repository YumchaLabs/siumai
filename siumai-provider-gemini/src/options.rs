use serde::{Deserialize, Serialize};
use siumai_core::{ModelFamily, TypedProviderOptions};

/// Whether one Gemini Interactions call may be stored by the provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GeminiInteractionStorage {
    /// Send `store: false`; this is Siumai's privacy-preserving default.
    #[default]
    Disabled,
    /// Send `store: true`; retrieval and background lifecycle APIs remain provider-owned.
    Enabled,
}

/// Thinking depth accepted by stable Gemini Interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GeminiThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

/// Whether Gemini should include provider thought summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GeminiThinkingSummaries {
    Auto,
    None,
}

/// Provider-owned options for stable Gemini Interactions language calls.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GeminiInteractionsOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<GeminiInteractionStorage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<GeminiThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_summaries: Option<GeminiThinkingSummaries>,
}

impl GeminiInteractionsOptions {
    pub const fn new() -> Self {
        Self {
            storage: None,
            thinking_level: None,
            thinking_summaries: None,
        }
    }

    pub const fn with_storage(mut self, storage: GeminiInteractionStorage) -> Self {
        self.storage = Some(storage);
        self
    }

    pub const fn with_thinking_level(mut self, level: GeminiThinkingLevel) -> Self {
        self.thinking_level = Some(level);
        self
    }

    pub const fn with_thinking_summaries(mut self, summaries: GeminiThinkingSummaries) -> Self {
        self.thinking_summaries = Some(summaries);
        self
    }
}

impl TypedProviderOptions for GeminiInteractionsOptions {
    const NAMESPACE: &'static str = "google";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some("interactions");
}

/// Exact aspect-ratio values accepted by the Gemini Interactions image wire schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GeminiImageAspectRatio {
    #[serde(rename = "1:1")]
    Square,
    #[serde(rename = "1:4")]
    PortraitOneFour,
    #[serde(rename = "1:8")]
    PortraitOneEight,
    #[serde(rename = "2:3")]
    PortraitTwoThree,
    #[serde(rename = "3:2")]
    LandscapeThreeTwo,
    #[serde(rename = "3:4")]
    PortraitThreeFour,
    #[serde(rename = "4:1")]
    LandscapeFourOne,
    #[serde(rename = "4:3")]
    LandscapeFourThree,
    #[serde(rename = "4:5")]
    PortraitFourFive,
    #[serde(rename = "5:4")]
    LandscapeFiveFour,
    #[serde(rename = "8:1")]
    LandscapeEightOne,
    #[serde(rename = "9:16")]
    PortraitNineSixteen,
    #[serde(rename = "16:9")]
    LandscapeSixteenNine,
    #[serde(rename = "21:9")]
    Ultrawide,
}

/// Exact output-resolution tiers accepted by the Gemini Interactions image wire schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GeminiImageSize {
    #[serde(rename = "512")]
    Pixels512,
    #[serde(rename = "1K")]
    OneK,
    #[serde(rename = "2K")]
    TwoK,
    #[serde(rename = "4K")]
    FourK,
}

/// Provider-owned options for Gemini image generation through Interactions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GeminiImageOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<GeminiImageAspectRatio>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_size: Option<GeminiImageSize>,
}

impl GeminiImageOptions {
    pub const fn new() -> Self {
        Self {
            aspect_ratio: None,
            image_size: None,
        }
    }

    pub const fn with_aspect_ratio(mut self, aspect_ratio: GeminiImageAspectRatio) -> Self {
        self.aspect_ratio = Some(aspect_ratio);
        self
    }

    pub const fn with_image_size(mut self, image_size: GeminiImageSize) -> Self {
        self.image_size = Some(image_size);
        self
    }
}

impl TypedProviderOptions for GeminiImageOptions {
    const NAMESPACE: &'static str = "google";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Image;
    const API_MODE: Option<&'static str> = Some("interactions");
}
