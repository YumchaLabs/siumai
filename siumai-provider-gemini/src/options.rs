use serde::{Deserialize, Serialize};
use siumai_core::{ModelFamily, TypedProviderOptions};

/// Aspect ratios accepted by current Gemini image models through Interactions.
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

impl GeminiImageAspectRatio {
    pub(crate) const fn is_extended(self) -> bool {
        matches!(
            self,
            Self::PortraitOneFour
                | Self::PortraitOneEight
                | Self::LandscapeFourOne
                | Self::LandscapeEightOne
        )
    }
}

/// Output resolution tiers accepted by current Gemini image models.
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
