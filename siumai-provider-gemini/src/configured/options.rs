use serde::{Deserialize, Serialize};
use siumai_core::{ModelFamily, TypedProviderOptions};

/// Aspect ratios accepted by current Gemini image models through Interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoogleImageAspectRatio {
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

impl GoogleImageAspectRatio {
    pub(crate) const fn as_wire(self) -> &'static str {
        match self {
            Self::Square => "1:1",
            Self::PortraitOneFour => "1:4",
            Self::PortraitOneEight => "1:8",
            Self::PortraitTwoThree => "2:3",
            Self::LandscapeThreeTwo => "3:2",
            Self::PortraitThreeFour => "3:4",
            Self::LandscapeFourOne => "4:1",
            Self::LandscapeFourThree => "4:3",
            Self::PortraitFourFive => "4:5",
            Self::LandscapeFiveFour => "5:4",
            Self::LandscapeEightOne => "8:1",
            Self::PortraitNineSixteen => "9:16",
            Self::LandscapeSixteenNine => "16:9",
            Self::Ultrawide => "21:9",
        }
    }

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
pub enum GoogleImageSize {
    #[serde(rename = "512")]
    Pixels512,
    #[serde(rename = "1K")]
    OneK,
    #[serde(rename = "2K")]
    TwoK,
    #[serde(rename = "4K")]
    FourK,
}

impl GoogleImageSize {
    pub(crate) const fn as_wire(self) -> &'static str {
        match self {
            Self::Pixels512 => "512",
            Self::OneK => "1K",
            Self::TwoK => "2K",
            Self::FourK => "4K",
        }
    }
}

/// Provider-owned options for Gemini image generation through Interactions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GoogleImageOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<GoogleImageAspectRatio>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_size: Option<GoogleImageSize>,
}

impl GoogleImageOptions {
    pub const fn new() -> Self {
        Self {
            aspect_ratio: None,
            image_size: None,
        }
    }

    pub const fn with_aspect_ratio(mut self, aspect_ratio: GoogleImageAspectRatio) -> Self {
        self.aspect_ratio = Some(aspect_ratio);
        self
    }

    pub const fn with_image_size(mut self, image_size: GoogleImageSize) -> Self {
        self.image_size = Some(image_size);
        self
    }
}

impl TypedProviderOptions for GoogleImageOptions {
    const NAMESPACE: &'static str = "google";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Image;
    const API_MODE: Option<&'static str> = Some("interactions-image");
}
