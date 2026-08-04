use serde::{Deserialize, Serialize};
use siumai_core::TypedProviderOptions;

/// Aspect ratios supported by Imagen on the Gemini API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoogleImagenAspectRatio {
    #[serde(rename = "1:1")]
    Square,
    #[serde(rename = "3:4")]
    PortraitThreeFour,
    #[serde(rename = "4:3")]
    LandscapeFourThree,
    #[serde(rename = "9:16")]
    PortraitNineSixteen,
    #[serde(rename = "16:9")]
    LandscapeSixteenNine,
}

impl GoogleImagenAspectRatio {
    pub(crate) const fn as_wire(self) -> &'static str {
        match self {
            Self::Square => "1:1",
            Self::PortraitThreeFour => "3:4",
            Self::LandscapeFourThree => "4:3",
            Self::PortraitNineSixteen => "9:16",
            Self::LandscapeSixteenNine => "16:9",
        }
    }
}

/// Person-generation policy supported by Imagen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoogleImagenPersonGeneration {
    DontAllow,
    AllowAdult,
    AllowAll,
}

impl GoogleImagenPersonGeneration {
    pub(crate) const fn as_wire(self) -> &'static str {
        match self {
            Self::DontAllow => "dont_allow",
            Self::AllowAdult => "allow_adult",
            Self::AllowAll => "allow_all",
        }
    }
}

/// Provider-owned options for the Gemini API Imagen `:predict` mode.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GoogleImagenOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<GoogleImagenAspectRatio>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub person_generation: Option<GoogleImagenPersonGeneration>,
}

impl TypedProviderOptions for GoogleImagenOptions {
    const NAMESPACE: &'static str = "google";
}
