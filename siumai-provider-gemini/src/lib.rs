//! siumai-provider-gemini
//!
//! Current Google image generation through the Gemini Interactions API.
#![deny(unsafe_code)]

mod configured;

pub use configured::{
    GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_LITE_IMAGE, GEMINI_3_PRO_IMAGE, GoogleCredential,
    GoogleImageAspectRatio, GoogleImageConfigError, GoogleImageModel, GoogleImageOptions,
    GoogleImageProfile, GoogleImageProfileError, GoogleImageProvider, GoogleImageProviderBuilder,
    GoogleImageSize, current_models,
};
