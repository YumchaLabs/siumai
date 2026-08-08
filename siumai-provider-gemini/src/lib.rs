//! Product-level Google Gemini provider for Siumai.
#![deny(unsafe_code)]

mod image;
mod models;
mod options;
mod profile;
mod provider;

pub use image::GeminiImageModel;
pub use models::{
    GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_LITE_IMAGE, GEMINI_3_PRO_IMAGE, current_image_models,
};
pub use options::{GeminiImageAspectRatio, GeminiImageOptions, GeminiImageSize};
pub use profile::{GeminiProfile, GeminiProfileError};
pub use provider::{GeminiConfigError, GeminiCredential, GeminiProvider, GeminiProviderBuilder};
