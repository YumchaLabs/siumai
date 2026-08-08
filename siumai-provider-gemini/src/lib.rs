//! Product-level Google Gemini provider for Siumai.
#![deny(unsafe_code)]

mod http;
mod image;
mod language;
mod models;
mod options;
mod profile;
mod provider;

pub use image::GeminiImageModel;
pub use language::GeminiLanguageModel;
pub use models::{
    GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_LITE_IMAGE, GEMINI_3_5_FLASH, GEMINI_3_5_FLASH_LITE,
    GEMINI_3_6_FLASH, GEMINI_3_PRO_IMAGE, current_image_models, current_interactions_models,
};
pub use options::{
    GeminiImageAspectRatio, GeminiImageOptions, GeminiImageSize, GeminiInteractionStorage,
    GeminiInteractionsOptions, GeminiThinkingLevel, GeminiThinkingSummaries,
};
pub use profile::{GeminiProfile, GeminiProfileError};
pub use provider::{GeminiConfigError, GeminiCredential, GeminiProvider, GeminiProviderBuilder};
pub use siumai_protocol_gemini::interactions::DecodedInteraction as GeminiDecodedInteraction;
