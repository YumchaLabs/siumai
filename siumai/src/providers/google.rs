//! Curated Google Gemini provider facade.

pub use siumai_provider_gemini::{
    GeminiConfigError, GeminiCredential, GeminiDecodedInteraction, GeminiImageModel,
    GeminiLanguageModel, GeminiProfile, GeminiProfileError, GeminiProvider, GeminiProviderBuilder,
};

pub mod models {
    pub use siumai_provider_gemini::{
        GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_LITE_IMAGE, GEMINI_3_5_FLASH,
        GEMINI_3_5_FLASH_LITE, GEMINI_3_6_FLASH, GEMINI_3_PRO_IMAGE, current_image_models,
        current_interactions_models,
    };
}

pub mod options {
    pub use siumai_provider_gemini::{
        GeminiImageAspectRatio, GeminiImageOptions, GeminiImageSize, GeminiInteractionStorage,
        GeminiInteractionsOptions, GeminiThinkingLevel, GeminiThinkingSummaries,
    };
}
