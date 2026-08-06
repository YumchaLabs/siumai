//! Curated Google Gemini Interactions image provider facade.

pub use siumai_provider_gemini::{
    GoogleCredential, GoogleImageConfigError, GoogleImageModel, GoogleImageProfile,
    GoogleImageProfileError, GoogleImageProvider, GoogleImageProviderBuilder,
};

pub mod models {
    pub use siumai_provider_gemini::{
        GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_LITE_IMAGE, GEMINI_3_PRO_IMAGE, current_models,
    };
}

pub mod options {
    pub use siumai_provider_gemini::{GoogleImageAspectRatio, GoogleImageOptions, GoogleImageSize};
}
