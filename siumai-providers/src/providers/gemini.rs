//! Gemini provider (re-export).
//!
//! The Gemini implementation lives in the provider crate `siumai-provider-gemini`.

pub use siumai_provider_gemini::providers::gemini::*;

// Provider-owned typed options and metadata (kept out of `siumai-core`).
pub use siumai_provider_gemini::provider_metadata::gemini::{
    GeminiChatResponseExt, GeminiMetadata, GeminiSource,
};
pub use siumai_provider_gemini::provider_options::gemini::{
    GeminiHarmBlockThreshold, GeminiHarmCategory, GeminiOptions, GeminiResponseModality,
    GeminiSafetySetting, GeminiThinkingConfig, GeminiThinkingLevel,
};
