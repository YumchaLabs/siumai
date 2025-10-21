//! Google Gemini API Standard
//!
//! This module implements the Google Gemini API format.
//! Note: This is provider-specific and not widely adopted by other providers.

use crate::provider_core::ChatTransformers;

/// Gemini Chat API Standard
///
/// Note: Currently this is just a placeholder. Gemini uses a unique API format
/// that is not widely adopted by other providers, so it remains provider-specific.
#[derive(Clone)]
pub struct GeminiChatStandard;

impl GeminiChatStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_transformers(&self, provider_id: &str) -> ChatTransformers {
        // Gemini uses its own transformers from providers/gemini/
        // This is just a placeholder for consistency
        let _ = provider_id;
        unimplemented!("Gemini standard is provider-specific")
    }
}

impl Default for GeminiChatStandard {
    fn default() -> Self {
        Self::new()
    }
}
