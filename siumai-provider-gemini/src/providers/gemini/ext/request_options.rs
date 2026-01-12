use crate::types::ChatRequest;

/// Gemini request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait GeminiChatRequestExt {
    /// Convenience: attach Gemini-specific options to `provider_options_map["gemini"]`.
    fn with_gemini_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl GeminiChatRequestExt for ChatRequest {
    fn with_gemini_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("gemini", value)
    }
}
