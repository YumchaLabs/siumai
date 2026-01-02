use crate::types::ChatRequest;

/// Groq request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait GroqChatRequestExt {
    /// Convenience: attach Groq-specific options to `provider_options_map["groq"]`.
    fn with_groq_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl GroqChatRequestExt for ChatRequest {
    fn with_groq_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("groq", value)
    }
}
