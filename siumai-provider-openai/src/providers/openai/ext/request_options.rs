use crate::types::ChatRequest;

/// OpenAI request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait OpenAiChatRequestExt {
    /// Convenience: attach OpenAI-specific options to `provider_options_map["openai"]`.
    fn with_openai_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl OpenAiChatRequestExt for ChatRequest {
    fn with_openai_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("openai", value)
    }
}
