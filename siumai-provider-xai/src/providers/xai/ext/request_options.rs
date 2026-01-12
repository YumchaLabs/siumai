use crate::types::ChatRequest;

/// xAI request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait XaiChatRequestExt {
    /// Convenience: attach xAI-specific options to `provider_options_map["xai"]`.
    fn with_xai_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl XaiChatRequestExt for ChatRequest {
    fn with_xai_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("xai", value)
    }
}
