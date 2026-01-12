use crate::types::ChatRequest;

/// Anthropic request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait AnthropicChatRequestExt {
    /// Convenience: attach Anthropic-specific options to `provider_options_map["anthropic"]`.
    fn with_anthropic_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl AnthropicChatRequestExt for ChatRequest {
    fn with_anthropic_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("anthropic", value)
    }
}
