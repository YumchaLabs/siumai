use crate::types::ChatRequest;

/// Ollama request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait OllamaChatRequestExt {
    /// Convenience: attach Ollama-specific options to `provider_options_map["ollama"]`.
    fn with_ollama_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl OllamaChatRequestExt for ChatRequest {
    fn with_ollama_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("ollama", value)
    }
}
