use crate::types::ChatRequest;

/// OpenRouter request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait OpenRouterChatRequestExt {
    /// Convenience: attach OpenRouter-specific options to `provider_options_map["openrouter"]`.
    fn with_openrouter_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl OpenRouterChatRequestExt for ChatRequest {
    fn with_openrouter_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("openrouter", value)
    }
}

/// Perplexity request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait PerplexityChatRequestExt {
    /// Convenience: attach Perplexity-specific options to `provider_options_map["perplexity"]`.
    fn with_perplexity_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl PerplexityChatRequestExt for ChatRequest {
    fn with_perplexity_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("perplexity", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::{
        OpenRouterOptions, OpenRouterTransform, PerplexityOptions, PerplexitySearchMode,
    };
    use crate::types::ChatMessage;

    #[test]
    fn chat_request_ext_attaches_openrouter_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_openrouter_options(
                OpenRouterOptions::new()
                    .with_transform(OpenRouterTransform::MiddleOut)
                    .with_param("someVendorParam", serde_json::json!(true)),
            );

        let value = request
            .provider_options_map
            .get("openrouter")
            .expect("openrouter options present");
        assert_eq!(value["transforms"], serde_json::json!(["middle-out"]));
        assert_eq!(value["someVendorParam"], serde_json::json!(true));
    }

    #[test]
    fn chat_request_ext_attaches_perplexity_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_perplexity_options(
                PerplexityOptions::new()
                    .with_search_mode(PerplexitySearchMode::Academic)
                    .with_return_images(true),
            );

        let value = request
            .provider_options_map
            .get("perplexity")
            .expect("perplexity options present");
        assert_eq!(value["search_mode"], serde_json::json!("academic"));
        assert_eq!(value["return_images"], serde_json::json!(true));
    }
}
