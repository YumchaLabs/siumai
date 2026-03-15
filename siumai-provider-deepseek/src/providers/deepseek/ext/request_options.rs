use crate::types::ChatRequest;

/// DeepSeek request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait DeepSeekChatRequestExt {
    /// Convenience: attach DeepSeek-specific options to `provider_options_map["deepseek"]`.
    fn with_deepseek_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl DeepSeekChatRequestExt for ChatRequest {
    fn with_deepseek_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("deepseek", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::DeepSeekOptions;
    use crate::types::ChatMessage;

    #[test]
    fn chat_request_ext_attaches_deepseek_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_deepseek_options(DeepSeekOptions::new().with_reasoning_budget(4096));

        let value = request
            .provider_options_map
            .get("deepseek")
            .expect("deepseek options present");
        assert_eq!(value["enable_reasoning"], serde_json::json!(true));
        assert_eq!(value["reasoning_budget"], serde_json::json!(4096));
    }
}
