use crate::types::ChatRequest;

/// MiniMaxi request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait MinimaxiChatRequestExt {
    /// Convenience: attach MiniMaxi-specific options to `provider_options_map["minimaxi"]`.
    fn with_minimaxi_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl MinimaxiChatRequestExt for ChatRequest {
    fn with_minimaxi_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("minimaxi", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::MinimaxiOptions;
    use crate::types::ChatMessage;

    #[test]
    fn chat_request_ext_attaches_minimaxi_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_minimaxi_options(MinimaxiOptions::new().with_reasoning_budget(4096));

        let value = request
            .provider_options_map
            .get("minimaxi")
            .expect("minimaxi options present");
        assert_eq!(value["thinking_mode"]["enabled"], serde_json::json!(true));
        assert_eq!(
            value["thinking_mode"]["thinking_budget"],
            serde_json::json!(4096)
        );
    }
}
