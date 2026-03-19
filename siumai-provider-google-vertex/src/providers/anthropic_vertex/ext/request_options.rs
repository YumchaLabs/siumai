use crate::types::ChatRequest;

/// Anthropic-on-Vertex request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait VertexAnthropicChatRequestExt {
    /// Convenience: attach Anthropic-on-Vertex options to `provider_options_map["anthropic"]`.
    fn with_anthropic_vertex_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl VertexAnthropicChatRequestExt for ChatRequest {
    fn with_anthropic_vertex_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("anthropic", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::anthropic_vertex::{
        VertexAnthropicOptions, VertexAnthropicThinkingMode,
    };
    use crate::types::ChatMessage;

    #[test]
    fn chat_request_ext_attaches_anthropic_vertex_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_anthropic_vertex_options(
                VertexAnthropicOptions::new()
                    .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048))),
            );

        let value = request
            .provider_options_map
            .get("anthropic")
            .expect("anthropic options present");
        assert_eq!(value["thinking_mode"]["enabled"], serde_json::json!(true));
        assert_eq!(
            value["thinking_mode"]["thinking_budget"],
            serde_json::json!(2048)
        );
    }
}
