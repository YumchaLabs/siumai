use crate::types::ChatRequest;

/// MiniMax request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait MinimaxChatRequestExt {
    /// Convenience: attach MiniMax-specific options to `provider_options_map["minimax"]`.
    fn with_minimax_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl MinimaxChatRequestExt for ChatRequest {
    fn with_minimax_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        self.with_provider_option("minimax", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::{MinimaxOptions, MinimaxThinking};
    use crate::types::ChatMessage;

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn minimax_request_option_extension_source_does_not_read_response_metadata() {
        let source = include_str!("request_options.rs");
        let request_source =
            source_section(source, "pub trait MinimaxChatRequestExt", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "MiniMax request option extension helpers must stay request-only"
            );
        }
    }

    #[test]
    fn chat_request_ext_attaches_minimax_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_minimax_options(MinimaxOptions::new().with_thinking(MinimaxThinking::Adaptive));

        let value = request
            .provider_options_map
            .get("minimax")
            .expect("minimax options present");
        assert_eq!(value["thinking"], serde_json::json!({ "type": "adaptive" }));
    }
}
