//! `MiniMax`-specific response metadata.
//!
//! MiniMax chat currently reuses the Anthropic wire format, so the typed metadata surface reuses
//! the Anthropic-shaped metadata payload while normalizing the top-level provider key to
//! `provider_metadata["minimax"]`.

/// MiniMax citations reuse the Anthropic-compatible citation payload.
pub type MinimaxCitation =
    siumai_protocol_anthropic::provider_metadata::anthropic::AnthropicCitation;
/// MiniMax citation blocks reuse the Anthropic-compatible citation payload.
pub type MinimaxCitationsBlock =
    siumai_protocol_anthropic::provider_metadata::anthropic::AnthropicCitationsBlock;
/// MiniMax server-tool usage reuse the Anthropic-compatible payload.
pub type MinimaxServerToolUse =
    siumai_protocol_anthropic::provider_metadata::anthropic::AnthropicServerToolUse;
/// MiniMax tool caller metadata reuses the Anthropic-compatible payload.
pub type MinimaxToolCaller =
    siumai_protocol_anthropic::provider_metadata::anthropic::AnthropicToolCaller;
/// MiniMax tool-call metadata reuses the Anthropic-compatible payload.
pub type MinimaxToolCallMetadata =
    siumai_protocol_anthropic::provider_metadata::anthropic::AnthropicToolCallMetadata;
/// A normalized source entry reused from the Anthropic-compatible family.
pub type MinimaxSource = siumai_protocol_anthropic::provider_metadata::anthropic::AnthropicSource;
/// MiniMax-specific metadata from chat responses.
pub type MinimaxMetadata =
    siumai_protocol_anthropic::provider_metadata::anthropic::AnthropicMetadata;

/// Typed helper for MiniMax metadata extraction from `ChatResponse`.
pub trait MinimaxChatResponseExt {
    fn minimax_metadata(&self) -> Option<MinimaxMetadata>;
}

impl MinimaxChatResponseExt for crate::types::ChatResponse {
    fn minimax_metadata(&self) -> Option<MinimaxMetadata> {
        let provider_metadata = self.provider_metadata.as_ref()?;
        let meta = provider_metadata
            .get("minimax")
            .or_else(|| provider_metadata.get("anthropic"))?
            .clone();
        serde_json::from_value(meta).ok()
    }
}

/// Typed helper for MiniMax metadata extraction from `ContentPart::ToolCall`.
pub trait MinimaxContentPartExt {
    fn minimax_tool_call_metadata(&self) -> Option<MinimaxToolCallMetadata>;
}

impl MinimaxContentPartExt for crate::types::ContentPart {
    fn minimax_tool_call_metadata(&self) -> Option<MinimaxToolCallMetadata> {
        use crate::types::ContentPart;

        let ContentPart::ToolCall {
            provider_metadata: Some(metadata),
            ..
        } = self
        else {
            return None;
        };

        let meta = metadata
            .get("minimax")
            .or_else(|| metadata.get("anthropic"))?;
        serde_json::from_value(meta.clone()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn production_source() -> &'static str {
        include_str!("minimax.rs")
            .split_once("#[cfg(test)]")
            .expect("test marker should exist")
            .0
    }

    #[test]
    fn minimax_provider_metadata_source_does_not_read_request_provider_options() {
        let source = production_source();

        assert!(
            !source.contains("providerOptions"),
            "MiniMax provider_metadata typed views must not read request-side providerOptions"
        );
        assert!(
            !source.contains("provider_options"),
            "MiniMax provider_metadata typed views must not read request-side provider_options"
        );
        assert!(
            !source.contains("provider_options_map"),
            "MiniMax provider_metadata typed views must not read request provider options maps"
        );
    }

    #[test]
    fn minimax_metadata_parses_normalized_provider_key() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert(
            "sources".to_string(),
            serde_json::json!([
                {
                    "id": "src_1",
                    "source_type": "document",
                    "title": "Example",
                    "filename": "example.pdf"
                }
            ]),
        );
        inner.insert("thinking".to_string(), serde_json::json!("step by step"));

        let mut outer = HashMap::new();
        outer.insert(
            "minimax".to_string(),
            serde_json::Value::Object(inner.into_iter().collect()),
        );
        resp.provider_metadata = Some(outer);

        let meta = resp.minimax_metadata().expect("minimax metadata");
        assert_eq!(meta.sources.as_ref().map(Vec::len), Some(1));
        assert_eq!(meta.thinking.as_deref(), Some("step by step"));
    }

    #[test]
    fn minimax_metadata_accepts_legacy_anthropic_provider_key() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert("thinking".to_string(), serde_json::json!("legacy"));

        let mut outer = HashMap::new();
        outer.insert(
            "anthropic".to_string(),
            serde_json::Value::Object(inner.into_iter().collect()),
        );
        resp.provider_metadata = Some(outer);

        let meta = resp.minimax_metadata().expect("legacy minimax metadata");
        assert_eq!(meta.thinking.as_deref(), Some("legacy"));
    }

    #[test]
    fn minimax_tool_call_metadata_accepts_normalized_and_legacy_provider_keys() {
        let normalized = crate::types::ContentPart::ToolCall {
            tool_call_id: "call_1".to_string(),
            tool_name: "rollDie".to_string(),
            arguments: serde_json::json!({"player":"player1"}),
            provider_executed: None,
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "minimax".to_string(),
                serde_json::json!({
                    "caller": {
                        "type": "direct",
                        "toolId": "tool_123"
                    }
                }),
            )])),
        };

        let meta = normalized
            .minimax_tool_call_metadata()
            .expect("normalized minimax tool-call metadata");
        assert_eq!(
            meta.caller
                .as_ref()
                .and_then(|caller| caller.kind.as_deref()),
            Some("direct")
        );
        assert_eq!(
            meta.caller
                .as_ref()
                .and_then(|caller| caller.tool_id.as_deref()),
            Some("tool_123")
        );

        let legacy = crate::types::ContentPart::ToolCall {
            tool_call_id: "call_2".to_string(),
            tool_name: "rollDie".to_string(),
            arguments: serde_json::json!({"player":"player2"}),
            provider_executed: None,
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "caller": {
                        "type": "code_execution_20250825",
                        "toolId": "tool_legacy"
                    }
                }),
            )])),
        };

        let meta = legacy
            .minimax_tool_call_metadata()
            .expect("legacy minimax tool-call metadata");
        assert_eq!(
            meta.caller
                .as_ref()
                .and_then(|caller| caller.kind.as_deref()),
            Some("code_execution_20250825")
        );
        assert_eq!(
            meta.caller
                .as_ref()
                .and_then(|caller| caller.tool_id.as_deref()),
            Some("tool_legacy")
        );
    }
}
