//! Anthropic-specific response metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Anthropic server tool usage (provider-hosted tools) reported in `usage.server_tool_use`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicServerToolUse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_requests: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_fetch_requests: Option<u32>,
}

/// A single Anthropic citation object.
///
/// Anthropic currently uses multiple citation shapes; we preserve unknown fields via `data`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicCitation {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub data: HashMap<String, serde_json::Value>,
}

/// Citations grouped by the content block that emitted them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicCitationsBlock {
    pub content_block_index: u32,
    pub citations: Vec<AnthropicCitation>,
}

/// A normalized "source" entry (Vercel-aligned), typically produced from web search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicSource {
    /// Source identifier (stable within a response).
    pub id: String,

    /// Source type (e.g. "url" for web search results, "document" for citations).
    pub source_type: String,

    /// Source URL (when available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Optional title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Media type for document sources (e.g. "application/pdf", "text/plain").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,

    /// Filename for document sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,

    /// Optional page age as reported by Anthropic (kept as a string for portability).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,

    /// Provider-encrypted content (if present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,

    /// Tool call id that produced this source (when applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Provider-specific metadata for the source (e.g. citation offsets, file/container ids).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
}

/// Anthropic-specific metadata from chat responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicMetadata {
    /// Number of input tokens used to create the cache entry
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,

    /// Number of input tokens read from the cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,

    /// Number of tokens used for thinking/reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<u32>,

    /// Raw thinking content (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,

    /// Thinking signature captured from Anthropic reasoning blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,

    /// Redacted thinking payload captured from Anthropic reasoning blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted_thinking_data: Option<String>,

    /// Service tier (when returned by Anthropic).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Provider-hosted tool usage counters (e.g. web search requests).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<AnthropicServerToolUse>,

    /// Citations emitted by Anthropic (grouped by content block).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<AnthropicCitationsBlock>>,

    /// Sources extracted from provider-hosted tool results (Vercel-aligned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<AnthropicSource>>,

    /// Container information (code execution / skills; Vercel-aligned provider metadata shape).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<AnthropicContainerMetadata>,

    /// Context management response (Vercel-aligned provider metadata shape).
    #[serde(skip_serializing_if = "Option::is_none", rename = "contextManagement")]
    pub context_management: Option<serde_json::Value>,
}

/// Container metadata returned by Anthropic when container tools are used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicContainerMetadata {
    /// Identifier for the container used in this request.
    pub id: Option<String>,

    /// Container expiration time as an RFC3339 string.
    #[serde(skip_serializing_if = "Option::is_none", rename = "expiresAt")]
    pub expires_at: Option<String>,

    /// Skills loaded in the container (when applicable).
    pub skills: Option<Vec<AnthropicContainerSkill>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicContainerSkill {
    /// Skill type ("anthropic" or "custom").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,

    /// Skill id.
    #[serde(skip_serializing_if = "Option::is_none", rename = "skillId")]
    pub skill_id: Option<String>,

    /// Skill version (or "latest").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Anthropic tool caller information carried on tool-call content parts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicToolCaller {
    /// Caller type (for example `direct` or `code_execution_20250825`).
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub kind: Option<String>,

    /// Provider tool id when the tool call was triggered programmatically.
    #[serde(skip_serializing_if = "Option::is_none", alias = "toolId")]
    pub tool_id: Option<String>,
}

/// Anthropic-specific metadata attached to `ContentPart::ToolCall`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicToolCallMetadata {
    /// Tool caller information reported by Anthropic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller: Option<AnthropicToolCaller>,
}

/// Anthropic-specific metadata attached to `ContentPart::Text`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicTextContentPartMetadata {
    /// Raw citations reported on the text block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<AnthropicCitation>>,
}

impl crate::types::provider_metadata::FromMetadata for AnthropicMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

impl crate::types::provider_metadata::FromMetadata for AnthropicToolCallMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

/// Typed helper for Anthropic metadata extraction from `ChatResponse`.
pub trait AnthropicChatResponseExt {
    fn anthropic_metadata(&self) -> Option<AnthropicMetadata>;
}

impl AnthropicChatResponseExt for crate::types::ChatResponse {
    fn anthropic_metadata(&self) -> Option<AnthropicMetadata> {
        use crate::types::provider_metadata::FromMetadata;
        let meta = self.provider_metadata.as_ref()?.get("anthropic")?;
        AnthropicMetadata::from_metadata(meta)
    }
}

/// Typed helper for Anthropic metadata extraction from `ContentPart::ToolCall`.
pub trait AnthropicContentPartExt {
    fn anthropic_tool_call_metadata(&self) -> Option<AnthropicToolCallMetadata>;
    fn anthropic_text_metadata(&self) -> Option<AnthropicTextContentPartMetadata>;
}

impl AnthropicContentPartExt for crate::types::ContentPart {
    fn anthropic_tool_call_metadata(&self) -> Option<AnthropicToolCallMetadata> {
        use crate::types::ContentPart;

        let ContentPart::ToolCall {
            provider_metadata: Some(metadata),
            ..
        } = self
        else {
            return None;
        };

        let meta = metadata.get("anthropic")?;
        serde_json::from_value(meta.clone()).ok()
    }

    fn anthropic_text_metadata(&self) -> Option<AnthropicTextContentPartMetadata> {
        use crate::types::ContentPart;

        let ContentPart::Text {
            provider_metadata: Some(metadata),
            ..
        } = self
        else {
            return None;
        };

        let meta = metadata.get("anthropic")?;
        serde_json::from_value(meta.clone()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_metadata_parses_thinking_replay_fields() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert("thinking_signature".to_string(), serde_json::json!("sig-1"));
        inner.insert(
            "redacted_thinking_data".to_string(),
            serde_json::json!("abc123"),
        );

        let mut outer = HashMap::new();
        outer.insert("anthropic".to_string(), inner);
        resp.provider_metadata = Some(outer);

        let meta = resp.anthropic_metadata().expect("anthropic metadata");
        assert_eq!(meta.thinking_signature.as_deref(), Some("sig-1"));
        assert_eq!(meta.redacted_thinking_data.as_deref(), Some("abc123"));
    }

    #[test]
    fn anthropic_tool_call_metadata_parses_caller_fields() {
        let part = crate::types::ContentPart::ToolCall {
            tool_call_id: "call_1".to_string(),
            tool_name: "rollDie".to_string(),
            arguments: serde_json::json!({"player":"player1"}),
            provider_executed: None,
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "caller": {
                        "type": "code_execution_20250825",
                        "tool_id": "srvtoolu_1"
                    }
                }),
            )])),
        };

        let meta = part
            .anthropic_tool_call_metadata()
            .expect("anthropic tool call metadata");
        let caller = meta.caller.expect("caller");
        assert_eq!(caller.kind.as_deref(), Some("code_execution_20250825"));
        assert_eq!(caller.tool_id.as_deref(), Some("srvtoolu_1"));
    }

    #[test]
    fn anthropic_text_metadata_parses_citations() {
        let part = crate::types::ContentPart::Text {
            text: "grounded answer".to_string(),
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://example.com",
                            "title": "Example"
                        }
                    ]
                }),
            )])),
        };

        let meta = part
            .anthropic_text_metadata()
            .expect("anthropic text metadata");
        let citations = meta.citations.expect("citations");
        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0].kind, "web_search_result_location");
        assert_eq!(
            citations[0].data.get("url"),
            Some(&serde_json::json!("https://example.com"))
        );
    }
}
