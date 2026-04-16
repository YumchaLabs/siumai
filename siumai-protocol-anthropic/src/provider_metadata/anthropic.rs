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
    /// Raw Anthropic usage object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,

    /// Number of input tokens used to create the cache entry
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "cacheCreationInputTokens"
    )]
    pub cache_creation_input_tokens: Option<u32>,

    /// Stop sequence that terminated generation.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "stopSequence",
        alias = "stop_sequence"
    )]
    pub stop_sequence: Option<String>,

    /// Usage breakdown by Anthropic sampling iteration (for example compaction + message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iterations: Option<Vec<AnthropicUsageIteration>>,

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
    pub context_management: Option<AnthropicContextManagement>,
}

/// AI SDK-style alias for Anthropic message metadata.
pub type AnthropicMessageMetadata = AnthropicMetadata;

/// Anthropic usage breakdown entry for a single sampling iteration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicUsageIteration {
    /// Iteration type (`compaction` or `message`).
    pub r#type: String,

    /// Number of input tokens used during this iteration.
    #[serde(rename = "inputTokens", alias = "input_tokens")]
    pub input_tokens: u32,

    /// Number of output tokens produced during this iteration.
    #[serde(rename = "outputTokens", alias = "output_tokens")]
    pub output_tokens: u32,
}

/// Anthropic response-side context management metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicContextManagement {
    /// Context-management edits that were applied during the request.
    #[serde(rename = "appliedEdits", alias = "applied_edits", default)]
    pub applied_edits: Vec<AnthropicContextManagementEdit>,
}

/// A single context-management edit reported by Anthropic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum AnthropicContextManagementEdit {
    /// Tool-use clearing edit returned by Anthropic.
    #[serde(rename = "clear_tool_uses_20250919")]
    ClearToolUses20250919 {
        #[serde(rename = "clearedToolUses", alias = "cleared_tool_uses")]
        cleared_tool_uses: u32,
        #[serde(rename = "clearedInputTokens", alias = "cleared_input_tokens")]
        cleared_input_tokens: u32,
    },

    /// Thinking-history clearing edit returned by Anthropic.
    #[serde(rename = "clear_thinking_20251015")]
    ClearThinking20251015 {
        #[serde(rename = "clearedThinkingTurns", alias = "cleared_thinking_turns")]
        cleared_thinking_turns: u32,
        #[serde(rename = "clearedInputTokens", alias = "cleared_input_tokens")]
        cleared_input_tokens: u32,
    },

    /// Conversation compaction edit returned by Anthropic.
    #[serde(rename = "compact_20260112")]
    Compact20260112,
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

    /// Raw Anthropic provider-hosted tool name when it differs from the normalized tool name.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "serverToolName",
        alias = "server_tool_name"
    )]
    pub server_tool_name: Option<String>,

    /// MCP server name carried on Anthropic `mcp_tool_use` blocks.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "serverName",
        alias = "server_name"
    )]
    pub server_name: Option<String>,
}

/// Anthropic-specific metadata attached to `ContentPart::Text`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicTextContentPartMetadata {
    /// Raw citations reported on the text block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<AnthropicCitation>>,
}

/// Anthropic-specific metadata attached to `ContentPart::Reasoning`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicReasoningContentPartMetadata {
    /// Thinking signature required to replay Anthropic `thinking` blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,

    /// Redacted thinking payload required to replay Anthropic `redacted_thinking` blocks.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "redactedData",
        alias = "redacted_data"
    )]
    pub redacted_data: Option<String>,
}

fn anthropic_reasoning_content_part_metadata(
    part: &crate::types::ContentPart,
) -> Option<AnthropicReasoningContentPartMetadata> {
    use crate::types::ContentPart;

    let ContentPart::Reasoning {
        provider_metadata: Some(metadata),
        ..
    } = part
    else {
        return None;
    };

    let meta = metadata.get("anthropic")?;
    serde_json::from_value(meta.clone()).ok()
}

impl crate::types::provider_metadata::FromMetadata for AnthropicMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        let mut parsed: AnthropicMetadata =
            serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()?;

        if parsed.usage.is_none() {
            parsed.usage = metadata.get("usage").cloned();
        }

        let usage_obj = parsed.usage.as_ref().and_then(|usage| usage.as_object());

        if parsed.cache_creation_input_tokens.is_none() {
            parsed.cache_creation_input_tokens = usage_obj
                .and_then(|usage| usage.get("cache_creation_input_tokens"))
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok());
        }

        if parsed.cache_read_input_tokens.is_none() {
            parsed.cache_read_input_tokens = usage_obj
                .and_then(|usage| usage.get("cache_read_input_tokens"))
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok());
        }

        if parsed.iterations.is_none() {
            parsed.iterations = usage_obj
                .and_then(|usage| usage.get("iterations"))
                .and_then(|value| serde_json::from_value(value.clone()).ok());
        }

        if parsed.service_tier.is_none() {
            parsed.service_tier = usage_obj
                .and_then(|usage| usage.get("service_tier"))
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned);
        }

        if parsed.server_tool_use.is_none() {
            parsed.server_tool_use = usage_obj
                .and_then(|usage| usage.get("server_tool_use"))
                .and_then(|value| serde_json::from_value(value.clone()).ok());
        }

        Some(parsed)
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
    fn anthropic_metadata_with_key(&self, key: &str) -> Option<AnthropicMetadata>;
}

impl AnthropicChatResponseExt for crate::types::ChatResponse {
    fn anthropic_metadata(&self) -> Option<AnthropicMetadata> {
        self.anthropic_metadata_with_key("anthropic")
    }

    fn anthropic_metadata_with_key(&self, key: &str) -> Option<AnthropicMetadata> {
        use crate::types::provider_metadata::FromMetadata;

        let has_top_level = self
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get(key))
            .is_some();
        let mut parsed = self
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get(key))
            .and_then(|metadata| metadata.as_object())
            .and_then(|metadata| {
                AnthropicMetadata::from_metadata(&metadata.clone().into_iter().collect())
            })
            .unwrap_or_default();

        if let crate::types::MessageContent::MultiModal(parts) = &self.content {
            for part in parts {
                if parsed.thinking.is_none()
                    && let crate::types::ContentPart::Reasoning { text, .. } = part
                    && !text.is_empty()
                {
                    parsed.thinking = Some(text.clone());
                }

                let Some(reasoning) = anthropic_reasoning_content_part_metadata(part) else {
                    continue;
                };

                if parsed.thinking_signature.is_none() {
                    parsed.thinking_signature = reasoning.signature;
                }
                if parsed.redacted_thinking_data.is_none() {
                    parsed.redacted_thinking_data = reasoning.redacted_data;
                }
            }
        }

        if has_top_level
            || parsed.thinking.is_some()
            || parsed.thinking_signature.is_some()
            || parsed.redacted_thinking_data.is_some()
        {
            Some(parsed)
        } else {
            None
        }
    }
}

/// Typed helper for Anthropic metadata extraction from `ContentPart::ToolCall`.
pub trait AnthropicContentPartExt {
    fn anthropic_tool_call_metadata(&self) -> Option<AnthropicToolCallMetadata>;
    fn anthropic_text_metadata(&self) -> Option<AnthropicTextContentPartMetadata>;
    fn anthropic_reasoning_metadata(&self) -> Option<AnthropicReasoningContentPartMetadata>;
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

    fn anthropic_reasoning_metadata(&self) -> Option<AnthropicReasoningContentPartMetadata> {
        anthropic_reasoning_content_part_metadata(self)
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
        outer.insert(
            "anthropic".to_string(),
            serde_json::Value::Object(inner.into_iter().collect()),
        );
        resp.provider_metadata = Some(outer);

        let meta = resp.anthropic_metadata().expect("anthropic metadata");
        assert_eq!(meta.thinking_signature.as_deref(), Some("sig-1"));
        assert_eq!(meta.redacted_thinking_data.as_deref(), Some("abc123"));
    }

    #[test]
    fn anthropic_metadata_supports_custom_provider_key() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert("stopSequence".to_string(), serde_json::json!("STOP"));

        let mut outer = HashMap::new();
        outer.insert(
            "my-custom-anthropic".to_string(),
            serde_json::Value::Object(inner.into_iter().collect()),
        );
        resp.provider_metadata = Some(outer);

        assert!(resp.anthropic_metadata().is_none());
        let meta = resp
            .anthropic_metadata_with_key("my-custom-anthropic")
            .expect("custom anthropic metadata");
        assert_eq!(meta.stop_sequence.as_deref(), Some("STOP"));
    }

    #[test]
    fn anthropic_metadata_derives_extended_usage_from_nested_usage_object() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert(
            "usage".to_string(),
            serde_json::json!({
                "input_tokens": 17,
                "output_tokens": 1,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 5,
                "service_tier": "standard",
                "server_tool_use": {
                    "web_search_requests": 2
                }
            }),
        );
        inner.insert(
            "cacheCreationInputTokens".to_string(),
            serde_json::json!(10),
        );
        inner.insert("stopSequence".to_string(), serde_json::json!("</tool>"));
        inner.insert(
            "iterations".to_string(),
            serde_json::json!([
                {
                    "type": "compaction",
                    "inputTokens": 6,
                    "outputTokens": 1
                },
                {
                    "type": "message",
                    "inputTokens": 11,
                    "outputTokens": 2
                }
            ]),
        );
        inner.insert(
            "contextManagement".to_string(),
            serde_json::json!({
                "appliedEdits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "clearedToolUses": 3,
                        "clearedInputTokens": 1000
                    },
                    {
                        "type": "clear_thinking_20251015",
                        "clearedThinkingTurns": 2,
                        "clearedInputTokens": 500
                    },
                    {
                        "type": "compact_20260112"
                    }
                ]
            }),
        );

        let mut outer = HashMap::new();
        outer.insert(
            "anthropic".to_string(),
            serde_json::Value::Object(inner.into_iter().collect()),
        );
        resp.provider_metadata = Some(outer);

        let meta = resp.anthropic_metadata().expect("anthropic metadata");
        assert_eq!(meta.cache_creation_input_tokens, Some(10));
        assert_eq!(meta.stop_sequence.as_deref(), Some("</tool>"));
        assert_eq!(meta.cache_read_input_tokens, Some(5));
        assert_eq!(meta.service_tier.as_deref(), Some("standard"));
        assert_eq!(
            meta.server_tool_use
                .as_ref()
                .and_then(|usage| usage.web_search_requests),
            Some(2)
        );
        let iterations = meta.iterations.as_ref().expect("iterations");
        assert_eq!(iterations.len(), 2);
        assert_eq!(iterations[0].r#type, "compaction");
        assert_eq!(iterations[0].input_tokens, 6);
        assert_eq!(iterations[0].output_tokens, 1);
        assert_eq!(iterations[1].r#type, "message");
        assert_eq!(iterations[1].input_tokens, 11);
        assert_eq!(iterations[1].output_tokens, 2);
        let context_management = meta
            .context_management
            .as_ref()
            .expect("typed context management");
        assert_eq!(context_management.applied_edits.len(), 3);
        assert!(matches!(
            context_management.applied_edits[0],
            AnthropicContextManagementEdit::ClearToolUses20250919 {
                cleared_tool_uses: 3,
                cleared_input_tokens: 1000
            }
        ));
        assert!(matches!(
            context_management.applied_edits[1],
            AnthropicContextManagementEdit::ClearThinking20251015 {
                cleared_thinking_turns: 2,
                cleared_input_tokens: 500
            }
        ));
        assert!(matches!(
            context_management.applied_edits[2],
            AnthropicContextManagementEdit::Compact20260112
        ));
        assert_eq!(
            meta.usage
                .as_ref()
                .and_then(|usage| usage.get("cache_read_input_tokens"))
                .and_then(|value| value.as_u64()),
            Some(5)
        );
    }

    #[test]
    fn anthropic_tool_call_metadata_parses_caller_fields() {
        let part = crate::types::ContentPart::ToolCall {
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
    fn anthropic_tool_call_metadata_parses_server_tool_name() {
        let part = crate::types::ContentPart::ToolCall {
            tool_call_id: "srvtoolu_1".to_string(),
            tool_name: "tool_search".to_string(),
            arguments: serde_json::json!({"pattern":"weather"}),
            provider_executed: Some(true),
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "serverToolName": "tool_search_tool_regex"
                }),
            )])),
        };

        let meta = part
            .anthropic_tool_call_metadata()
            .expect("anthropic tool call metadata");
        assert_eq!(
            meta.server_tool_name.as_deref(),
            Some("tool_search_tool_regex")
        );
    }

    #[test]
    fn anthropic_tool_call_metadata_parses_server_name() {
        let part = crate::types::ContentPart::ToolCall {
            tool_call_id: "mcptoolu_1".to_string(),
            tool_name: "echo".to_string(),
            arguments: serde_json::json!({"message":"hello"}),
            provider_executed: Some(true),
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "serverName": "echo-prod"
                }),
            )])),
        };

        let meta = part
            .anthropic_tool_call_metadata()
            .expect("anthropic tool call metadata");
        assert_eq!(meta.server_name.as_deref(), Some("echo-prod"));
    }

    #[test]
    fn anthropic_text_metadata_parses_citations() {
        let part = crate::types::ContentPart::Text {
            text: "grounded answer".to_string(),
            provider_options: crate::types::ProviderOptionsMap::default(),
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

    #[test]
    fn anthropic_reasoning_metadata_parses_part_level_fields() {
        let part = crate::types::ContentPart::Reasoning {
            text: "internal".to_string(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "signature": "sig-1",
                    "redactedData": "abc123"
                }),
            )])),
        };

        let meta = part
            .anthropic_reasoning_metadata()
            .expect("anthropic reasoning metadata");
        assert_eq!(meta.signature.as_deref(), Some("sig-1"));
        assert_eq!(meta.redacted_data.as_deref(), Some("abc123"));
    }

    #[test]
    fn anthropic_metadata_backfills_reasoning_fields_from_content_parts() {
        let mut resp =
            crate::types::ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
                crate::types::ContentPart::Reasoning {
                    text: "internal".to_string(),
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: Some(HashMap::from([(
                        "anthropic".to_string(),
                        serde_json::json!({
                            "signature": "sig-1",
                            "redactedData": "abc123"
                        }),
                    )])),
                },
            ]));
        resp.provider_metadata = None;

        let meta = resp.anthropic_metadata().expect("anthropic metadata");
        assert_eq!(meta.thinking.as_deref(), Some("internal"));
        assert_eq!(meta.thinking_signature.as_deref(), Some("sig-1"));
        assert_eq!(meta.redacted_thinking_data.as_deref(), Some("abc123"));
    }

    #[test]
    fn message_metadata_alias_deserializes_the_same_shape() {
        let value = serde_json::json!({
            "stopSequence": "STOP",
            "iterations": [
                {
                    "type": "message",
                    "inputTokens": 10,
                    "outputTokens": 2
                }
            ]
        });

        let meta: AnthropicMessageMetadata =
            serde_json::from_value(value).expect("deserialize anthropic message metadata alias");
        assert_eq!(meta.stop_sequence.as_deref(), Some("STOP"));
        let iterations = meta.iterations.expect("iterations");
        assert_eq!(iterations.len(), 1);
        assert_eq!(iterations[0].r#type, "message");
        assert_eq!(iterations[0].input_tokens, 10);
        assert_eq!(iterations[0].output_tokens, 2);
    }
}
