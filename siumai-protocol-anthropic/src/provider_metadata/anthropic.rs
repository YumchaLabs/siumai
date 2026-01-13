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

impl crate::types::provider_metadata::FromMetadata for AnthropicMetadata {
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
