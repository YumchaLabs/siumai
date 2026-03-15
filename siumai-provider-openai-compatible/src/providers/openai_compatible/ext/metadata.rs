//! Typed response metadata helpers for OpenAI-compatible vendor extensions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenRouter-specific response metadata from chat responses.
///
/// OpenRouter currently reuses the OpenAI-like metadata shape (`sources`, `logprobs`,
/// `system_fingerprint`, etc.), so the typed view is an alias of the shared
/// `OpenAiMetadata` model while still exposing a vendor-owned accessor.
pub type OpenRouterMetadata = siumai_protocol_openai::provider_metadata::openai::OpenAiMetadata;
pub type OpenRouterSource = siumai_protocol_openai::provider_metadata::openai::OpenAiSource;
pub type OpenRouterSourceMetadata =
    siumai_protocol_openai::provider_metadata::openai::OpenAiSourceMetadata;
pub type OpenRouterContentPartMetadata =
    siumai_protocol_openai::provider_metadata::openai::OpenAiContentPartMetadata;

/// Perplexity-specific response metadata from chat responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerplexityMetadata {
    /// Returned citation URLs from hosted search / answer synthesis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<String>>,

    /// Images returned by hosted search / answer synthesis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<PerplexityImage>>,

    /// Usage details that are not part of the Stable usage surface.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<PerplexityUsage>,

    /// Preserve unknown provider-specific metadata for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Perplexity image metadata entry.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerplexityImage {
    /// Returned image URL.
    pub image_url: String,

    /// Source page URL that the image came from.
    pub origin_url: String,

    /// Image height in pixels.
    pub height: u32,

    /// Image width in pixels.
    pub width: u32,

    /// Preserve unknown image-level metadata for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Perplexity-specific usage details.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerplexityUsage {
    /// Tokens consumed by returned citations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_tokens: Option<u32>,

    /// Number of search queries executed by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_search_queries: Option<u32>,

    /// Tokens consumed by provider-side reasoning work.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,

    /// Preserve unknown usage-level metadata for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl crate::types::provider_metadata::FromMetadata for PerplexityMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

/// Typed helper for OpenRouter metadata extraction from `ChatResponse`.
pub trait OpenRouterChatResponseExt {
    fn openrouter_metadata(&self) -> Option<OpenRouterMetadata>;
}

impl OpenRouterChatResponseExt for crate::types::ChatResponse {
    fn openrouter_metadata(&self) -> Option<OpenRouterMetadata> {
        use crate::types::provider_metadata::FromMetadata;
        let meta = self.provider_metadata.as_ref()?.get("openrouter")?;
        OpenRouterMetadata::from_metadata(meta)
    }
}

/// Typed helper for OpenRouter metadata extraction from `OpenRouterSource`.
pub trait OpenRouterSourceExt {
    fn openrouter_metadata(&self) -> Option<OpenRouterSourceMetadata>;
}

impl OpenRouterSourceExt for OpenRouterSource {
    fn openrouter_metadata(&self) -> Option<OpenRouterSourceMetadata> {
        let metadata = self.provider_metadata.clone()?;
        let inner = metadata.get("openrouter").cloned().unwrap_or(metadata);
        serde_json::from_value(inner).ok()
    }
}

/// Typed helper for OpenRouter metadata extraction from `ContentPart`.
pub trait OpenRouterContentPartExt {
    fn openrouter_metadata(&self) -> Option<OpenRouterContentPartMetadata>;
}

impl OpenRouterContentPartExt for crate::types::ContentPart {
    fn openrouter_metadata(&self) -> Option<OpenRouterContentPartMetadata> {
        use crate::types::ContentPart;

        let provider_metadata = match self {
            ContentPart::Text {
                provider_metadata, ..
            }
            | ContentPart::Image {
                provider_metadata, ..
            }
            | ContentPart::Audio {
                provider_metadata, ..
            }
            | ContentPart::File {
                provider_metadata, ..
            }
            | ContentPart::ToolCall {
                provider_metadata, ..
            }
            | ContentPart::ToolResult {
                provider_metadata, ..
            }
            | ContentPart::Reasoning {
                provider_metadata, ..
            } => provider_metadata.as_ref()?,
            ContentPart::Source { .. }
            | ContentPart::ToolApprovalResponse { .. }
            | ContentPart::ToolApprovalRequest { .. } => return None,
        };

        let meta = provider_metadata.get("openrouter")?;
        serde_json::from_value(meta.clone()).ok()
    }
}

/// Typed helper for Perplexity metadata extraction from `ChatResponse`.
pub trait PerplexityChatResponseExt {
    fn perplexity_metadata(&self) -> Option<PerplexityMetadata>;
}

impl PerplexityChatResponseExt for crate::types::ChatResponse {
    fn perplexity_metadata(&self) -> Option<PerplexityMetadata> {
        use crate::types::provider_metadata::FromMetadata;
        let meta = self.provider_metadata.as_ref()?.get("perplexity")?;
        PerplexityMetadata::from_metadata(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_metadata_parses_sources_and_logprobs() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert(
            "sources".to_string(),
            serde_json::json!([{
                "id": "src_1",
                "source_type": "url",
                "url": "https://example.com/rust",
                "title": "Rust"
            }]),
        );
        inner.insert(
            "logprobs".to_string(),
            serde_json::json!([{
                "token": "hello",
                "logprob": -0.1,
                "bytes": [104, 101, 108, 108, 111],
                "top_logprobs": []
            }]),
        );

        let mut outer = HashMap::new();
        outer.insert("openrouter".to_string(), inner);
        resp.provider_metadata = Some(outer);

        let meta = resp.openrouter_metadata().expect("openrouter metadata");
        assert_eq!(meta.sources.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.sources
                .as_ref()
                .and_then(|sources| sources.first())
                .map(|source| source.url.as_str()),
            Some("https://example.com/rust")
        );
        assert_eq!(
            meta.logprobs.as_ref().and_then(|value| value.get(0)),
            Some(&serde_json::json!({
                "token": "hello",
                "logprob": -0.1,
                "bytes": [104, 101, 108, 108, 111],
                "top_logprobs": []
            }))
        );
    }

    #[test]
    fn openrouter_source_metadata_parses_nested_provider_fields() {
        let source = OpenRouterSource {
            id: "src_1".to_string(),
            source_type: "url".to_string(),
            url: "https://openrouter.ai/docs".to_string(),
            title: Some("OpenRouter Docs".to_string()),
            tool_call_id: None,
            media_type: None,
            filename: None,
            provider_metadata: Some(serde_json::json!({
                "openrouter": {
                    "fileId": "file_123",
                    "containerId": "container_456",
                    "index": 2
                }
            })),
            snippet: Some("Docs".to_string()),
        };

        let meta = source
            .openrouter_metadata()
            .expect("openrouter source metadata");
        assert_eq!(meta.file_id.as_deref(), Some("file_123"));
        assert_eq!(meta.container_id.as_deref(), Some("container_456"));
        assert_eq!(meta.index, Some(2));
    }

    #[test]
    fn openrouter_content_part_metadata_parses_item_id_and_reasoning_payload() {
        let part = crate::types::ContentPart::Reasoning {
            text: "thinking".to_string(),
            provider_metadata: Some(HashMap::from([(
                "openrouter".to_string(),
                serde_json::json!({
                    "itemId": "or_1",
                    "reasoningEncryptedContent": "enc_456"
                }),
            )])),
        };

        let meta = part
            .openrouter_metadata()
            .expect("openrouter content part metadata");
        assert_eq!(meta.item_id.as_deref(), Some("or_1"));
        assert_eq!(meta.reasoning_encrypted_content.as_deref(), Some("enc_456"));
    }

    #[test]
    fn perplexity_metadata_parses_usage_images_and_extra_fields() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert(
            "images".to_string(),
            serde_json::json!([
                {
                    "image_url": "https://images.example.com/rust.png",
                    "origin_url": "https://example.com/rust",
                    "height": 900,
                    "width": 1600
                }
            ]),
        );
        inner.insert(
            "usage".to_string(),
            serde_json::json!({
                "citation_tokens": 12,
                "num_search_queries": 3,
                "reasoning_tokens": 4
            }),
        );
        inner.insert(
            "citations".to_string(),
            serde_json::json!(["https://example.com/rust"]),
        );

        let mut outer = HashMap::new();
        outer.insert("perplexity".to_string(), inner);
        resp.provider_metadata = Some(outer);

        let meta = resp.perplexity_metadata().expect("perplexity metadata");
        assert_eq!(
            meta.citations.as_ref(),
            Some(&vec!["https://example.com/rust".to_string()])
        );
        assert_eq!(meta.images.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.citation_tokens),
            Some(12)
        );
        assert_eq!(
            meta.usage
                .as_ref()
                .and_then(|usage| usage.num_search_queries),
            Some(3)
        );
        assert_eq!(
            meta.usage.as_ref().and_then(|usage| usage.reasoning_tokens),
            Some(4)
        );
        assert_eq!(meta.extra.get("citations"), None);
    }
}
