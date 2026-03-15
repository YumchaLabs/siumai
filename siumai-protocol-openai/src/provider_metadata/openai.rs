//! OpenAI-specific response metadata

use serde::{Deserialize, Serialize};

/// A normalized "source" entry (Vercel-aligned), typically produced from web search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OpenAiSource {
    /// Source identifier (stable within a response).
    pub id: String,

    /// Source type (currently only "url" for web search results).
    pub source_type: String,

    /// Source URL.
    pub url: String,

    /// Optional title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Tool call id that produced this source (when applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Media type for document sources (e.g. "text/plain").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,

    /// Filename for document sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,

    /// Provider-specific metadata for the source (e.g. fileId/containerId/index).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,

    /// Provider-native snippet/summary if present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

/// OpenAI-specific source metadata carried on `OpenAiSource.provider_metadata`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiSourceMetadata {
    /// Provider file id for file/document-backed citations.
    #[serde(skip_serializing_if = "Option::is_none", rename = "fileId")]
    pub file_id: Option<String>,

    /// Container id for container file citations.
    #[serde(skip_serializing_if = "Option::is_none", rename = "containerId")]
    pub container_id: Option<String>,

    /// Provider-native index when OpenAI surfaces one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

/// OpenAI-specific metadata from chat responses
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiMetadata {
    /// Number of tokens used for reasoning (o1/o3 models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,

    /// System fingerprint for this response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,

    /// Service tier used for this request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Revised prompt (for image generation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,

    /// Sources extracted from provider-hosted tool results (Vercel-aligned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<OpenAiSource>>,

    /// Logprobs extracted from Chat Completions / Responses outputs (Vercel-aligned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// OpenAI-specific metadata attached to unified content parts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiContentPartMetadata {
    /// Stable OpenAI item id surfaced on text/reasoning/tool parts.
    #[serde(skip_serializing_if = "Option::is_none", rename = "itemId")]
    pub item_id: Option<String>,

    /// Encrypted reasoning payload surfaced on reasoning parts when available.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "reasoningEncryptedContent"
    )]
    pub reasoning_encrypted_content: Option<String>,
}

impl crate::types::provider_metadata::FromMetadata for OpenAiMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

/// Typed helper for OpenAI metadata extraction from `ChatResponse`.
pub trait OpenAiChatResponseExt {
    fn openai_metadata(&self) -> Option<OpenAiMetadata>;
}

impl OpenAiChatResponseExt for crate::types::ChatResponse {
    fn openai_metadata(&self) -> Option<OpenAiMetadata> {
        use crate::types::provider_metadata::FromMetadata;
        let meta = self.provider_metadata.as_ref()?.get("openai")?;
        OpenAiMetadata::from_metadata(meta)
    }
}

/// Typed helper for OpenAI metadata extraction from `OpenAiSource`.
pub trait OpenAiSourceExt {
    fn openai_metadata(&self) -> Option<OpenAiSourceMetadata>;
}

impl OpenAiSourceExt for OpenAiSource {
    fn openai_metadata(&self) -> Option<OpenAiSourceMetadata> {
        let metadata = self.provider_metadata.clone()?;
        let inner = metadata.get("openai").cloned().unwrap_or(metadata);
        serde_json::from_value(inner).ok()
    }
}

/// Typed helper for OpenAI metadata extraction from `ContentPart`.
pub trait OpenAiContentPartExt {
    fn openai_metadata(&self) -> Option<OpenAiContentPartMetadata>;
}

impl OpenAiContentPartExt for crate::types::ContentPart {
    fn openai_metadata(&self) -> Option<OpenAiContentPartMetadata> {
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

        let meta = provider_metadata.get("openai")?;
        serde_json::from_value(meta.clone()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn openai_metadata_parses_logprobs() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert(
            "logprobs".to_string(),
            serde_json::json!([[{ "token": "N", "logprob": -0.1 }]]),
        );

        let mut outer = HashMap::new();
        outer.insert("openai".to_string(), inner);
        resp.provider_metadata = Some(outer);

        let meta = resp.openai_metadata().expect("openai metadata");
        assert!(meta.logprobs.is_some());
    }

    #[test]
    fn openai_source_metadata_parses_file_and_container_fields() {
        let source = OpenAiSource {
            id: "ann:0".to_string(),
            source_type: "document".to_string(),
            url: "file_123".to_string(),
            title: Some("Doc".to_string()),
            tool_call_id: None,
            media_type: Some("text/plain".to_string()),
            filename: Some("notes.txt".to_string()),
            provider_metadata: Some(serde_json::json!({
                "fileId": "file_123",
                "containerId": "container_456",
                "index": 7
            })),
            snippet: None,
        };

        let meta = source.openai_metadata().expect("openai source metadata");
        assert_eq!(meta.file_id.as_deref(), Some("file_123"));
        assert_eq!(meta.container_id.as_deref(), Some("container_456"));
        assert_eq!(meta.index, Some(7));
    }

    #[test]
    fn openai_content_part_metadata_parses_item_id_and_reasoning_payload() {
        let part = crate::types::ContentPart::Reasoning {
            text: "thinking".to_string(),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({
                    "itemId": "rs_1",
                    "reasoningEncryptedContent": "enc_123"
                }),
            )])),
        };

        let meta = crate::provider_metadata::openai::OpenAiContentPartExt::openai_metadata(&part)
            .expect("openai content part metadata");
        assert_eq!(meta.item_id.as_deref(), Some("rs_1"));
        assert_eq!(meta.reasoning_encrypted_content.as_deref(), Some("enc_123"));
    }
}
