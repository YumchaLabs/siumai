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
    /// OpenAI annotation/source discriminator (`file_citation`, `container_file_citation`, `file_path`).
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub metadata_type: Option<String>,

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
    /// Stable OpenAI Responses response id surfaced via provider metadata.
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseId")]
    pub response_id: Option<String>,

    /// Number of tokens used for reasoning (o1/o3 models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,

    /// Number of prediction tokens accepted by the model.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "acceptedPredictionTokens"
    )]
    pub accepted_prediction_tokens: Option<u32>,

    /// Number of prediction tokens rejected by the model.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "rejectedPredictionTokens"
    )]
    pub rejected_prediction_tokens: Option<u32>,

    /// System fingerprint for this response
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "systemFingerprint",
        alias = "system_fingerprint"
    )]
    pub system_fingerprint: Option<String>,

    /// Service tier used for this request
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "serviceTier",
        alias = "service_tier"
    )]
    pub service_tier: Option<String>,

    /// Revised prompt (for image generation)
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "revisedPrompt",
        alias = "revised_prompt"
    )]
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
    /// OpenAI content/custom discriminator (`compaction`, etc.).
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub metadata_type: Option<String>,

    /// Stable OpenAI item id surfaced on text/reasoning/tool parts.
    #[serde(skip_serializing_if = "Option::is_none", rename = "itemId")]
    pub item_id: Option<String>,

    /// Message phase surfaced on Responses output text parts when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,

    /// Raw OpenAI output-text annotations preserved on text parts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<serde_json::Value>>,

    /// Encrypted reasoning payload surfaced on reasoning parts when available.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "reasoningEncryptedContent"
    )]
    pub reasoning_encrypted_content: Option<String>,

    /// Encrypted compaction payload surfaced on compaction custom parts when available.
    #[serde(skip_serializing_if = "Option::is_none", rename = "encryptedContent")]
    pub encrypted_content: Option<String>,
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
    fn openai_metadata_with_key(&self, key: &str) -> Option<OpenAiMetadata>;
}

impl OpenAiChatResponseExt for crate::types::ChatResponse {
    fn openai_metadata(&self) -> Option<OpenAiMetadata> {
        self.openai_metadata_with_key("openai")
    }

    fn openai_metadata_with_key(&self, key: &str) -> Option<OpenAiMetadata> {
        let mut meta = self
            .provider_metadata
            .as_ref()
            .and_then(|metadata| {
                crate::types::provider_metadata::provider_metadata_object(metadata, key)
            })?
            .clone();

        if !meta.contains_key("responseId")
            && let Some(id) = self.id.clone()
        {
            meta.insert("responseId".to_string(), serde_json::Value::String(id));
        }

        if !meta.contains_key("serviceTier")
            && !meta.contains_key("service_tier")
            && let Some(service_tier) = self.service_tier.clone()
        {
            meta.insert(
                "serviceTier".to_string(),
                serde_json::Value::String(service_tier),
            );
        }

        if !meta.contains_key("systemFingerprint")
            && !meta.contains_key("system_fingerprint")
            && let Some(system_fingerprint) = self.system_fingerprint.clone()
        {
            meta.insert(
                "systemFingerprint".to_string(),
                serde_json::Value::String(system_fingerprint),
            );
        }

        serde_json::from_value(serde_json::Value::Object(meta)).ok()
    }
}

/// Typed helper for OpenAI metadata extraction from `OpenAiSource`.
pub trait OpenAiSourceExt {
    fn openai_metadata(&self) -> Option<OpenAiSourceMetadata>;
    fn openai_metadata_with_key(&self, key: &str) -> Option<OpenAiSourceMetadata>;
}

impl OpenAiSourceExt for OpenAiSource {
    fn openai_metadata(&self) -> Option<OpenAiSourceMetadata> {
        self.openai_metadata_with_key("openai")
    }

    fn openai_metadata_with_key(&self, key: &str) -> Option<OpenAiSourceMetadata> {
        let metadata = self.provider_metadata.clone()?;
        let inner = metadata.get(key).cloned().unwrap_or(metadata);
        serde_json::from_value(inner).ok()
    }
}

/// Typed helper for OpenAI metadata extraction from `ContentPart`.
pub trait OpenAiContentPartExt {
    fn openai_metadata(&self) -> Option<OpenAiContentPartMetadata>;
    fn openai_metadata_with_key(&self, key: &str) -> Option<OpenAiContentPartMetadata>;
}

impl OpenAiContentPartExt for crate::types::ContentPart {
    fn openai_metadata(&self) -> Option<OpenAiContentPartMetadata> {
        self.openai_metadata_with_key("openai")
    }

    fn openai_metadata_with_key(&self, key: &str) -> Option<OpenAiContentPartMetadata> {
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
            | ContentPart::ReasoningFile {
                provider_metadata, ..
            }
            | ContentPart::Custom {
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

        let meta = provider_metadata.get(key)?;
        serde_json::from_value(meta.clone()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn production_source() -> &'static str {
        include_str!("openai.rs")
            .split_once("#[cfg(test)]")
            .expect("test marker should exist")
            .0
    }

    #[test]
    fn openai_provider_metadata_source_does_not_read_request_provider_options() {
        let source = production_source();

        for forbidden in [
            "providerOptions",
            "provider_options",
            "provider_options_map",
            "ProviderOptionsMap",
        ] {
            assert!(
                !source.contains(forbidden),
                "OpenAI typed provider metadata views must not read request-side {forbidden}"
            );
        }
    }

    #[test]
    fn openai_metadata_parses_logprobs_and_prediction_tokens() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));
        resp.id = Some("resp_123".to_string());
        resp.service_tier = Some("flex".to_string());
        resp.system_fingerprint = Some("fp_123".to_string());

        let mut inner = HashMap::new();
        inner.insert(
            "logprobs".to_string(),
            serde_json::json!([[{ "token": "N", "logprob": -0.1 }]]),
        );
        inner.insert("acceptedPredictionTokens".to_string(), serde_json::json!(5));
        inner.insert("rejectedPredictionTokens".to_string(), serde_json::json!(6));

        let mut outer = HashMap::new();
        outer.insert(
            "openai".to_string(),
            serde_json::Value::Object(inner.into_iter().collect()),
        );
        resp.provider_metadata = Some(outer);

        let meta = resp.openai_metadata().expect("openai metadata");
        assert_eq!(meta.response_id.as_deref(), Some("resp_123"));
        assert!(meta.logprobs.is_some());
        assert_eq!(meta.accepted_prediction_tokens, Some(5));
        assert_eq!(meta.rejected_prediction_tokens, Some(6));
        assert_eq!(meta.service_tier.as_deref(), Some("flex"));
        assert_eq!(meta.system_fingerprint.as_deref(), Some("fp_123"));
    }

    #[test]
    fn openai_metadata_reads_default_and_custom_keys() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut default_inner = HashMap::new();
        default_inner.insert("serviceTier".to_string(), serde_json::json!("default"));

        let mut custom_inner = HashMap::new();
        custom_inner.insert("serviceTier".to_string(), serde_json::json!("priority"));

        let mut outer = HashMap::new();
        outer.insert(
            "openai".to_string(),
            serde_json::Value::Object(default_inner.into_iter().collect()),
        );
        outer.insert(
            "azure".to_string(),
            serde_json::Value::Object(custom_inner.into_iter().collect()),
        );
        resp.provider_metadata = Some(outer);

        assert_eq!(
            resp.openai_metadata().and_then(|meta| meta.service_tier),
            Some("default".to_string())
        );
        assert_eq!(
            resp.openai_metadata_with_key("azure")
                .and_then(|meta| meta.service_tier),
            Some("priority".to_string())
        );
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
                "type": "container_file_citation",
                "fileId": "file_123",
                "containerId": "container_456",
                "index": 7
            })),
            snippet: None,
        };

        let meta = source.openai_metadata().expect("openai source metadata");
        assert_eq!(
            meta.metadata_type.as_deref(),
            Some("container_file_citation")
        );
        assert_eq!(meta.file_id.as_deref(), Some("file_123"));
        assert_eq!(meta.container_id.as_deref(), Some("container_456"));
        assert_eq!(meta.index, Some(7));
    }

    #[test]
    fn openai_content_part_metadata_parses_text_phase_annotations_and_reasoning_payloads() {
        let text_part = crate::types::ContentPart::Text {
            text: "hello".to_string(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({
                    "itemId": "msg_1",
                    "phase": "final_answer",
                    "annotations": [
                        {
                            "type": "file_citation",
                            "file_id": "file_123",
                            "index": 7
                        }
                    ]
                }),
            )])),
        };

        let text_meta =
            crate::provider_metadata::openai::OpenAiContentPartExt::openai_metadata(&text_part)
                .expect("openai text metadata");
        assert_eq!(text_meta.item_id.as_deref(), Some("msg_1"));
        assert_eq!(text_meta.phase.as_deref(), Some("final_answer"));
        assert_eq!(
            text_meta
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.first())
                .and_then(|annotation| annotation.get("type"))
                .and_then(|value| value.as_str()),
            Some("file_citation")
        );

        let reasoning_part = crate::types::ContentPart::Reasoning {
            text: "thinking".to_string(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({
                    "itemId": "rs_1",
                    "reasoningEncryptedContent": "enc_123"
                }),
            )])),
        };

        let reasoning_meta =
            crate::provider_metadata::openai::OpenAiContentPartExt::openai_metadata(
                &reasoning_part,
            )
            .expect("openai content part metadata");
        assert_eq!(reasoning_meta.item_id.as_deref(), Some("rs_1"));
        assert_eq!(
            reasoning_meta.reasoning_encrypted_content.as_deref(),
            Some("enc_123")
        );

        let custom_part = crate::types::ContentPart::Custom {
            kind: "openai.compaction".to_string(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({
                    "type": "compaction",
                    "itemId": "cmp_1",
                    "encryptedContent": "enc_compaction"
                }),
            )])),
        };

        let custom_meta =
            crate::provider_metadata::openai::OpenAiContentPartExt::openai_metadata(&custom_part)
                .expect("openai custom metadata");
        assert_eq!(custom_meta.metadata_type.as_deref(), Some("compaction"));
        assert_eq!(custom_meta.item_id.as_deref(), Some("cmp_1"));
        assert_eq!(
            custom_meta.encrypted_content.as_deref(),
            Some("enc_compaction")
        );
    }

    #[test]
    fn openai_content_part_metadata_reads_custom_key() {
        let part = crate::types::ContentPart::Reasoning {
            text: "thinking".to_string(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "azure".to_string(),
                serde_json::json!({
                    "itemId": "rs_1",
                    "reasoningEncryptedContent": "enc_123"
                }),
            )])),
        };

        assert!(part.openai_metadata().is_none());
        let meta = part
            .openai_metadata_with_key("azure")
            .expect("openai content-part metadata");
        assert_eq!(meta.item_id.as_deref(), Some("rs_1"));
        assert_eq!(meta.reasoning_encrypted_content.as_deref(), Some("enc_123"));
    }
}
