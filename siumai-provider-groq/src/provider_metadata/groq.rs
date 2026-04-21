//! `Groq`-specific response metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A normalized source entry reused from the OpenAI-compatible family.
pub use siumai_protocol_openai::provider_metadata::openai::OpenAiSource as GroqSource;
/// Groq-specific typed source metadata carried on `GroqSource.provider_metadata`.
pub use siumai_protocol_openai::provider_metadata::openai::OpenAiSourceMetadata as GroqSourceMetadata;

/// Groq-specific metadata from chat responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GroqMetadata {
    /// Stable response id mirrored from the AI SDK response-metadata lane.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Stable model id mirrored from the AI SDK response-metadata lane.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "modelId",
        alias = "model"
    )]
    pub model_id: Option<String>,

    /// Stable response timestamp mirrored from the AI SDK response-metadata lane.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "timestamp",
        alias = "created"
    )]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,

    /// Sources extracted from provider-hosted tool results (Vercel-aligned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<GroqSource>>,

    /// Logprobs extracted from Chat Completions outputs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,

    /// Preserve unknown provider-specific metadata for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl crate::types::provider_metadata::FromMetadata for GroqMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

/// Typed helper for Groq metadata extraction from `ChatResponse`.
pub trait GroqChatResponseExt {
    fn groq_metadata(&self) -> Option<GroqMetadata>;
    fn groq_response_metadata(&self) -> Option<crate::types::ResponseMetadata>;
}

impl GroqChatResponseExt for crate::types::ChatResponse {
    fn groq_metadata(&self) -> Option<GroqMetadata> {
        let mut meta = self
            .provider_metadata
            .as_ref()
            .and_then(|metadata| {
                crate::types::provider_metadata::provider_metadata_object(metadata, "groq")
            })?
            .clone();

        if !meta.contains_key("id")
            && let Some(id) = self.id.clone()
        {
            meta.insert("id".to_string(), serde_json::Value::String(id));
        }

        if !meta.contains_key("modelId")
            && !meta.contains_key("model")
            && let Some(model_id) = self.model.clone()
        {
            meta.insert("modelId".to_string(), serde_json::Value::String(model_id));
        }

        serde_json::from_value(serde_json::Value::Object(meta)).ok()
    }

    fn groq_response_metadata(&self) -> Option<crate::types::ResponseMetadata> {
        let meta = self.groq_metadata()?;

        Some(crate::types::ResponseMetadata {
            id: self.id.clone().or(meta.id),
            model: self.model.clone().or(meta.model_id),
            created: meta.timestamp,
            provider: "groq".to_string(),
            request_id: None,
            headers: None,
        })
    }
}

/// Typed helper for Groq metadata extraction from `GroqSource`.
pub trait GroqSourceExt {
    fn groq_metadata(&self) -> Option<GroqSourceMetadata>;
}

impl GroqSourceExt for GroqSource {
    fn groq_metadata(&self) -> Option<GroqSourceMetadata> {
        let metadata = self.provider_metadata.clone()?;

        if let Some(inner) = metadata.get("groq").cloned() {
            return serde_json::from_value(inner).ok();
        }

        siumai_protocol_openai::provider_metadata::openai::OpenAiSourceExt::openai_metadata(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groq_metadata_parses_sources_and_logprobs() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));
        resp.id = Some("chatcmpl-groq-test".to_string());
        resp.model = Some("llama-3.3-70b-versatile".to_string());

        let mut inner = HashMap::new();
        inner.insert(
            "timestamp".to_string(),
            serde_json::json!("2025-03-08T00:00:00Z"),
        );
        inner.insert(
            "sources".to_string(),
            serde_json::json!([
                {
                    "id": "src_1",
                    "source_type": "url",
                    "url": "https://example.com",
                    "title": "Example"
                }
            ]),
        );
        inner.insert(
            "logprobs".to_string(),
            serde_json::json!([{ "token": "hello", "logprob": -0.1 }]),
        );
        inner.insert("vendor_extra".to_string(), serde_json::json!(true));

        let mut outer = HashMap::new();
        outer.insert(
            "groq".to_string(),
            serde_json::Value::Object(inner.into_iter().collect()),
        );
        resp.provider_metadata = Some(outer);

        let meta = resp.groq_metadata().expect("groq metadata");
        assert_eq!(meta.id.as_deref(), Some("chatcmpl-groq-test"));
        assert_eq!(meta.model_id.as_deref(), Some("llama-3.3-70b-versatile"));
        assert_eq!(
            meta.timestamp.map(|timestamp| timestamp.timestamp()),
            Some(1_741_392_000)
        );
        assert_eq!(meta.sources.as_ref().map(Vec::len), Some(1));
        assert!(meta.logprobs.is_some());
        assert_eq!(
            meta.extra.get("vendor_extra"),
            Some(&serde_json::json!(true))
        );

        let response_metadata = resp
            .groq_response_metadata()
            .expect("groq response metadata");
        assert_eq!(response_metadata.id.as_deref(), Some("chatcmpl-groq-test"));
        assert_eq!(
            response_metadata.model.as_deref(),
            Some("llama-3.3-70b-versatile")
        );
        assert_eq!(
            response_metadata
                .created
                .map(|timestamp| timestamp.timestamp()),
            Some(1_741_392_000)
        );
    }

    #[test]
    fn groq_source_metadata_parses_provider_owned_and_compatible_shapes() {
        let provider_owned = GroqSource {
            id: "src_doc_1".to_string(),
            source_type: "document".to_string(),
            url: "file_123".to_string(),
            title: Some("Doc".to_string()),
            tool_call_id: None,
            media_type: Some("text/plain".to_string()),
            filename: Some("notes.txt".to_string()),
            provider_metadata: Some(serde_json::json!({
                "groq": {
                    "fileId": "file_123",
                    "containerId": "container_456",
                    "index": 7
                }
            })),
            snippet: None,
        };

        let meta = provider_owned
            .groq_metadata()
            .expect("groq source metadata from provider-owned shape");
        assert_eq!(meta.file_id.as_deref(), Some("file_123"));
        assert_eq!(meta.container_id.as_deref(), Some("container_456"));
        assert_eq!(meta.index, Some(7));

        let compatible = GroqSource {
            id: "src_doc_2".to_string(),
            source_type: "document".to_string(),
            url: "file_789".to_string(),
            title: Some("Doc 2".to_string()),
            tool_call_id: None,
            media_type: Some("application/octet-stream".to_string()),
            filename: Some("artifact.bin".to_string()),
            provider_metadata: Some(serde_json::json!({
                "openai": {
                    "fileId": "file_789",
                    "index": 5
                }
            })),
            snippet: None,
        };

        let meta = compatible
            .groq_metadata()
            .expect("groq source metadata from compatible shape");
        assert_eq!(meta.file_id.as_deref(), Some("file_789"));
        assert!(meta.container_id.is_none());
        assert_eq!(meta.index, Some(5));
    }
}
