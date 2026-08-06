//! Typed views over provider-owned Groq response metadata.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{LanguageResponse, ResponseMetadata, TranscriptionResponse};

/// Groq-specific metadata retained from a language response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GroqLanguageMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executed_tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_groq: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Ergonomic typed access to metadata on a Groq language response.
pub trait GroqLanguageResponseExt {
    fn groq_metadata(&self) -> Option<GroqLanguageMetadata>;
    fn groq_response_metadata(&self) -> ResponseMetadata;
}

impl GroqLanguageResponseExt for LanguageResponse {
    fn groq_metadata(&self) -> Option<GroqLanguageMetadata> {
        let mut metadata = self
            .provider_metadata()
            .get("groq")
            .and_then(Value::as_object)
            .cloned()?;
        if !metadata.contains_key("id")
            && let Some(id) = self.id()
        {
            metadata.insert("id".to_string(), Value::String(id.to_string()));
        }
        if !metadata.contains_key("modelId")
            && let Some(model) = self.model()
        {
            metadata.insert("modelId".to_string(), Value::String(model.to_string()));
        }
        serde_json::from_value(Value::Object(metadata)).ok()
    }

    fn groq_response_metadata(&self) -> ResponseMetadata {
        let request_id = self
            .groq_metadata()
            .and_then(|metadata| metadata.request_id);
        ResponseMetadata {
            response_id: self.id().map(str::to_owned),
            request_id,
            model: self.model().cloned(),
        }
    }
}

/// Groq-specific metadata retained from an audio transcription response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GroqTranscriptionMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_groq: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Ergonomic typed access to metadata on a Groq transcription response.
pub trait GroqTranscriptionResponseExt {
    fn groq_metadata(&self) -> Option<GroqTranscriptionMetadata>;
}

impl GroqTranscriptionResponseExt for TranscriptionResponse {
    fn groq_metadata(&self) -> Option<GroqTranscriptionMetadata> {
        self.provider
            .get("groq")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
    }
}

#[cfg(test)]
mod tests {
    use siumai_core::{FinishReason, LanguageResponseStatus, ModelId, Usage};

    use super::*;

    #[test]
    fn language_metadata_falls_back_to_canonical_identity() {
        let response = LanguageResponse::new(
            LanguageResponseStatus::Completed,
            Vec::new(),
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap()
        .with_id("chatcmpl-1")
        .with_model(ModelId::new("future-model").unwrap())
        .with_provider_metadata(BTreeMap::from([(
            "groq".to_string(),
            serde_json::json!({"requestId":"request-1","serviceTier":"flex"}),
        )]));

        let metadata = response.groq_metadata().unwrap();
        assert_eq!(metadata.id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(metadata.request_id.as_deref(), Some("request-1"));
        assert_eq!(metadata.model_id.as_deref(), Some("future-model"));
        assert_eq!(metadata.service_tier.as_deref(), Some("flex"));
        assert_eq!(
            response.groq_response_metadata().request_id.as_deref(),
            Some("request-1")
        );
    }
}
