use crate::types::Warning;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{Embedding, EmbeddingModelUsage, JSONValue, ProviderMetadata, ProviderOptions};

/// Optional raw response data returned by embedding and reranking helpers.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelCallResponseData {
    /// Response headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Raw response body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl ModelCallResponseData {
    /// Create an empty response data envelope.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach response headers.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }

    /// Attach a raw response body.
    pub fn with_body(mut self, body: JSONValue) -> Self {
        self.body = Some(body);
        self
    }
}

/// Value input accepted by AI SDK embed callback payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbedValue {
    /// Single embedded value.
    Single(String),
    /// Multiple embedded values.
    Many(Vec<String>),
}

impl From<String> for EmbedValue {
    fn from(value: String) -> Self {
        Self::Single(value)
    }
}

impl From<&str> for EmbedValue {
    fn from(value: &str) -> Self {
        Self::Single(value.to_string())
    }
}

impl From<Vec<String>> for EmbedValue {
    fn from(value: Vec<String>) -> Self {
        Self::Many(value)
    }
}

/// Embedding output accepted by AI SDK embed callback payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbedOutput {
    /// Single embedding vector.
    Single(Embedding),
    /// Multiple embedding vectors.
    Many(Vec<Embedding>),
}

impl From<Embedding> for EmbedOutput {
    fn from(value: Embedding) -> Self {
        Self::Single(value)
    }
}

impl From<Vec<Embedding>> for EmbedOutput {
    fn from(value: Vec<Embedding>) -> Self {
        Self::Many(value)
    }
}

/// Response data accepted by AI SDK embed callback payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbedResponseData {
    /// Single model-call response envelope.
    Single(ModelCallResponseData),
    /// Multiple model-call response envelopes.
    Many(Vec<Option<ModelCallResponseData>>),
}

impl From<ModelCallResponseData> for EmbedResponseData {
    fn from(value: ModelCallResponseData) -> Self {
        Self::Single(value)
    }
}

impl From<Vec<Option<ModelCallResponseData>>> for EmbedResponseData {
    fn from(value: Vec<Option<ModelCallResponseData>>) -> Self {
        Self::Many(value)
    }
}

/// Passive AI SDK-style result envelope for an `embed` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbedResult {
    /// Value that was embedded.
    pub value: String,
    /// Embedding vector for the value.
    pub embedding: Embedding,
    /// Embedding token usage.
    pub usage: EmbeddingModelUsage,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Optional raw response data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<ModelCallResponseData>,
}

impl EmbedResult {
    /// Create an embedding result.
    pub fn new(value: impl Into<String>, embedding: Embedding, usage: EmbeddingModelUsage) -> Self {
        Self {
            value: value.into(),
            embedding,
            usage,
            warnings: Vec::new(),
            provider_metadata: None,
            response: None,
        }
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Attach raw response data.
    pub fn with_response(mut self, response: ModelCallResponseData) -> Self {
        self.response = Some(response);
        self
    }
}

/// Passive AI SDK-style result envelope for an `embedMany` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbedManyResult {
    /// Values that were embedded.
    pub values: Vec<String>,
    /// Embeddings in the same order as `values`.
    pub embeddings: Vec<Embedding>,
    /// Embedding token usage.
    pub usage: EmbeddingModelUsage,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Optional raw response data for each underlying model call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responses: Option<Vec<Option<ModelCallResponseData>>>,
}

impl EmbedManyResult {
    /// Create an embed-many result.
    pub fn new(
        values: Vec<String>,
        embeddings: Vec<Embedding>,
        usage: EmbeddingModelUsage,
    ) -> Self {
        Self {
            values,
            embeddings,
            usage,
            warnings: Vec::new(),
            provider_metadata: None,
            responses: None,
        }
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Attach raw response data for underlying model calls.
    pub fn with_responses(mut self, responses: Vec<Option<ModelCallResponseData>>) -> Self {
        self.responses = Some(responses);
        self
    }
}

/// Event payload for AI SDK embed/embedMany `onStart` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbedStartEvent {
    /// Unique call id.
    pub call_id: String,
    /// Operation id, e.g. `ai.embed` or `ai.embedMany`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Value or values being embedded.
    pub value: EmbedValue,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Additional HTTP headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Provider-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// Event payload for AI SDK embed/embedMany `onFinish` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbedEndEvent {
    /// Unique call id.
    pub call_id: String,
    /// Operation id, e.g. `ai.embed` or `ai.embedMany`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Value or values that were embedded.
    pub value: EmbedValue,
    /// Resulting embedding or embeddings.
    pub embedding: EmbedOutput,
    /// Embedding token usage.
    pub usage: EmbeddingModelUsage,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Optional raw response data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<EmbedResponseData>,
}

/// Event payload for the start of an underlying embedding-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingModelCallStartEvent {
    /// Unique outer embed call id.
    pub call_id: String,
    /// Unique id for this underlying model invocation.
    pub embed_call_id: String,
    /// Operation id, e.g. `ai.embed.doEmbed`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Values being embedded in this model call.
    pub values: Vec<String>,
}

/// Event payload for the end of an underlying embedding-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingModelCallEndEvent {
    /// Unique outer embed call id.
    pub call_id: String,
    /// Unique id for this underlying model invocation.
    pub embed_call_id: String,
    /// Operation id, e.g. `ai.embed.doEmbed`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Values embedded in this model call.
    pub values: Vec<String>,
    /// Resulting embeddings from this model call.
    pub embeddings: Vec<Embedding>,
    /// Token usage for this model call.
    pub usage: EmbeddingModelUsage,
}
