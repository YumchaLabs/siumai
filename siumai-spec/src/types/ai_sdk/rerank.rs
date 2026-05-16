use crate::types::Warning;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{JSONValue, ProviderMetadata, ProviderOptions};
use crate::types::HttpResponseInfo;

/// AI SDK-style response data returned by reranking helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankResponseMetadata {
    /// Response id when the provider sends one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Timestamp for the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    pub model_id: String,
    /// Response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Raw response body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl RerankResponseMetadata {
    /// Create rerank response metadata.
    pub fn new(timestamp: DateTime<Utc>, model_id: impl Into<String>) -> Self {
        Self {
            id: None,
            timestamp,
            model_id: model_id.into(),
            headers: None,
            body: None,
        }
    }

    /// Attach a response id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
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

impl From<HttpResponseInfo> for RerankResponseMetadata {
    fn from(value: HttpResponseInfo) -> Self {
        Self {
            id: None,
            timestamp: value.timestamp,
            model_id: value.model_id.unwrap_or_default(),
            headers: (!value.headers.is_empty()).then_some(value.headers),
            body: value.body,
        }
    }
}

impl From<&HttpResponseInfo> for RerankResponseMetadata {
    fn from(value: &HttpResponseInfo) -> Self {
        Self {
            id: None,
            timestamp: value.timestamp,
            model_id: value.model_id.clone().unwrap_or_default(),
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
            body: value.body.clone(),
        }
    }
}

/// Single ranking entry in an AI SDK-style rerank result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankRanking<VALUE = JSONValue> {
    /// Original input index.
    pub original_index: u32,
    /// Relevance score.
    pub score: f64,
    /// Reranked document value.
    pub document: VALUE,
}

impl<VALUE> RerankRanking<VALUE> {
    /// Create a rerank ranking entry.
    pub fn new(original_index: u32, score: f64, document: VALUE) -> Self {
        Self {
            original_index,
            score,
            document,
        }
    }
}

/// Passive AI SDK-style result envelope for a `rerank` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankResult<VALUE = JSONValue> {
    /// Original documents that were reranked.
    pub original_documents: Vec<VALUE>,
    /// Reranked documents sorted by descending relevance.
    pub reranked_documents: Vec<VALUE>,
    /// Ranking entries with original indices, scores, and documents.
    pub ranking: Vec<RerankRanking<VALUE>>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Response metadata.
    pub response: RerankResponseMetadata,
}

impl<VALUE: Clone> RerankResult<VALUE> {
    /// Create a rerank result from original documents, ranking entries, and response metadata.
    pub fn new(
        original_documents: Vec<VALUE>,
        ranking: Vec<RerankRanking<VALUE>>,
        response: RerankResponseMetadata,
    ) -> Self {
        let reranked_documents = ranking.iter().map(|entry| entry.document.clone()).collect();

        Self {
            original_documents,
            reranked_documents,
            ranking,
            provider_metadata: None,
            response,
        }
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

/// Event payload for AI SDK rerank `onStart` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankStartEvent {
    /// Unique call id.
    pub call_id: String,
    /// Operation id, normally `ai.rerank`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Documents being reranked.
    pub documents: Vec<JSONValue>,
    /// Query used for reranking.
    pub query: String,
    /// Number of top documents to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Additional HTTP headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Provider-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// Event payload for AI SDK rerank `onFinish` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankEndEvent {
    /// Unique call id.
    pub call_id: String,
    /// Operation id, normally `ai.rerank`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Documents that were reranked.
    pub documents: Vec<JSONValue>,
    /// Query used for reranking.
    pub query: String,
    /// Ranking entries.
    pub ranking: Vec<RerankRanking<JSONValue>>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Response metadata.
    pub response: RerankResponseMetadata,
}

/// Event payload for the start of an underlying reranking-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankingModelCallStartEvent {
    /// Unique outer rerank call id.
    pub call_id: String,
    /// Operation id, normally `ai.rerank.doRerank`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Documents being reranked.
    pub documents: Vec<JSONValue>,
    /// Document family, usually `text` or `object`.
    pub documents_type: String,
    /// Query used for reranking.
    pub query: String,
    /// Number of top documents to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
}

/// Ranking summary returned by an underlying reranking-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankingModelCallRanking {
    /// Original document index.
    pub index: u32,
    /// Provider relevance score.
    pub relevance_score: f64,
}

impl RerankingModelCallRanking {
    /// Create a reranking model-call ranking entry.
    pub fn new(index: u32, relevance_score: f64) -> Self {
        Self {
            index,
            relevance_score,
        }
    }
}

/// Event payload for the end of an underlying reranking-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankingModelCallEndEvent {
    /// Unique outer rerank call id.
    pub call_id: String,
    /// Operation id, normally `ai.rerank.doRerank`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Document family, usually `text` or `object`.
    pub documents_type: String,
    /// Ranking summaries from the model call.
    pub ranking: Vec<RerankingModelCallRanking>,
}
