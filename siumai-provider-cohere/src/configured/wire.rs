use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::provider_options::{CohereEmbeddingInputType, CohereEmbeddingTruncate};

#[derive(Serialize)]
pub(crate) struct EmbeddingWireRequest<'a> {
    pub(crate) model: &'a str,
    pub(crate) embedding_types: [&'static str; 1],
    pub(crate) texts: &'a [String],
    pub(crate) input_type: CohereEmbeddingInputType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) truncate: Option<CohereEmbeddingTruncate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_dimension: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingWireResponse {
    #[serde(default)]
    pub(crate) id: Option<String>,
    pub(crate) embeddings: EmbeddingVectors,
    #[serde(default)]
    pub(crate) meta: Value,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingVectors {
    pub(crate) float: Vec<Vec<f32>>,
}

#[derive(Serialize)]
pub(crate) struct RerankWireRequest<'a> {
    pub(crate) model: &'a str,
    pub(crate) query: &'a str,
    pub(crate) documents: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens_per_doc: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) priority: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RerankWireResponse {
    #[serde(default)]
    pub(crate) id: Option<String>,
    pub(crate) results: Vec<RerankWireResult>,
    #[serde(default)]
    pub(crate) meta: Value,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RerankWireResult {
    pub(crate) index: usize,
    pub(crate) relevance_score: f64,
}
