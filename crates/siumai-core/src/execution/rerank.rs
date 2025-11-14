//! Minimal rerank transformers for standards

use crate::error::LlmError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Provider-agnostic rerank input
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RerankInput {
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<u32>,
    pub return_documents: Option<bool>,
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Provider-agnostic rerank result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankItem {
    pub index: u32,
    pub relevance_score: f64,
    pub document: Option<String>,
}

/// Provider-agnostic rerank output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankOutput {
    pub id: Option<String>,
    pub results: Vec<RerankItem>,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub trait RerankRequestTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform(&self, req: &RerankInput) -> Result<serde_json::Value, LlmError>;
}

pub trait RerankResponseTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform_response(&self, raw: &serde_json::Value) -> Result<RerankOutput, LlmError>;
}
