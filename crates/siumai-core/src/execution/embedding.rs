//! Minimal embedding transformers for standards

use crate::error::LlmError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Provider-agnostic input for embedding transformers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingInput {
    pub input: Vec<String>,
    pub model: Option<String>,
    pub dimensions: Option<u32>,
    /// "float" or "base64" if provider supports it
    pub encoding_format: Option<String>,
    pub user: Option<String>,
    pub title: Option<String>,
}

/// Token usage information for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Provider-agnostic embedding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub usage: Option<EmbeddingUsage>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Request transformer focused on embedding operations
pub trait EmbeddingRequestTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform_embedding(&self, req: &EmbeddingInput) -> Result<serde_json::Value, LlmError>;
}

/// Response transformer focused on embedding operations
pub trait EmbeddingResponseTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResult, LlmError>;
}
