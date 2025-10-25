//! Embedding Response Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::common::EmbeddingUsage;

/// Embedding response containing vectors and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Embedding vectors (one per input text)
    pub embeddings: Vec<Vec<f32>>,
    /// Model that generated the embeddings
    pub model: String,
    /// Token usage information
    pub usage: Option<EmbeddingUsage>,
    /// Provider-specific metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl EmbeddingResponse {
    /// Create a new embedding response
    pub fn new(embeddings: Vec<Vec<f32>>, model: String) -> Self {
        Self {
            embeddings,
            model,
            usage: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the number of embeddings
    pub fn count(&self) -> usize {
        self.embeddings.len()
    }

    /// Get the dimension of embeddings (assumes all have same dimension)
    pub fn dimension(&self) -> Option<usize> {
        self.embeddings.first().map(|e| e.len())
    }

    /// Check if response is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get embedding at index
    pub fn get(&self, index: usize) -> Option<&Vec<f32>> {
        self.embeddings.get(index)
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set usage information
    pub fn with_usage(mut self, usage: EmbeddingUsage) -> Self {
        self.usage = Some(usage);
        self
    }
}

/// Batch embedding response
#[derive(Debug, Clone)]
pub struct BatchEmbeddingResponse {
    /// Individual responses (same order as requests)
    pub responses: Vec<Result<EmbeddingResponse, String>>,
    /// Overall batch metadata
    pub metadata: HashMap<String, serde_json::Value>,
}
