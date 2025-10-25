//! Common Embedding Types

use serde::{Deserialize, Serialize};

/// Supported embedding formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingFormat {
    /// Standard float32 vectors
    Float,
    /// Base64 encoded vectors (if supported)
    Base64,
}

/// Token usage information for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// Number of input tokens processed
    pub prompt_tokens: u32,
    /// Total tokens (usually same as prompt_tokens for embeddings)
    pub total_tokens: u32,
}

impl EmbeddingUsage {
    /// Create new usage information
    pub fn new(prompt_tokens: u32, total_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            total_tokens,
        }
    }
}

/// Embedding task type for optimization (provider-specific)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EmbeddingTaskType {
    /// Retrieval query
    RetrievalQuery,
    /// Retrieval document
    RetrievalDocument,
    /// Semantic similarity
    SemanticSimilarity,
    /// Classification
    Classification,
    /// Clustering
    Clustering,
    /// Question answering
    QuestionAnswering,
    /// Fact verification
    FactVerification,
    /// Unspecified task
    Unspecified,
}

/// Embedding model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    /// Model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Embedding dimension
    pub dimension: usize,
    /// Maximum input tokens
    pub max_input_tokens: usize,
    /// Supported task types
    pub supported_tasks: Vec<EmbeddingTaskType>,
    /// Whether the model supports custom dimensions
    pub supports_custom_dimensions: bool,
}

impl EmbeddingModelInfo {
    /// Create new model info
    pub fn new(id: String, name: String, dimension: usize, max_input_tokens: usize) -> Self {
        Self {
            id,
            name,
            dimension,
            max_input_tokens,
            supported_tasks: vec![EmbeddingTaskType::Unspecified],
            supports_custom_dimensions: false,
        }
    }

    /// Add supported task type
    pub fn with_task(mut self, task: EmbeddingTaskType) -> Self {
        self.supported_tasks.push(task);
        self
    }

    /// Enable custom dimensions support
    pub fn with_custom_dimensions(mut self) -> Self {
        self.supports_custom_dimensions = true;
        self
    }
}

