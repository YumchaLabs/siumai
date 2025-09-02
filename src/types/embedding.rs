//! Embedding Types and Structures
//!
//! This module defines all types related to text embedding functionality,
//! including requests, responses, and configuration options.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Embedding request configuration
#[derive(Debug, Clone, Default)]
pub struct EmbeddingRequest {
    /// Input texts to embed
    pub input: Vec<String>,
    /// Model to use for embeddings
    pub model: Option<String>,
    /// Custom dimensions (if supported by provider)
    pub dimensions: Option<u32>,
    /// Encoding format preference
    pub encoding_format: Option<EmbeddingFormat>,
    /// User identifier for tracking
    pub user: Option<String>,
    /// Provider-specific parameters
    pub provider_params: HashMap<String, serde_json::Value>,
}

impl EmbeddingRequest {
    /// Create a new embedding request with input texts
    pub fn new(input: Vec<String>) -> Self {
        Self {
            input,
            ..Default::default()
        }
    }

    /// Create an embedding request for a single text
    pub fn single(text: impl Into<String>) -> Self {
        Self::new(vec![text.into()])
    }

    /// Create an embedding request optimized for retrieval queries
    pub fn query(text: impl Into<String>) -> Self {
        Self::single(text).with_task_type(EmbeddingTaskType::RetrievalQuery)
    }

    /// Create an embedding request optimized for retrieval documents
    pub fn document(text: impl Into<String>) -> Self {
        Self::single(text).with_task_type(EmbeddingTaskType::RetrievalDocument)
    }

    /// Create an embedding request for semantic similarity
    pub fn similarity(text: impl Into<String>) -> Self {
        Self::single(text).with_task_type(EmbeddingTaskType::SemanticSimilarity)
    }

    /// Create an embedding request for classification
    pub fn classification(text: impl Into<String>) -> Self {
        Self::single(text).with_task_type(EmbeddingTaskType::Classification)
    }

    /// Create an embedding request for clustering
    pub fn clustering(text: impl Into<String>) -> Self {
        Self::single(text).with_task_type(EmbeddingTaskType::Clustering)
    }

    /// Create an embedding request for question answering
    pub fn question_answering(text: impl Into<String>) -> Self {
        Self::single(text).with_task_type(EmbeddingTaskType::QuestionAnswering)
    }

    /// Create an embedding request for fact verification
    pub fn fact_verification(text: impl Into<String>) -> Self {
        Self::single(text).with_task_type(EmbeddingTaskType::FactVerification)
    }

    /// Set the model to use
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set custom dimensions
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set encoding format
    pub fn with_encoding_format(mut self, format: EmbeddingFormat) -> Self {
        self.encoding_format = Some(format);
        self
    }

    /// Set user identifier
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Add provider-specific parameter
    pub fn with_provider_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.provider_params.insert(key.into(), value);
        self
    }

    /// Set task type for optimization (provider-specific)
    pub fn with_task_type(mut self, task_type: EmbeddingTaskType) -> Self {
        let task_str = match task_type {
            EmbeddingTaskType::RetrievalQuery => "RETRIEVAL_QUERY",
            EmbeddingTaskType::RetrievalDocument => "RETRIEVAL_DOCUMENT",
            EmbeddingTaskType::SemanticSimilarity => "SEMANTIC_SIMILARITY",
            EmbeddingTaskType::Classification => "CLASSIFICATION",
            EmbeddingTaskType::Clustering => "CLUSTERING",
            EmbeddingTaskType::QuestionAnswering => "QUESTION_ANSWERING",
            EmbeddingTaskType::FactVerification => "FACT_VERIFICATION",
            EmbeddingTaskType::Unspecified => "TASK_TYPE_UNSPECIFIED",
        };
        self.provider_params.insert(
            "task_type".to_string(),
            serde_json::Value::String(task_str.to_string()),
        );
        self
    }

    /// Set provider-specific parameters from a map
    pub fn with_provider_params(mut self, params: HashMap<String, serde_json::Value>) -> Self {
        self.provider_params.extend(params);
        self
    }
}

/// Supported embedding formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingFormat {
    /// Standard float32 vectors
    Float,
    /// Base64 encoded vectors (if supported)
    Base64,
}

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

/// Batch embedding request for processing multiple sets of texts
#[derive(Debug, Clone)]
pub struct BatchEmbeddingRequest {
    /// Multiple embedding requests
    pub requests: Vec<EmbeddingRequest>,
    /// Batch processing options
    pub batch_options: BatchOptions,
}

/// Options for batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchOptions {
    /// Maximum concurrent requests
    pub max_concurrency: Option<usize>,
    /// Timeout for each request
    pub request_timeout: Option<std::time::Duration>,
    /// Whether to fail fast on first error
    pub fail_fast: bool,
}

/// Batch embedding response
#[derive(Debug, Clone)]
pub struct BatchEmbeddingResponse {
    /// Individual responses (same order as requests)
    pub responses: Vec<Result<EmbeddingResponse, String>>,
    /// Overall batch metadata
    pub metadata: HashMap<String, serde_json::Value>,
}
