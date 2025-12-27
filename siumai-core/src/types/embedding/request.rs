//! Embedding Request Types

// no extra imports

use super::common::{EmbeddingFormat, EmbeddingTaskType};
use crate::types::HttpConfig;

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
    /// Optional embedding task type (typed, provider-agnostic)
    pub task_type: Option<EmbeddingTaskType>,
    /// Optional title/context hint (used by some providers, e.g., Gemini)
    pub title: Option<String>,
    /// Provider-specific typed options (v0.12+).
    ///
    /// This mirrors `ChatRequest::provider_options` and allows strongly-typed
    /// provider features to be used with embeddings. Providers can inject these
    /// into outbound JSON via `ProviderSpec::embedding_before_send()`.
    #[allow(clippy::derivable_impls)]
    pub provider_options: crate::types::ProviderOptions,
    /// Per-request HTTP configuration (headers, timeout, etc.)
    pub http_config: Option<HttpConfig>,
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

    /// Set provider-specific typed options (v0.12+)
    ///
    /// Prefer this over `provider_params` for type safety.
    pub fn with_provider_options(mut self, options: crate::types::ProviderOptions) -> Self {
        self.provider_options = options;
        self
    }

    /// Set task type for optimization (provider-specific)
    pub fn with_task_type(mut self, task_type: EmbeddingTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    /// Set a title/context hint (used by some providers)
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set per-request HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }

    /// Add a custom header for this request
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut config = self.http_config.take().unwrap_or_default();
        config.headers.insert(key.into(), value.into());
        self.http_config = Some(config);
        self
    }
}

/// Batch embedding request for processing multiple requests
#[derive(Debug, Clone)]
pub struct BatchEmbeddingRequest {
    /// Multiple embedding requests
    pub requests: Vec<EmbeddingRequest>,
    /// Batch processing options
    pub batch_options: BatchOptions,
}

impl BatchEmbeddingRequest {
    /// Create a new batch request
    pub fn new(requests: Vec<EmbeddingRequest>) -> Self {
        Self {
            requests,
            batch_options: BatchOptions::default(),
        }
    }

    /// Set batch options
    pub fn with_options(mut self, options: BatchOptions) -> Self {
        self.batch_options = options;
        self
    }

    /// Set maximum concurrency
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.batch_options.max_concurrency = Some(max_concurrency);
        self
    }
}

/// Batch processing options
#[derive(Debug, Clone, Default)]
pub struct BatchOptions {
    /// Maximum concurrent requests
    pub max_concurrency: Option<usize>,
    /// Timeout for each request
    pub request_timeout: Option<std::time::Duration>,
    /// Whether to fail fast on first error
    pub fail_fast: bool,
}
