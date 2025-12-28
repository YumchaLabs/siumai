//! OpenAI-specific Data Types
//!
//! Contains data structures specific to the `OpenAI` API.

pub use crate::standards::openai::types::*;

/// OpenAI-specific parameters
#[derive(Debug, Clone, Default)]
pub struct OpenAiSpecificParams {
    /// Organization ID
    pub organization: Option<String>,
    /// Project ID
    pub project: Option<String>,
    /// Response format for structured output
    pub response_format: Option<serde_json::Value>,
    /// Logit bias
    pub logit_bias: Option<serde_json::Value>,
    /// Whether to return logprobs
    pub logprobs: Option<bool>,
    /// Number of logprobs to return
    pub top_logprobs: Option<u32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// User identifier
    pub user: Option<String>,
}

// ================================================================================================
// Embedding Types
// ================================================================================================

/// OpenAI-specific embedding configuration options
///
/// This struct provides type-safe configuration for OpenAI embedding requests,
/// including custom dimensions, encoding formats, and user tracking.
///
/// # Example
/// ```rust,no_run
/// use siumai::providers::openai::OpenAiEmbeddingOptions;
/// use siumai::types::EmbeddingFormat;
///
/// let options = OpenAiEmbeddingOptions::new()
///     .with_dimensions(512)
///     .with_encoding_format(EmbeddingFormat::Float)
///     .with_user("user123");
/// ```
#[derive(Debug, Clone, Default)]
pub struct OpenAiEmbeddingOptions {
    /// Custom dimensions (for text-embedding-3 models)
    /// Must be supported by the specific model being used
    pub dimensions: Option<u32>,
    /// Encoding format preference (float or base64)
    pub encoding_format: Option<crate::types::EmbeddingFormat>,
    /// User identifier for tracking and abuse monitoring
    pub user: Option<String>,
}

impl OpenAiEmbeddingOptions {
    /// Create new OpenAI embedding options with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set custom dimensions
    ///
    /// Only supported by text-embedding-3-small and text-embedding-3-large models.
    /// The dimensions must be less than or equal to the model's maximum dimensions.
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set encoding format
    ///
    /// - `Float`: Returns embeddings as arrays of floats (default)
    /// - `Base64`: Returns embeddings as base64-encoded strings (more compact)
    pub fn with_encoding_format(mut self, format: crate::types::EmbeddingFormat) -> Self {
        self.encoding_format = Some(format);
        self
    }

    /// Set user identifier
    ///
    /// A unique identifier representing your end-user, which can help OpenAI
    /// to monitor and detect abuse.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Apply these options to an EmbeddingRequest
    ///
    /// This method modifies the provided EmbeddingRequest to include
    /// OpenAI-specific parameters.
    pub fn apply_to_request(
        self,
        mut request: crate::types::EmbeddingRequest,
    ) -> crate::types::EmbeddingRequest {
        if let Some(dims) = self.dimensions {
            request.dimensions = Some(dims);
        }
        if let Some(format) = self.encoding_format {
            request.encoding_format = Some(format);
        }
        if let Some(user) = self.user {
            request.user = Some(user);
        }
        request
    }
}

/// Extension trait for EmbeddingRequest to add OpenAI-specific configuration
pub trait OpenAiEmbeddingRequestExt {
    /// Configure this request with OpenAI-specific options
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::types::EmbeddingRequest;
    /// use siumai::providers::openai::{OpenAiEmbeddingOptions, OpenAiEmbeddingRequestExt};
    /// use siumai::types::EmbeddingFormat;
    ///
    /// let request = EmbeddingRequest::new(vec!["text".to_string()])
    ///     .with_openai_config(
    ///         OpenAiEmbeddingOptions::new()
    ///             .with_dimensions(512)
    ///             .with_encoding_format(EmbeddingFormat::Float)
    ///     );
    /// ```
    fn with_openai_config(self, config: OpenAiEmbeddingOptions) -> Self;

    /// Quick method to set OpenAI custom dimensions
    fn with_openai_dimensions(self, dimensions: u32) -> Self;

    /// Quick method to set OpenAI encoding format
    fn with_openai_encoding_format(self, format: crate::types::EmbeddingFormat) -> Self;

    /// Quick method to set OpenAI user identifier
    fn with_openai_user(self, user: impl Into<String>) -> Self;
}

impl OpenAiEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_openai_config(self, config: OpenAiEmbeddingOptions) -> Self {
        config.apply_to_request(self)
    }

    fn with_openai_dimensions(self, dimensions: u32) -> Self {
        let mut request = self;
        request.dimensions = Some(dimensions);
        request
    }

    fn with_openai_encoding_format(self, format: crate::types::EmbeddingFormat) -> Self {
        let mut request = self;
        request.encoding_format = Some(format);
        request
    }

    fn with_openai_user(self, user: impl Into<String>) -> Self {
        let mut request = self;
        request.user = Some(user.into());
        request
    }
}
