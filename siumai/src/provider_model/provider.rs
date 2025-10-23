//! Provider Trait
//!
//! Defines the Provider trait which acts as a lightweight factory for creating Model instances.

use super::model::{ChatModel, EmbeddingModel, ImageModel, RerankModel};
use crate::error::LlmError;

/// Provider trait - lightweight factory for creating Model instances
///
/// A Provider is responsible for:
/// - Managing provider-level configuration (API key, base URL, etc.)
/// - Creating Model instances for different endpoints
/// - NOT responsible for business logic or HTTP requests
///
/// ## Design Principles
///
/// 1. **Lightweight**: Provider should only create Models, not execute requests
/// 2. **Flexible**: Each endpoint returns an independent Model
/// 3. **Extensible**: Providers can add custom endpoints (e.g., rerank, responses)
///
/// ## Example
///
/// ```rust,ignore
/// pub struct OpenAiProvider {
///     config: ProviderConfig,
/// }
///
/// impl Provider for OpenAiProvider {
///     fn id(&self) -> &str { "openai" }
///     
///     fn chat(&self, model: &str) -> Result<Box<dyn ChatModel>, LlmError> {
///         Ok(Box::new(OpenAiChatModel::new(self.config.clone(), model)))
///     }
///     
///     fn embedding(&self, model: &str) -> Result<Box<dyn EmbeddingModel>, LlmError> {
///         Ok(Box::new(OpenAiEmbeddingModel::new(self.config.clone(), model)))
///     }
/// }
/// ```
pub trait Provider: Send + Sync {
    /// Provider identifier (e.g., "openai", "anthropic", "gemini")
    fn id(&self) -> &str;

    /// Create a chat model instance
    ///
    /// # Arguments
    /// * `model` - Model name (e.g., "gpt-4", "claude-3-opus")
    ///
    /// # Returns
    /// A ChatModel instance that can create executors
    fn chat(&self, model: &str) -> Result<Box<dyn ChatModel>, LlmError>;

    /// Create an embedding model instance
    ///
    /// # Arguments
    /// * `model` - Model name (e.g., "text-embedding-3-small")
    ///
    /// # Returns
    /// An EmbeddingModel instance that can create executors
    fn embedding(&self, _model: &str) -> Result<Box<dyn EmbeddingModel>, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "Provider {} does not support embedding",
            self.id()
        )))
    }

    /// Create an image generation model instance
    ///
    /// # Arguments
    /// * `model` - Model name (e.g., "dall-e-3")
    ///
    /// # Returns
    /// An ImageModel instance that can create executors
    fn image(&self, _model: &str) -> Result<Box<dyn ImageModel>, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "Provider {} does not support image generation",
            self.id()
        )))
    }

    /// Create a rerank model instance
    ///
    /// # Arguments
    /// * `model` - Model name (e.g., "bge-reranker-v2-m3")
    ///
    /// # Returns
    /// A RerankModel instance that can create executors
    ///
    /// # Note
    /// This is an optional endpoint. Most providers don't support rerank.
    /// Only providers like SiliconFlow, Cohere implement this.
    fn rerank(&self, _model: &str) -> Result<Box<dyn RerankModel>, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "Provider {} does not support rerank",
            self.id()
        )))
    }
}
