//! Model Traits
//!
//! Defines Model traits for different endpoints (chat, embedding, image, rerank).
//! Models encapsulate endpoint-specific configuration and Executor creation logic.

use crate::executors::chat::HttpChatExecutor;
use crate::executors::embedding::HttpEmbeddingExecutor;
use crate::executors::image::HttpImageExecutor;
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

/// Chat Model trait
///
/// A ChatModel encapsulates:
/// - Model-specific configuration (model name, base URL, API key, etc.)
/// - Logic for creating HttpChatExecutor
/// - Integration with Standard Layer (e.g., OpenAiChatStandard)
///
/// ## Design Principles
///
/// 1. **Configuration Holder**: Holds endpoint-specific configuration
/// 2. **Executor Factory**: Creates HttpChatExecutor with proper transformers
/// 3. **Standard Integration**: Uses Standard Layer for reusable implementations
/// 4. **Adapter Support**: Supports Adapter pattern for provider-specific differences
///
/// ## Example
///
/// ```rust,ignore
/// pub struct OpenAiChatModel {
///     config: ModelConfig,
///     model: String,
///     adapter: Option<Arc<dyn OpenAiChatAdapter>>,
/// }
///
/// impl ChatModel for OpenAiChatModel {
///     fn create_executor(
///         &self,
///         http_client: reqwest::Client,
///         interceptors: Vec<Arc<dyn HttpInterceptor>>,
///         middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
///         retry_options: Option<RetryOptions>,
///     ) -> HttpChatExecutor {
///         // Use OpenAI Standard to create transformers
///         let standard = OpenAiChatStandard::new_with_adapter(self.adapter.clone());
///         let spec = standard.create_spec(&self.config.provider_id);
///         let ctx = ProviderContext::new(...);
///         let transformers = spec.choose_chat_transformers(&ChatRequest::default(), &ctx);
///         
///         // Create executor
///         HttpChatExecutor {
///             provider_id: self.config.provider_id.clone(),
///             http_client,
///             request_transformer: transformers.request,
///             response_transformer: transformers.response,
///             stream_transformer: transformers.stream,
///             interceptors,
///             middlewares,
///             retry_options,
///             // ...
///         }
///     }
/// }
/// ```
pub trait ChatModel: Send + Sync {
    /// Create an HttpChatExecutor with the given configuration
    ///
    /// # Arguments
    /// * `http_client` - reqwest::Client for HTTP requests
    /// * `interceptors` - HTTP interceptors for request/response observation
    /// * `middlewares` - Language model middlewares for request transformation
    /// * `retry_options` - Retry configuration (401 retry, idempotent, etc.)
    ///
    /// # Returns
    /// An HttpChatExecutor ready to execute chat requests
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpChatExecutor;
}

/// Embedding Model trait
///
/// Similar to ChatModel but for embedding endpoints.
pub trait EmbeddingModel: Send + Sync {
    /// Create an HttpEmbeddingExecutor with the given configuration
    ///
    /// # Arguments
    /// * `http_client` - reqwest::Client for HTTP requests
    /// * `interceptors` - HTTP interceptors for request/response observation
    /// * `retry_options` - Retry configuration (401 retry, idempotent, etc.)
    ///
    /// # Returns
    /// An HttpEmbeddingExecutor ready to execute embedding requests
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpEmbeddingExecutor;
}

/// Image Model trait
///
/// Similar to ChatModel but for image generation endpoints.
pub trait ImageModel: Send + Sync {
    /// Create an HttpImageExecutor with the given configuration
    ///
    /// # Arguments
    /// * `http_client` - reqwest::Client for HTTP requests
    /// * `interceptors` - HTTP interceptors for request/response observation
    /// * `retry_options` - Retry configuration (401 retry, idempotent, etc.)
    ///
    /// # Returns
    /// An HttpImageExecutor ready to execute image generation requests
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpImageExecutor;
}

/// Rerank Model trait
///
/// For providers that support rerank endpoints (e.g., SiliconFlow, Cohere).
pub trait RerankModel: Send + Sync {
    /// Create a rerank executor with the given configuration
    ///
    /// # Arguments
    /// * `http_client` - reqwest::Client for HTTP requests
    /// * `interceptors` - HTTP interceptors for request/response observation
    /// * `retry_options` - Retry configuration (401 retry, idempotent, etc.)
    ///
    /// # Returns
    /// A rerank executor ready to execute rerank requests
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> crate::executors::rerank::HttpRerankExecutor;
}
