//! Siumai LLM Interface
//!
//! This module provides the main siumai interface for calling different provider functionality,
//! similar to `llm_dart`'s approach. It uses dynamic dispatch to route calls to the
//! appropriate provider implementation.

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;
use crate::utils::http_interceptor::HttpInterceptor;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// The main siumai LLM provider that can dynamically dispatch to different capabilities
///
/// This is inspired by `llm_dart`'s unified interface design, allowing you to
/// call different provider functionality through a single interface.
pub struct Siumai {
    /// The underlying provider client
    client: Arc<dyn LlmClient>,
    /// Provider-specific metadata
    metadata: ProviderMetadata,
    /// Optional retry options for chat calls
    retry_options: Option<RetryOptions>,
}

impl Clone for Siumai {
    fn clone(&self) -> Self {
        // Clone the client using Arc (cheap reference counting)
        Self {
            client: Arc::clone(&self.client),
            metadata: self.metadata.clone(),
            retry_options: self.retry_options.clone(),
        }
    }
}

impl std::fmt::Debug for Siumai {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Siumai")
            .field("provider_type", &self.metadata.provider_type)
            .field("provider_name", &self.metadata.provider_name)
            .field(
                "supported_models_count",
                &self.metadata.supported_models.len(),
            )
            .field("capabilities", &self.metadata.capabilities)
            .finish()
    }
}

/// Metadata about the provider
#[derive(Debug, Clone)]
pub struct ProviderMetadata {
    pub provider_type: ProviderType,
    pub provider_name: String,
    pub supported_models: Vec<String>,
    pub capabilities: ProviderCapabilities,
}

impl Siumai {
    /// Create a new siumai provider
    pub fn new(client: Arc<dyn LlmClient>) -> Self {
        let metadata = ProviderMetadata {
            provider_type: client.provider_type(),
            provider_name: client.provider_name().to_string(),
            supported_models: client.supported_models(),
            capabilities: client.capabilities(),
        };

        Self {
            client,
            metadata,
            retry_options: None,
        }
    }

    /// Attach retry options (builder-style)
    pub fn with_retry_options(mut self, options: Option<RetryOptions>) -> Self {
        self.retry_options = options;
        self
    }

    /// Check if a capability is supported
    pub fn supports(&self, capability: &str) -> bool {
        self.metadata.capabilities.supports(capability)
    }

    /// Get provider metadata
    pub const fn metadata(&self) -> &ProviderMetadata {
        &self.metadata
    }

    /// Get the underlying client
    pub fn client(&self) -> &dyn LlmClient {
        self.client.as_ref()
    }

    /// Type-safe audio capability access
    ///
    /// Note: This method provides access regardless of reported capability support.
    /// Actual support depends on the specific model being used.
    pub fn audio_capability(&self) -> AudioCapabilityProxy<'_> {
        AudioCapabilityProxy::new(self, self.supports("audio"))
    }

    /// Type-safe embedding capability access
    ///
    /// Note: This method provides access regardless of reported capability support.
    /// Actual support depends on the specific model being used.
    pub fn embedding_capability(&self) -> EmbeddingCapabilityProxy<'_> {
        EmbeddingCapabilityProxy::new(self, self.supports("embedding"))
    }

    /// Type-safe vision capability access
    ///
    /// Note: This method provides access regardless of reported capability support.
    /// Actual support depends on the specific model being used.
    pub fn vision_capability(&self) -> VisionCapabilityProxy<'_> {
        VisionCapabilityProxy::new(self, self.supports("vision"))
    }

    /// Generate embeddings for the given input texts
    ///
    /// This is a convenience method that directly calls the embedding functionality
    /// without requiring the user to go through the capability proxy.
    ///
    /// # Arguments
    /// * `texts` - List of strings to generate embeddings for
    ///
    /// # Returns
    /// List of embedding vectors (one per input text)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Siumai::builder()
    ///     .openai()
    ///     .api_key("your-api-key")
    ///     .build()
    ///     .await?;
    ///
    /// let texts = vec!["Hello, world!".to_string()];
    /// let response = client.embed(texts).await?;
    /// println!("Got {} embeddings", response.embeddings.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        EmbeddingCapability::embed(self, texts).await
    }

    /// Generate images (unified)
    pub async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if let Some(img) = self.client.as_image_generation_capability() {
            img.generate_images(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support image generation.",
                self.client.provider_name()
            )))
        }
    }

    /// Files - upload (unified)
    pub async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.upload_file(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_name()
            )))
        }
    }

    /// Files - list (unified)
    pub async fn list_files(
        &self,
        query: Option<FileListQuery>,
    ) -> Result<FileListResponse, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.list_files(query).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_name()
            )))
        }
    }

    /// Files - retrieve (unified)
    pub async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.retrieve_file(file_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_name()
            )))
        }
    }

    /// Files - delete (unified)
    pub async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.delete_file(file_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_name()
            )))
        }
    }

    /// Files - get content (unified)
    pub async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.get_file_content(file_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_name()
            )))
        }
    }

    /// Image edit (unified)
    pub async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if let Some(img) = self.client.as_image_generation_capability() {
            img.edit_image(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support image generation.",
                self.client.provider_name()
            )))
        }
    }

    /// Image variation (unified)
    pub async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if let Some(img) = self.client.as_image_generation_capability() {
            img.create_variation(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support image generation.",
                self.client.provider_name()
            )))
        }
    }
}

#[async_trait::async_trait]
impl ChatCapability for Siumai {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let m = messages.clone();
                    let t = tools.clone();
                    async move { self.client.chat_with_tools(m, t).await }
                },
                opts.clone(),
            )
            .await
        } else {
            self.client.chat_with_tools(messages, tools).await
        }
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let m = messages.clone();
                    let t = tools.clone();
                    async move { self.client.chat_stream(m, t).await }
                },
                opts.clone(),
            )
            .await
        } else {
            self.client.chat_stream(messages, tools).await
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingCapability for Siumai {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        // Use the new capability method instead of downcasting
        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.embed(texts).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality. Consider using OpenAI, Gemini, or Ollama for embeddings.",
                self.client.provider_name()
            )))
        }
    }

    fn embedding_dimension(&self) -> usize {
        // Use the new capability method to get dimension
        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.embedding_dimension()
        } else {
            // Fallback to default dimension based on provider
            match self.client.provider_name() {
                "openai" => 1536,
                "ollama" => 384,
                "gemini" => 768,
                _ => 1536,
            }
        }
    }

    fn max_tokens_per_embedding(&self) -> usize {
        // Use the new capability method to get max tokens
        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.max_tokens_per_embedding()
        } else {
            // Fallback to default based on provider
            match self.client.provider_name() {
                "openai" => 8192,
                "ollama" => 8192,
                "gemini" => 2048,
                _ => 8192,
            }
        }
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        // Use the new capability method to get supported models
        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.supported_embedding_models()
        } else {
            // Fallback to default models based on provider
            match self.client.provider_name() {
                "openai" => vec![
                    "text-embedding-3-small".to_string(),
                    "text-embedding-3-large".to_string(),
                    "text-embedding-ada-002".to_string(),
                ],
                "ollama" => vec![
                    "nomic-embed-text".to_string(),
                    "mxbai-embed-large".to_string(),
                ],
                "gemini" => vec![
                    "embedding-001".to_string(),
                    "text-embedding-004".to_string(),
                ],
                _ => vec![],
            }
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingExtensions for Siumai {
    /// Generate embeddings with advanced configuration including task types.
    ///
    /// This method enables the unified interface to support provider-specific features
    /// like Gemini's task type optimization, OpenAI's custom dimensions, etc.
    ///
    /// # Arguments
    /// * `request` - Detailed embedding request with configuration
    ///
    /// # Returns
    /// Embedding response with vectors and metadata
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    /// use siumai::types::{EmbeddingRequest, EmbeddingTaskType};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Siumai::builder()
    ///         .gemini()
    ///         .api_key("your-api-key")
    ///         .model("gemini-embedding-001")
    ///         .build()
    ///         .await?;
    ///
    ///     // Use task type for optimization
    ///     let request = EmbeddingRequest::query("What is machine learning?");
    ///     let response = client.embed_with_config(request).await?;
    ///     Ok(())
    /// }
    /// ```
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        // For now, we'll provide a simplified implementation that works with the current architecture
        // The advanced features like task types will be supported through provider-specific interfaces

        if let Some(embedding_client) = self.client.as_embedding_capability() {
            // Use the basic embed method - the underlying implementations
            // can be enhanced to support more features in the future
            embedding_client.embed(request.input).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality. Consider using OpenAI, Gemini, or Ollama for embeddings.",
                self.client.provider_name()
            )))
        }
    }

    /// List available embedding models with their capabilities.
    ///
    /// # Returns
    /// Vector of embedding model information including supported task types
    async fn list_embedding_models(&self) -> Result<Vec<EmbeddingModelInfo>, LlmError> {
        // For now, provide basic model information based on the provider
        if let Some(_embedding_client) = self.client.as_embedding_capability() {
            let models = self.supported_embedding_models();
            let model_infos = models
                .into_iter()
                .map(|id| {
                    let mut model_info = EmbeddingModelInfo::new(
                        id.clone(),
                        id,
                        self.embedding_dimension(),
                        self.max_tokens_per_embedding(),
                    );

                    // Add task type support for Gemini models
                    if self.client.provider_name() == "gemini" {
                        model_info = model_info
                            .with_task(EmbeddingTaskType::RetrievalQuery)
                            .with_task(EmbeddingTaskType::RetrievalDocument)
                            .with_task(EmbeddingTaskType::SemanticSimilarity)
                            .with_task(EmbeddingTaskType::Classification)
                            .with_task(EmbeddingTaskType::Clustering)
                            .with_task(EmbeddingTaskType::QuestionAnswering)
                            .with_task(EmbeddingTaskType::FactVerification);
                    }

                    // Add custom dimensions support for OpenAI models
                    if self.client.provider_name() == "openai" {
                        model_info = model_info.with_custom_dimensions();
                    }

                    model_info
                })
                .collect();
            Ok(model_infos)
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality.",
                self.client.provider_name()
            )))
        }
    }
}

impl LlmClient for Siumai {
    fn provider_name(&self) -> &'static str {
        // We need to return a static str, so we'll use a match
        match self.metadata.provider_type {
            ProviderType::OpenAi => "openai",
            ProviderType::Anthropic => "anthropic",
            ProviderType::Gemini => "gemini",
            ProviderType::XAI => "xai",
            ProviderType::Ollama => "ollama",
            ProviderType::Custom(_) => "custom",
            ProviderType::Groq => "groq",
        }
    }

    fn supported_models(&self) -> Vec<String> {
        self.metadata.supported_models.clone()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.metadata.capabilities.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

/// Unified Interface Builder - Provider Abstraction Layer
///
/// ## 🎯 Core Responsibility: Unified Provider Interface
///
/// SiumaiBuilder provides a **unified interface** for creating LLM clients
/// across different providers while abstracting away provider-specific details.
///
/// ### ✅ What SiumaiBuilder Does:
/// - **Provider Abstraction**: Unified interface for all LLM providers
/// - **Parameter Unification**: Common parameter interface (temperature, max_tokens, etc.)
/// - **Reasoning Abstraction**: Unified reasoning interface across providers
/// - **Configuration Validation**: Validates configuration before client creation
/// - **Provider Selection**: Determines which provider to use based on configuration
/// - **Parameter Delegation**: Delegates to appropriate builders for actual construction
///
/// ### ❌ What SiumaiBuilder Does NOT Do:
/// - **Direct Client Creation**: Does not directly create HTTP clients
/// - **Parameter Mapping**: Does not handle provider-specific parameter mapping
/// - **HTTP Configuration**: Does not configure HTTP settings directly
///
/// ## 🏗️ Architecture Position
///
/// ```text
/// User Code
///     ↓
/// SiumaiBuilder (Unified Interface Layer) ← YOU ARE HERE
///     ↓
/// ┌─────────────────┬─────────────────────────────────────┐
/// ↓                 ↓                                     ↓
/// LlmBuilder        RequestBuilder                Provider Clients
/// (Client Config)   (Parameter Management)        (Implementation)
/// ```
///
/// ## 🔄 Delegation Pattern
///
/// SiumaiBuilder acts as a **coordinator** that delegates to specialized builders:
///
/// 1. **Parameter Validation**: Uses RequestBuilder for parameter validation
/// 2. **Client Construction**: Uses LlmBuilder or direct client constructors
/// 3. **Provider Selection**: Chooses appropriate implementation based on provider type
///
/// ### Example Flow:
/// ```rust,no_run
/// use siumai::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // 1. User configures through unified interface
///     let client = Siumai::builder()
///         .anthropic()                    // Provider selection
///         .api_key("your-api-key")        // Required API key
///         .model("claude-3-5-sonnet")     // Common parameter
///         .temperature(0.7)               // Common parameter
///         .reasoning(true)                // Unified reasoning
///         .build().await?;                // Delegation to appropriate builders
///     Ok(())
/// }
/// ```
///
/// This design allows users to switch providers with minimal code changes
/// while maintaining access to provider-specific features when needed.
pub struct SiumaiBuilder {
    pub(crate) provider_type: Option<ProviderType>,
    pub(crate) provider_name: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    capabilities: Vec<String>,
    common_params: CommonParams,
    http_config: HttpConfig,
    organization: Option<String>,
    project: Option<String>,
    tracing_config: Option<crate::tracing::TracingConfig>,
    // Unified reasoning configuration
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
    // Unified retry configuration
    retry_options: Option<RetryOptions>,
    // Optional provider-specific parameters provided by the user
    user_provider_params: Option<ProviderParams>,
    /// Optional HTTP interceptors applied to chat requests (unified interface)
    pub(crate) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data)
    pub(crate) http_debug: bool,
    #[cfg(feature = "google")]
    /// Optional Bearer token provider for Gemini (e.g., Vertex AI).
    pub(crate) gemini_token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
}

// Keep module slim by moving heavy build logic out
pub mod build;

impl SiumaiBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            provider_type: None,
            provider_name: None,
            api_key: None,
            base_url: None,
            capabilities: Vec::new(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            organization: None,
            project: None,
            tracing_config: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            retry_options: None,
            user_provider_params: None,
            http_interceptors: Vec::new(),
            http_debug: false,
            #[cfg(feature = "google")]
            gemini_token_provider: None,
        }
    }

    /// Set the provider type
    pub fn provider(mut self, provider_type: ProviderType) -> Self {
        self.provider_type = Some(provider_type);
        self
    }

    /// Set the provider by name (dynamic dispatch)
    /// This provides the llm_dart-style ai().provider('name') interface
    pub fn provider_name<S: Into<String>>(mut self, name: S) -> Self {
        let name = name.into();
        self.provider_name = Some(name.clone());

        // Map provider name to type
        self.provider_type = Some(match name.as_str() {
            "openai" => ProviderType::OpenAi,
            "openai-chat" => ProviderType::OpenAi,
            "openai-responses" => ProviderType::OpenAi,
            "anthropic" => ProviderType::Anthropic,
            "gemini" => ProviderType::Gemini,
            "ollama" => ProviderType::Ollama,
            "xai" => ProviderType::XAI,
            "groq" => ProviderType::Groq,
            "siliconflow" => ProviderType::Custom("siliconflow".to_string()),
            "deepseek" => ProviderType::Custom("deepseek".to_string()),
            "openrouter" => ProviderType::Custom("openrouter".to_string()),
            _ => ProviderType::Custom(name.clone()),
        });
        // Also set Responses API toggle automatically for known aliases
        #[cfg(feature = "openai")]
        {
            if name == "openai-chat" {
                let params = ProviderParams::new().with_param("responses_api", false);
                self.user_provider_params = Some(match self.user_provider_params.take() {
                    Some(p) => p.merge(params),
                    None => params,
                });
            } else if name == "openai-responses" {
                let params = ProviderParams::new().with_param("responses_api", true);
                self.user_provider_params = Some(match self.user_provider_params.take() {
                    Some(p) => p.merge(params),
                    None => params,
                });
            }
        }
        self
    }

    // Provider convenience methods are now defined in src/provider_builders.rs

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Attach provider-specific parameters (advanced usage)
    /// Use namespaced keys where applicable, e.g.:
    /// - "responses_api": true (OpenAI Responses API)
    /// - "thinking_budget": 4096 (Anthropic/Gemini)
    /// - "think": true (Ollama)
    pub fn with_provider_params(mut self, params: ProviderParams) -> Self {
        self.user_provider_params = Some(params);
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set temperature
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set top_p (nucleus sampling parameter)
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set random seed for reproducible outputs
    pub const fn seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Enable or disable reasoning mode (unified interface)
    ///
    /// This method provides a unified interface for enabling reasoning across all providers.
    /// It maps to provider-specific methods:
    /// - Anthropic: `thinking_budget` (10k tokens when enabled)
    /// - Gemini: `thinking` (dynamic when enabled)
    /// - Ollama: `reasoning` (enabled/disabled)
    /// - DeepSeek: `reasoning` (enabled/disabled)
    pub const fn reasoning(mut self, enabled: bool) -> Self {
        self.reasoning_enabled = Some(enabled);
        self
    }

    /// Set reasoning budget (unified interface)
    ///
    /// This method provides a unified interface for setting reasoning budgets.
    /// Different providers interpret this differently:
    /// - Anthropic: Direct token budget
    /// - Gemini: Token budget (-1 for dynamic, 0 for disabled)
    /// - Ollama: Ignored (uses boolean reasoning mode)
    /// - DeepSeek: Ignored (uses boolean reasoning mode)
    pub const fn reasoning_budget(mut self, budget: i32) -> Self {
        self.reasoning_budget = Some(budget);
        // If budget is set, automatically enable reasoning
        if budget > 0 {
            self.reasoning_enabled = Some(true);
        } else if budget == 0 {
            self.reasoning_enabled = Some(false);
        }
        self
    }

    /// Set organization (for `OpenAI`)
    pub fn organization<S: Into<String>>(mut self, organization: S) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Set project (for `OpenAI`)
    pub fn project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Enable a specific capability
    pub fn with_capability<S: Into<String>>(mut self, capability: S) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    /// Install a custom HTTP interceptor at the unified interface level.
    /// Interceptors will be applied to all chat requests for the built client.
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data).
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.http_debug = enabled;
        self
    }

    /// Enable audio capability
    pub fn with_audio(self) -> Self {
        self.with_capability("audio")
    }

    /// Enable vision capability
    pub fn with_vision(self) -> Self {
        self.with_capability("vision")
    }

    /// Enable embedding capability
    pub fn with_embedding(self) -> Self {
        self.with_capability("embedding")
    }

    /// Enable image generation capability
    pub fn with_image_generation(self) -> Self {
        self.with_capability("image_generation")
    }

    // === HTTP configuration (fine-grained) ===

    /// Set HTTP request timeout
    pub fn http_timeout(mut self, timeout: Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set HTTP connect timeout
    pub fn http_connect_timeout(mut self, timeout: Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Set HTTP user agent
    pub fn http_user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.http_config.user_agent = Some(user_agent.into());
        self
    }

    /// Set HTTP proxy URL
    pub fn http_proxy<S: Into<String>>(mut self, proxy_url: S) -> Self {
        self.http_config.proxy = Some(proxy_url.into());
        self
    }

    /// Add a default HTTP header
    pub fn http_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.http_config.headers.insert(key.into(), value.into());
        self
    }

    /// Merge multiple default HTTP headers
    pub fn http_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.http_config.headers.extend(headers);
        self
    }

    /// Control whether to disable compression for streaming (SSE) requests.
    ///
    /// When set to `true` (default), streaming requests send
    /// `Accept-Encoding: identity` to avoid proxy compression that can
    /// destabilize long-lived SSE connections.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    /// Convenience: Set `x-goog-user-project` billing/quota header for Google Vertex AI.
    /// This is a helper over `http_header` specialized for Vertex billing projects.
    pub fn with_x_goog_user_project(mut self, project_id: impl Into<String>) -> Self {
        self.http_config
            .headers
            .insert("x-goog-user-project".to_string(), project_id.into());
        self
    }

    /// Convenience: Build and set a Vertex AI base URL using project, location and publisher.
    /// Publisher should be `google` for Gemini, or `anthropic` for Anthropic on Vertex.
    pub fn base_url_for_vertex(mut self, project: &str, location: &str, publisher: &str) -> Self {
        let base = crate::utils::vertex::vertex_base_url(project, location, publisher);
        self.base_url = Some(base);
        self
    }

    /// Attach a Bearer token provider for Gemini (Vertex AI enterprise auth).
    #[cfg(feature = "google")]
    pub fn with_gemini_token_provider(
        mut self,
        provider: std::sync::Arc<dyn crate::auth::TokenProvider>,
    ) -> Self {
        self.gemini_token_provider = Some(provider);
        self
    }

    /// Use Application Default Credentials as Gemini (Vertex AI) token source.
    /// This is a convenience method that sets an ADC-based TokenProvider internally.
    #[cfg(feature = "google")]
    pub fn with_gemini_adc(mut self) -> Self {
        let http = reqwest::blocking::Client::new();
        let adc = crate::auth::adc::AdcTokenProvider::new(http);
        self.gemini_token_provider = Some(std::sync::Arc::new(adc));
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    ///
    /// This allows you to configure detailed tracing and monitoring for this client.
    /// The tracing configuration will override any global tracing settings.
    ///
    /// # Arguments
    /// * `config` - The tracing configuration to use
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Siumai::builder()
    ///     .openai()
    ///     .api_key("your-key")
    ///     .model("gpt-4o-mini")
    ///     .tracing(TracingConfig::debug())
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::json_production())
    }

    /// Enable simple tracing (uses debug configuration)
    pub fn enable_tracing(self) -> Self {
        self.debug_tracing()
    }

    /// Disable tracing explicitly
    pub fn disable_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::disabled())
    }

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options);
        self
    }

    /// Build the siumai provider (delegates to provider/build.rs)
    pub async fn build(self) -> Result<Siumai, LlmError> {
        crate::provider::build::build(self).await
    }
}

/// Type-safe proxy for audio capabilities
pub struct AudioCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> AudioCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports audio support (for reference only)
    ///
    /// Note: This is based on static capability information and may not reflect
    /// the actual capabilities of the current model. Use as a hint, not a restriction.
    /// The library will never block operations based on this information.
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider name for debugging
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }

    /// Get a support status message (optional, for user-controlled warnings)
    ///
    /// Returns a message about support status that you can choose to display or ignore.
    /// The library itself will not automatically warn or log anything.
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports audio support", self.provider_name())
        } else {
            format!(
                "Provider {} does not report audio support, but this may still work depending on the model",
                self.provider_name()
            )
        }
    }

    /// Placeholder for future audio operations
    ///
    /// This will attempt the operation regardless of reported support.
    /// Actual errors will come from the API if the model doesn't support it.
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Audio operations not yet implemented. Use provider-specific client.".to_string(),
        ))
    }
}

/// Type-safe proxy for embedding capabilities
pub struct EmbeddingCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> EmbeddingCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports embedding support (for reference only)
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider name for debugging
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!(
                "Provider {} reports embedding support",
                self.provider_name()
            )
        } else {
            format!(
                "Provider {} does not report embedding support, but this may still work depending on the model",
                self.provider_name()
            )
        }
    }

    /// Generate embeddings for the given input texts
    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.provider.embed(texts).await
    }

    /// Get the dimension of embeddings produced by this provider
    pub fn embedding_dimension(&self) -> usize {
        self.provider.embedding_dimension()
    }

    /// Get the maximum number of tokens that can be embedded at once
    pub fn max_tokens_per_embedding(&self) -> usize {
        self.provider.max_tokens_per_embedding()
    }

    /// Get supported embedding models for this provider
    pub fn supported_embedding_models(&self) -> Vec<String> {
        self.provider.supported_embedding_models()
    }

    // removed deprecated placeholder_operation; use embed() instead
}

/// Type-safe proxy for vision capabilities
pub struct VisionCapabilityProxy<'a> {
    provider: &'a Siumai,
    reported_support: bool,
}

impl<'a> VisionCapabilityProxy<'a> {
    pub const fn new(provider: &'a Siumai, reported_support: bool) -> Self {
        Self {
            provider,
            reported_support,
        }
    }

    /// Check if the provider reports vision support (for reference only)
    pub const fn is_reported_as_supported(&self) -> bool {
        self.reported_support
    }

    /// Get provider name for debugging
    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports vision support", self.provider_name())
        } else {
            format!(
                "Provider {} does not report vision support, but this may still work depending on the model",
                self.provider_name()
            )
        }
    }

    /// Placeholder for future vision operations
    pub async fn placeholder_operation(&self) -> Result<String, LlmError> {
        // No automatic warnings - let the user decide if they want to check support
        Err(LlmError::UnsupportedOperation(
            "Vision operations not yet implemented. Use provider-specific client.".to_string(),
        ))
    }
}

impl Default for SiumaiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SiumaiBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("SiumaiBuilder");

        debug_struct
            .field("provider_type", &self.provider_type)
            .field("provider_name", &self.provider_name)
            .field("base_url", &self.base_url)
            .field("model", &self.common_params.model)
            .field("temperature", &self.common_params.temperature)
            .field("max_tokens", &self.common_params.max_tokens)
            .field("top_p", &self.common_params.top_p)
            .field("seed", &self.common_params.seed)
            .field("capabilities_count", &self.capabilities.len())
            .field("reasoning_enabled", &self.reasoning_enabled)
            .field("reasoning_budget", &self.reasoning_budget)
            .field("has_tracing", &self.tracing_config.is_some())
            .field("timeout", &self.http_config.timeout);

        // Only show existence of sensitive fields, not their values
        if self.api_key.is_some() {
            debug_struct.field("has_api_key", &true);
        }
        if self.organization.is_some() {
            debug_struct.field("has_organization", &true);
        }
        if self.project.is_some() {
            debug_struct.field("has_project", &true);
        }

        debug_struct.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    // Mock provider for testing that doesn't support embedding
    #[derive(Debug)]
    struct MockProvider;

    #[async_trait]
    impl ChatCapability for MockProvider {
        async fn chat_with_tools(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                id: Some("mock-123".to_string()),
                content: MessageContent::Text("Mock response".to_string()),
                model: Some("mock-model".to_string()),
                usage: None,
                finish_reason: Some(FinishReason::Stop),
                tool_calls: None,
                thinking: None,
                metadata: std::collections::HashMap::new(),
            })
        }

        async fn chat_stream(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Option<Vec<Tool>>,
        ) -> Result<ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "Streaming not supported in mock".to_string(),
            ))
        }
    }

    impl LlmClient for MockProvider {
        fn provider_name(&self) -> &'static str {
            "mock"
        }

        fn supported_models(&self) -> Vec<String> {
            vec!["mock-model".to_string()]
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat()
            // Note: not adding .with_embedding() to test unsupported case
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn clone_box(&self) -> Box<dyn LlmClient> {
            Box::new(MockProvider)
        }
    }

    #[tokio::test]
    async fn test_siumai_embedding_unsupported_provider() {
        // Create a mock provider that doesn't support embedding
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Arc::new(mock_provider));

        // Test that embedding returns an error for unsupported provider
        let result = siumai.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());

        if let Err(LlmError::UnsupportedOperation(msg)) = result {
            assert!(msg.contains("does not support embedding functionality"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[test]
    fn test_embedding_capability_proxy() {
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Arc::new(mock_provider));

        let proxy = siumai.embedding_capability();
        assert_eq!(proxy.provider_name(), "custom"); // MockProvider gets mapped to "custom" type
        assert!(!proxy.is_reported_as_supported()); // Mock provider doesn't report embedding support
    }

    #[tokio::test]
    async fn test_embedding_capability_proxy_embed() {
        let mock_provider = MockProvider;
        let siumai = Siumai::new(Arc::new(mock_provider));

        let proxy = siumai.embedding_capability();
        let result = proxy.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());

        if let Err(LlmError::UnsupportedOperation(msg)) = result {
            assert!(msg.contains("does not support embedding functionality"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[tokio::test]
    async fn test_ollama_build_without_api_key() {
        // Test that Ollama can be built without API key
        let result = SiumaiBuilder::new()
            .ollama()
            .model("llama3.2")
            .build()
            .await;

        // This should not fail due to missing API key
        // Note: It might fail for other reasons (like Ollama not running), but not API key
        match result {
            Ok(_) => {
                // Success - Ollama client was created without API key
            }
            Err(LlmError::ConfigurationError(msg)) => {
                // Should not be an API key error
                assert!(
                    !msg.contains("API key not specified"),
                    "Ollama should not require API key, but got: {}",
                    msg
                );
            }
            Err(_) => {
                // Other errors are acceptable (e.g., network issues)
            }
        }
    }

    #[tokio::test]
    async fn test_openai_requires_api_key() {
        // Test that OpenAI still requires API key
        let result = SiumaiBuilder::new().openai().model("gpt-4o").build().await;

        // This should fail due to missing API key
        assert!(result.is_err());
        if let Err(LlmError::ConfigurationError(msg)) = result {
            assert!(msg.contains("API key not specified"));
        } else {
            panic!("Expected ConfigurationError for missing API key");
        }
    }
}
