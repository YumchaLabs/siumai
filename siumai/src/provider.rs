//! Siumai LLM Interface
//!
//! This module provides the main siumai interface for calling different provider functionality,
//! similar to `llm_dart`'s approach. It uses dynamic dispatch to route calls to the
//! appropriate provider implementation.

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::*;
use crate::types::*;
use std::borrow::Cow;
use std::sync::Arc;

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
            .field("provider_id", &self.metadata.provider_id)
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
    pub provider_id: String,
    pub supported_models: Vec<String>,
    pub capabilities: ProviderCapabilities,
}

impl Siumai {
    /// Create a new siumai provider
    pub fn new(client: Arc<dyn LlmClient>) -> Self {
        let metadata = ProviderMetadata {
            provider_type: client.provider_type(),
            provider_id: client.provider_id().into_owned(),
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
                self.client.provider_id()
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
                self.client.provider_id()
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
                self.client.provider_id()
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
                self.client.provider_id()
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
                self.client.provider_id()
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
                self.client.provider_id()
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
                self.client.provider_id()
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
                self.client.provider_id()
            )))
        }
    }

    /// Rerank documents (if supported by underlying provider)
    pub async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        if let Some(rerank_cap) = self.client.as_rerank_capability() {
            rerank_cap.rerank(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support rerank.",
                self.client.provider_id()
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
                self.client.provider_id()
            )))
        }
    }

    fn embedding_dimension(&self) -> usize {
        // Use the new capability method to get dimension
        if let Some(embedding_client) = self.client.as_embedding_capability() {
            embedding_client.embedding_dimension()
        } else {
            // Fallback to default dimension based on provider
            match self.client.provider_id().as_ref() {
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
            match self.client.provider_id().as_ref() {
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
            match self.client.provider_id().as_ref() {
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
                self.client.provider_id()
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
                    if self.client.provider_id() == "gemini" {
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
                    if self.client.provider_id() == "openai" {
                        model_info = model_info.with_custom_dimensions();
                    }

                    model_info
                })
                .collect();
            Ok(model_infos)
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support embedding functionality.",
                self.client.provider_id()
            )))
        }
    }
}

impl LlmClient for Siumai {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Owned(self.metadata.provider_id.clone())
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
mod siumai_builder;
pub use siumai_builder::SiumaiBuilder;

// Keep module slim by moving heavy build logic out
pub mod build;

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

    /// Get provider id for debugging
    pub fn provider_id(&self) -> String {
        self.provider.provider_id().into_owned()
    }

    /// Get a support status message (optional, for user-controlled warnings)
    ///
    /// Returns a message about support status that you can choose to display or ignore.
    /// The library itself will not automatically warn or log anything.
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports audio support", self.provider_id())
        } else {
            format!(
                "Provider {} does not report audio support, but this may still work depending on the model",
                self.provider_id()
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

    /// Get provider id for debugging
    pub fn provider_id(&self) -> String {
        self.provider.provider_id().into_owned()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports embedding support", self.provider_id())
        } else {
            format!(
                "Provider {} does not report embedding support, but this may still work depending on the model",
                self.provider_id()
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

    /// Get provider id for debugging
    pub fn provider_id(&self) -> String {
        self.provider.provider_id().into_owned()
    }

    /// Get a support status message (optional, for user-controlled information)
    pub fn support_status_message(&self) -> String {
        if self.reported_support {
            format!("Provider {} reports vision support", self.provider_id())
        } else {
            format!(
                "Provider {} does not report vision support, but this may still work depending on the model",
                self.provider_id()
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

// Debug/Default impl moved to provider/siumai_builder.rs

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
                system_fingerprint: None,
                service_tier: None,
                audio: None,
                warnings: None,
                provider_metadata: None,
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
        fn provider_id(&self) -> Cow<'static, str> {
            Cow::Borrowed("mock")
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
        assert_eq!(proxy.provider_id(), "mock");
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
