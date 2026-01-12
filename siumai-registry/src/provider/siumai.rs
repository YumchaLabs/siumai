use crate::client::LlmClient;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::*;
use crate::types::*;
use std::borrow::Cow;
use std::sync::Arc;

use super::{AudioCapabilityProxy, EmbeddingCapabilityProxy, SiumaiBuilder, VisionCapabilityProxy};

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
    /// Create a new Siumai builder for unified interface.
    pub fn builder() -> SiumaiBuilder {
        SiumaiBuilder::new()
    }

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

    /// Create a siumai provider from a registry language model handle.
    ///
    /// This allows using the Vercel-style `registry.language_model("provider:model")`
    /// entry point and still benefit from the unified `Siumai` interface.
    pub fn from_language_model_handle(handle: crate::registry::LanguageModelHandle) -> Self {
        // Delegate metadata discovery to `LlmClient` methods on the handle; this avoids
        // runtime lookups into the global provider registry.
        Self::new(Arc::new(handle))
    }

    /// Convenience constructor: build from a registry model id like `"openai:gpt-4"`.
    #[cfg(feature = "builtins")]
    pub fn from_registry_model(id: &str) -> Result<Self, LlmError> {
        let handle = crate::registry::global().language_model(id)?;
        Ok(Self::from_language_model_handle(handle))
    }

    /// Convenience constructor: build from a registry model id like `"openai:gpt-4"`.
    #[cfg(not(feature = "builtins"))]
    pub fn from_registry_model(_id: &str) -> Result<Self, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "registry::global() is not available without built-in providers; enable a provider feature (e.g. `openai`) or construct a registry handle manually".to_string(),
        ))
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

    /// Escape hatch: downcast the underlying provider client.
    ///
    /// This is the recommended way to access provider-specific extension APIs while
    /// still constructing clients through the unified `Siumai` interface.
    pub fn downcast_client<T: 'static>(&self) -> Option<&T> {
        self.client.as_any().downcast_ref::<T>()
    }

    /// Escape hatch: downcast + clone the underlying provider client.
    ///
    /// This is useful when you need to call provider-specific builder-style APIs that
    /// consume `self` (common in Rust fluent APIs).
    pub fn downcast_client_cloned<T>(&self) -> Option<T>
    where
        T: Clone + 'static,
    {
        self.downcast_client::<T>().cloned()
    }

    /// Get a clone of the underlying client as `Arc`.
    pub fn client_arc(&self) -> Arc<dyn LlmClient> {
        Arc::clone(&self.client)
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
    #[deprecated(
        since = "0.11.0-beta.5",
        note = "Vercel-aligned unified surface does not expose a Vision capability. For image understanding, send images as multimodal Chat messages; for image generation, use generate_images()."
    )]
    #[allow(deprecated)]
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
impl ImageExtras for Siumai {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if let Some(img) = self.client.as_image_extras() {
            img.edit_image(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support image extras.",
                self.client.provider_id()
            )))
        }
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if let Some(img) = self.client.as_image_extras() {
            img.create_variation(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support image extras.",
                self.client.provider_id()
            )))
        }
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        self.client
            .as_image_extras()
            .map(|i| i.get_supported_sizes())
            .unwrap_or_default()
    }

    fn get_supported_formats(&self) -> Vec<String> {
        self.client
            .as_image_extras()
            .map(|i| i.get_supported_formats())
            .unwrap_or_default()
    }

    fn supports_image_editing(&self) -> bool {
        self.client
            .as_image_extras()
            .map(|i| i.supports_image_editing())
            .unwrap_or(false)
    }

    fn supports_image_variations(&self) -> bool {
        self.client
            .as_image_extras()
            .map(|i| i.supports_image_variations())
            .unwrap_or(false)
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
impl VideoGenerationCapability for Siumai {
    async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        if let Some(video) = self.client.as_video_generation_capability() {
            video.create_video_task(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support video generation.",
                self.client.provider_id()
            )))
        }
    }

    async fn query_video_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        if let Some(video) = self.client.as_video_generation_capability() {
            video.query_video_task(task_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support video generation.",
                self.client.provider_id()
            )))
        }
    }

    fn get_supported_models(&self) -> Vec<String> {
        self.client
            .as_video_generation_capability()
            .map(|v| v.get_supported_models())
            .unwrap_or_default()
    }

    fn get_supported_resolutions(&self, model: &str) -> Vec<String> {
        self.client
            .as_video_generation_capability()
            .map(|v| v.get_supported_resolutions(model))
            .unwrap_or_default()
    }

    fn get_supported_durations(&self, model: &str) -> Vec<u32> {
        self.client
            .as_video_generation_capability()
            .map(|v| v.get_supported_durations(model))
            .unwrap_or_default()
    }
}

#[async_trait::async_trait]
impl MusicGenerationCapability for Siumai {
    async fn generate_music(
        &self,
        request: MusicGenerationRequest,
    ) -> Result<MusicGenerationResponse, LlmError> {
        if let Some(music) = self.client.as_music_generation_capability() {
            music.generate_music(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support music generation.",
                self.client.provider_id()
            )))
        }
    }

    fn get_supported_music_models(&self) -> Vec<String> {
        self.client
            .as_music_generation_capability()
            .map(|m| m.get_supported_music_models())
            .unwrap_or_default()
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        self.client
            .as_music_generation_capability()
            .map(|m| m.get_supported_audio_formats())
            .unwrap_or_default()
    }

    fn supports_lyrics(&self) -> bool {
        self.client
            .as_music_generation_capability()
            .map(|m| m.supports_lyrics())
            .unwrap_or(false)
    }
}

#[async_trait::async_trait]
impl AudioCapability for Siumai {
    fn supported_features(&self) -> &[AudioFeature] {
        self.client
            .as_audio_capability()
            .map(|a| a.supported_features())
            .unwrap_or(&[])
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.text_to_speech(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support audio.",
                self.client.provider_id()
            )))
        }
    }

    async fn text_to_speech_stream(&self, request: TtsRequest) -> Result<AudioStream, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.text_to_speech_stream(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support audio streaming.",
                self.client.provider_id()
            )))
        }
    }

    async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.speech_to_text(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support speech-to-text.",
                self.client.provider_id()
            )))
        }
    }

    async fn speech_to_text_stream(&self, request: SttRequest) -> Result<AudioStream, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.speech_to_text_stream(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support streaming speech-to-text.",
                self.client.provider_id()
            )))
        }
    }

    async fn translate_audio(
        &self,
        request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.translate_audio(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support audio translation.",
                self.client.provider_id()
            )))
        }
    }

    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.get_voices().await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support voice listing.",
                self.client.provider_id()
            )))
        }
    }

    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.get_supported_languages().await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support language listing.",
                self.client.provider_id()
            )))
        }
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        self.client
            .as_audio_capability()
            .map(|a| a.get_supported_audio_formats())
            .unwrap_or_else(|| vec!["mp3".to_string(), "wav".to_string(), "ogg".to_string()])
    }
}

#[async_trait::async_trait]
impl FileManagementCapability for Siumai {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.upload_file(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.list_files(query).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.retrieve_file(file_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.delete_file(file_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        if let Some(files) = self.client.as_file_management_capability() {
            files.get_file_content(file_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support file management.",
                self.client.provider_id()
            )))
        }
    }
}

#[async_trait::async_trait]
impl ModerationCapability for Siumai {
    async fn moderate(&self, request: ModerationRequest) -> Result<ModerationResponse, LlmError> {
        if let Some(m) = self.client.as_moderation_capability() {
            m.moderate(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support moderation.",
                self.client.provider_id()
            )))
        }
    }
}

#[async_trait::async_trait]
impl ModelListingCapability for Siumai {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        if let Some(m) = self.client.as_model_listing_capability() {
            m.list_models().await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support model listing.",
                self.client.provider_id()
            )))
        }
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        if let Some(m) = self.client.as_model_listing_capability() {
            m.get_model(model_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support model listing.",
                self.client.provider_id()
            )))
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

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        self.client.as_embedding_capability()
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        self.client.as_audio_capability()
    }

    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        self.client.as_speech_capability()
    }

    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        self.client.as_transcription_capability()
    }

    #[allow(deprecated)]
    fn as_vision_capability(&self) -> Option<&dyn VisionCapability> {
        self.client.as_vision_capability()
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        self.client.as_image_generation_capability()
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        self.client.as_image_extras()
    }

    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        self.client.as_file_management_capability()
    }

    fn as_moderation_capability(&self) -> Option<&dyn ModerationCapability> {
        self.client.as_moderation_capability()
    }

    fn as_model_listing_capability(&self) -> Option<&dyn ModelListingCapability> {
        self.client.as_model_listing_capability()
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        self.client.as_rerank_capability()
    }

    fn as_video_generation_capability(&self) -> Option<&dyn VideoGenerationCapability> {
        self.client.as_video_generation_capability()
    }

    fn as_music_generation_capability(&self) -> Option<&dyn MusicGenerationCapability> {
        self.client.as_music_generation_capability()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}
