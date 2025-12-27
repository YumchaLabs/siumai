//! Client Module
//!
//! Defines a unified LLM client interface with dynamic dispatch support.

use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::*;
use crate::types::*;
use std::borrow::Cow;

/// Unified LLM client interface
pub trait LlmClient: ChatCapability + Send + Sync {
    /// Get the canonical provider id (e.g., "openai", "anthropic")
    fn provider_id(&self) -> Cow<'static, str>;

    /// Get the provider type. Default implementation maps from `provider_id()`.
    fn provider_type(&self) -> ProviderType {
        ProviderType::from_name(&self.provider_id())
    }

    /// Get the list of supported models
    fn supported_models(&self) -> Vec<String>;

    /// Get capability information
    fn capabilities(&self) -> ProviderCapabilities;

    /// Get as Any for dynamic casting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Clone the client into a boxed trait object
    fn clone_box(&self) -> Box<dyn LlmClient>;

    /// Get as embedding capability if supported
    ///
    /// Returns None by default. Providers that support embeddings
    /// should override this method to return Some(self).
    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        None
    }

    /// Get as audio capability if supported
    ///
    /// Returns None by default. Providers that support audio
    /// should override this method to return Some(self).
    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        None
    }

    /// Get as speech (TTS) capability if supported
    ///
    /// Returns None by default. Providers that support text-to-speech
    /// should override this method to return Some(self).
    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        None
    }

    /// Get as transcription (STT) capability if supported
    ///
    /// Returns None by default. Providers that support speech-to-text
    /// should override this method to return Some(self).
    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        None
    }

    /// Get as vision capability if supported
    ///
    /// Returns None by default. Providers that support vision
    /// should override this method to return Some(self).
    #[allow(deprecated)]
    fn as_vision_capability(&self) -> Option<&dyn VisionCapability> {
        None
    }

    /// Get as image generation capability if supported
    ///
    /// Returns None by default. Providers that support image generation
    /// should override this method to return Some(self).
    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        None
    }

    /// Get as image extras capability if supported (non-unified surface)
    ///
    /// Returns None by default. Providers that support image editing/variation
    /// or provide provider-specific image metadata should override this method
    /// to return Some(self).
    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        None
    }

    /// Get as file management capability if supported
    ///
    /// Returns None by default. Providers that support files
    /// should override this method to return Some(self).
    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        None
    }

    /// Get as moderation capability if supported
    ///
    /// Returns None by default. Providers that support moderation
    /// should override this method to return Some(self).
    fn as_moderation_capability(&self) -> Option<&dyn ModerationCapability> {
        None
    }

    /// Get as model listing capability if supported
    ///
    /// Returns None by default. Providers that support model listing
    /// should override this method to return Some(self).
    fn as_model_listing_capability(&self) -> Option<&dyn ModelListingCapability> {
        None
    }

    /// Get as rerank capability if supported
    ///
    /// Returns None by default. Providers that support rerank
    /// should override this method to return Some(self).
    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        None
    }

    /// Get as video generation capability if supported
    ///
    /// Returns None by default. Providers that support video generation
    /// should override this method to return Some(self).
    fn as_video_generation_capability(&self) -> Option<&dyn VideoGenerationCapability> {
        None
    }

    /// Get as music generation capability if supported
    ///
    /// Returns None by default. Providers that support music generation
    /// should override this method to return Some(self).
    fn as_music_generation_capability(&self) -> Option<&dyn MusicGenerationCapability> {
        None
    }
}

/// Client Wrapper - provides dynamic dispatch over provider clients
///
/// This wrapper allows storing different provider clients in a unified way,
/// enabling runtime polymorphism. It's primarily used for advanced scenarios
/// such as pooling or dynamic provider switching.
///
/// ## Usage
/// Most users should use the Builder pattern instead:
/// ```rust,no_run
/// use siumai::prelude::*;
///
/// async fn example() -> Result<(), Box<dyn std::error::Error>> {
///     // Preferred approach
///     let client = Siumai::builder()
///         .openai()
///         .api_key("key")
///         .build()
///         .await?;
///     Ok(())
/// }
/// ```
///
/// ## Advanced Usage
/// ClientWrapper is useful for advanced scenarios like client pools or
/// dynamic provider switching:
/// ```rust,no_run
/// use siumai::client::ClientWrapper;
/// use siumai::prelude::*;
///
/// async fn example() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a client first
///     let openai_client = Provider::openai()
///         .api_key("key")
///         .build()
///         .await?;
///
///     let wrapper = ClientWrapper::openai(Box::new(openai_client));
///     let provider_type = wrapper.provider_type();
///     let capabilities = wrapper.get_capabilities();
///     Ok(())
/// }
/// ```
pub enum ClientWrapper {
    /// Provider-agnostic wrapper around a boxed `LlmClient`.
    ///
    /// This exists primarily for pooling and dynamic dispatch scenarios where a concrete
    /// provider type is not known at compile time.
    ///
    /// Note: this intentionally does *not* encode provider identity in the type.
    /// Use `LlmClient::provider_id()` / `LlmClient::provider_type()` when needed.
    Client(Box<dyn LlmClient>),
}

impl Clone for ClientWrapper {
    fn clone(&self) -> Self {
        match self {
            ClientWrapper::Client(client) => ClientWrapper::Client(client.clone_box()),
        }
    }
}

impl std::fmt::Debug for ClientWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientWrapper::Client(_) => f
                .debug_tuple("ClientWrapper::Client")
                .field(&"[LlmClient]")
                .finish(),
        }
    }
}

impl ClientWrapper {
    /// Creates a provider-agnostic client wrapper.
    pub fn new(client: Box<dyn LlmClient>) -> Self {
        Self::Client(client)
    }

    /// Creates an `OpenAI` client wrapper
    pub fn openai(client: Box<dyn LlmClient>) -> Self {
        Self::new(client)
    }

    /// Creates an Anthropic client wrapper
    pub fn anthropic(client: Box<dyn LlmClient>) -> Self {
        Self::new(client)
    }

    /// Creates a Gemini client wrapper
    pub fn gemini(client: Box<dyn LlmClient>) -> Self {
        Self::new(client)
    }

    /// Creates a Groq client wrapper
    pub fn groq(client: Box<dyn LlmClient>) -> Self {
        Self::new(client)
    }

    /// Creates an xAI client wrapper
    pub fn xai(client: Box<dyn LlmClient>) -> Self {
        Self::new(client)
    }

    /// Creates an Ollama client wrapper
    pub fn ollama(client: Box<dyn LlmClient>) -> Self {
        Self::new(client)
    }

    /// Creates a custom client wrapper
    pub fn custom(client: Box<dyn LlmClient>) -> Self {
        Self::new(client)
    }

    /// Gets a reference to the internal client
    pub fn client(&self) -> &dyn LlmClient {
        match self {
            Self::Client(client) => client.as_ref(),
        }
    }

    /// Gets the provider type
    pub fn provider_type(&self) -> ProviderType {
        self.client().provider_type()
    }

    /// Check if the client supports a specific capability
    pub fn supports_capability(&self, capability: &str) -> bool {
        self.client().capabilities().supports(capability)
    }

    /// Get all supported capabilities
    pub fn get_capabilities(&self) -> ProviderCapabilities {
        self.client().capabilities()
    }
}

#[async_trait::async_trait]
impl ChatCapability for ClientWrapper {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.client().chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.client().chat_stream(messages, tools).await
    }
}

// UnifiedLlmClient has been removed as it was redundant with ClientWrapper.
//
// Use these alternatives instead:
// - Siumai::builder() for unified interface (recommended for most users)
// - ClientWrapper for dynamic dispatch (used internally)
// - Provider-specific clients for advanced features

// UnifiedLlmClient implementation removed - use ClientWrapper directly or Siumai::builder()

// UnifiedLlmClient trait implementations removed - functionality available through ClientWrapper

impl LlmClient for ClientWrapper {
    fn provider_id(&self) -> Cow<'static, str> {
        self.client().provider_id()
    }

    fn supported_models(&self) -> Vec<String> {
        self.client().supported_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.client().capabilities()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self.client().as_any()
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        self.client().as_embedding_capability()
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        self.client().as_audio_capability()
    }

    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        self.client().as_speech_capability()
    }

    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        self.client().as_transcription_capability()
    }

    #[allow(deprecated)]
    fn as_vision_capability(&self) -> Option<&dyn VisionCapability> {
        self.client().as_vision_capability()
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        self.client().as_image_generation_capability()
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        self.client().as_image_extras()
    }

    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        self.client().as_file_management_capability()
    }

    fn as_moderation_capability(&self) -> Option<&dyn ModerationCapability> {
        self.client().as_moderation_capability()
    }

    fn as_model_listing_capability(&self) -> Option<&dyn ModelListingCapability> {
        self.client().as_model_listing_capability()
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        self.client().as_rerank_capability()
    }

    fn as_video_generation_capability(&self) -> Option<&dyn VideoGenerationCapability> {
        self.client().as_video_generation_capability()
    }

    fn as_music_generation_capability(&self) -> Option<&dyn MusicGenerationCapability> {
        self.client().as_music_generation_capability()
    }
}

// Note: Connection pools and higher-level client management helpers
// have been moved to the `siumai-extras` crate (`siumai_extras::client`).
