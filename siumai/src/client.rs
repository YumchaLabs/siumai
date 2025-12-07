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

    /// Get as vision capability if supported
    ///
    /// Returns None by default. Providers that support vision
    /// should override this method to return Some(self).
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

    /// Get as file management capability if supported
    ///
    /// Returns None by default. Providers that support files
    /// should override this method to return Some(self).
    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        None
    }

    /// Get as rerank capability if supported
    ///
    /// Returns None by default. Providers that support rerank
    /// should override this method to return Some(self).
    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        None
    }
}

/// Client Wrapper - provides dynamic dispatch for different provider clients
///
/// This enum allows storing different provider clients in a unified way,
/// enabling runtime polymorphism. It's primarily used internally by the library
/// for implementing the unified interface.
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
    OpenAi(Box<dyn LlmClient>),
    Anthropic(Box<dyn LlmClient>),
    Gemini(Box<dyn LlmClient>),
    Groq(Box<dyn LlmClient>),
    XAI(Box<dyn LlmClient>),
    Ollama(Box<dyn LlmClient>),
    Custom(Box<dyn LlmClient>),
}

impl Clone for ClientWrapper {
    fn clone(&self) -> Self {
        match self {
            ClientWrapper::OpenAi(client) => ClientWrapper::OpenAi(client.clone_box()),
            ClientWrapper::Anthropic(client) => ClientWrapper::Anthropic(client.clone_box()),
            ClientWrapper::Gemini(client) => ClientWrapper::Gemini(client.clone_box()),
            ClientWrapper::Groq(client) => ClientWrapper::Groq(client.clone_box()),
            ClientWrapper::XAI(client) => ClientWrapper::XAI(client.clone_box()),
            ClientWrapper::Ollama(client) => ClientWrapper::Ollama(client.clone_box()),
            ClientWrapper::Custom(client) => ClientWrapper::Custom(client.clone_box()),
        }
    }
}

impl std::fmt::Debug for ClientWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientWrapper::OpenAi(_) => f
                .debug_tuple("ClientWrapper::OpenAi")
                .field(&"[LlmClient]")
                .finish(),
            ClientWrapper::Anthropic(_) => f
                .debug_tuple("ClientWrapper::Anthropic")
                .field(&"[LlmClient]")
                .finish(),
            ClientWrapper::Gemini(_) => f
                .debug_tuple("ClientWrapper::Gemini")
                .field(&"[LlmClient]")
                .finish(),
            ClientWrapper::Groq(_) => f
                .debug_tuple("ClientWrapper::Groq")
                .field(&"[LlmClient]")
                .finish(),
            ClientWrapper::XAI(_) => f
                .debug_tuple("ClientWrapper::XAI")
                .field(&"[LlmClient]")
                .finish(),
            ClientWrapper::Ollama(_) => f
                .debug_tuple("ClientWrapper::Ollama")
                .field(&"[LlmClient]")
                .finish(),
            ClientWrapper::Custom(_) => f
                .debug_tuple("ClientWrapper::Custom")
                .field(&"[LlmClient]")
                .finish(),
        }
    }
}

impl ClientWrapper {
    /// Creates an `OpenAI` client wrapper
    pub fn openai(client: Box<dyn LlmClient>) -> Self {
        Self::OpenAi(client)
    }

    /// Creates an Anthropic client wrapper
    pub fn anthropic(client: Box<dyn LlmClient>) -> Self {
        Self::Anthropic(client)
    }

    /// Creates a Gemini client wrapper
    pub fn gemini(client: Box<dyn LlmClient>) -> Self {
        Self::Gemini(client)
    }

    /// Creates a Groq client wrapper
    pub fn groq(client: Box<dyn LlmClient>) -> Self {
        Self::Groq(client)
    }

    /// Creates an xAI client wrapper
    pub fn xai(client: Box<dyn LlmClient>) -> Self {
        Self::XAI(client)
    }

    /// Creates an Ollama client wrapper
    pub fn ollama(client: Box<dyn LlmClient>) -> Self {
        Self::Ollama(client)
    }

    /// Creates a custom client wrapper
    pub fn custom(client: Box<dyn LlmClient>) -> Self {
        Self::Custom(client)
    }

    /// Gets a reference to the internal client
    pub fn client(&self) -> &dyn LlmClient {
        match self {
            Self::OpenAi(client) => client.as_ref(),
            Self::Anthropic(client) => client.as_ref(),
            Self::Gemini(client) => client.as_ref(),
            Self::Groq(client) => client.as_ref(),
            Self::XAI(client) => client.as_ref(),
            Self::Ollama(client) => client.as_ref(),
            Self::Custom(client) => client.as_ref(),
        }
    }

    /// Gets the provider type
    pub fn provider_type(&self) -> ProviderType {
        match self {
            Self::OpenAi(_) => ProviderType::OpenAi,
            Self::Anthropic(_) => ProviderType::Anthropic,
            Self::Gemini(_) => ProviderType::Gemini,
            Self::Groq(_) => ProviderType::Groq,
            Self::XAI(_) => ProviderType::XAI,
            Self::Ollama(_) => ProviderType::Ollama,
            Self::Custom(_) => ProviderType::Custom("unknown".to_string()),
        }
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
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }
}

// Note: Connection pools and higher-level client management helpers
// have been moved to the `siumai-extras` crate (`siumai_extras::client`).
