use crate::client::LlmClient;
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use crate::traits::*;
use crate::types::*;
use std::sync::Arc;

#[allow(deprecated)]
use super::VisionCapabilityProxy;
use super::{AudioCapabilityProxy, EmbeddingCapabilityProxy, SiumaiBuilder};

mod audio;
mod chat;
mod embedding;
mod embedding_extensions;
mod files;
mod image_extras;
mod llm_client;
mod model_listing;
mod moderation;
mod music;
mod video;

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
