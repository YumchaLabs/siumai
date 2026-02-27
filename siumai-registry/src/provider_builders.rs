//! SiumaiBuilder Provider Methods
//!
//! This module contains all provider-specific methods for SiumaiBuilder to keep the main provider.rs clean.
//! Each provider gets its own method that sets the appropriate provider type and name.

use crate::provider::SiumaiBuilder;
#[cfg(any(
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
use crate::types::ProviderType;
#[cfg(feature = "openai")]
use siumai_provider_openai_compatible::siumai_for_each_openai_compatible_provider;

// Generate SiumaiBuilder methods for all OpenAI-compatible providers
// Placed at module scope so methods can be expanded inside impl blocks.
#[cfg(feature = "openai")]
macro_rules! gen_siumaibuilder_method {
    ($name:ident, $id:expr) => {
        #[cfg(feature = "openai")]
        pub fn $name(mut self) -> Self {
            self.provider_type = Some(ProviderType::Custom($id.to_string()));
            self.provider_id = Some($id.to_string());
            self
        }
    };
}

/// Provider builder methods for SiumaiBuilder
impl SiumaiBuilder {
    // ========================================================================
    // Core Providers (with native implementations)
    // ========================================================================

    /// Create an `OpenAI` provider (convenience method)
    #[cfg(feature = "openai")]
    pub fn openai(mut self) -> Self {
        self.provider_type = Some(ProviderType::OpenAi);
        self.provider_id = Some("openai".to_string());
        self
    }

    /// Create an `OpenAI Chat` variant (explicit Chat Completions route)
    #[cfg(feature = "openai")]
    pub fn openai_chat(mut self) -> Self {
        self.provider_type = Some(ProviderType::OpenAi);
        self.provider_id = Some("openai-chat".to_string());
        self
    }

    /// Create an `OpenAI Responses` variant (explicit Responses API route)
    #[cfg(feature = "openai")]
    pub fn openai_responses(mut self) -> Self {
        self.provider_type = Some(ProviderType::OpenAi);
        self.provider_id = Some("openai-responses".to_string());
        self
    }

    /// Create an Azure OpenAI provider (Responses API by default, Vercel-aligned).
    #[cfg(feature = "azure")]
    pub fn azure(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("azure".to_string()));
        self.provider_id = Some("azure".to_string());
        self
    }

    /// Create an Azure OpenAI Chat Completions variant (Vercel-aligned `azure.chat`).
    #[cfg(feature = "azure")]
    pub fn azure_chat(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("azure".to_string()));
        self.provider_id = Some("azure-chat".to_string());
        self
    }

    /// Create an Anthropic provider (convenience method)
    #[cfg(feature = "anthropic")]
    pub fn anthropic(mut self) -> Self {
        self.provider_type = Some(ProviderType::Anthropic);
        self.provider_id = Some("anthropic".to_string());
        self
    }

    /// Create a Gemini provider (convenience method)
    #[cfg(feature = "google")]
    pub fn gemini(mut self) -> Self {
        self.provider_type = Some(ProviderType::Gemini);
        self.provider_id = Some("gemini".to_string());
        self
    }

    /// Create a Google Vertex provider (convenience method)
    #[cfg(feature = "google-vertex")]
    pub fn google_vertex(mut self) -> Self {
        self.provider_type = Some(ProviderType::Custom("vertex".to_string()));
        self.provider_id = Some("vertex".to_string());
        self
    }

    /// Alias for `google_vertex` (canonical provider id).
    #[cfg(feature = "google-vertex")]
    pub fn vertex(self) -> Self {
        self.google_vertex()
    }

    /// Create an Ollama provider (convenience method)
    #[cfg(feature = "ollama")]
    pub fn ollama(mut self) -> Self {
        self.provider_type = Some(ProviderType::Ollama);
        self.provider_id = Some("ollama".to_string());
        self
    }

    /// Create an xAI provider (convenience method)
    #[cfg(feature = "xai")]
    pub fn xai(mut self) -> Self {
        self.provider_type = Some(ProviderType::XAI);
        self.provider_id = Some("xai".to_string());
        self
    }

    /// Create a Groq provider (convenience method)
    #[cfg(feature = "groq")]
    pub fn groq(mut self) -> Self {
        self.provider_type = Some(ProviderType::Groq);
        self.provider_id = Some("groq".to_string());
        self
    }

    /// Create a `MiniMaxi` provider (convenience method)
    #[cfg(feature = "minimaxi")]
    pub fn minimaxi(mut self) -> Self {
        self.provider_type = Some(ProviderType::MiniMaxi);
        self.provider_id = Some("minimaxi".to_string());
        self
    }

    // ========================================================================
    // OpenAI-Compatible Providers (generated)
    // ========================================================================

    #[cfg(feature = "openai")]
    siumai_for_each_openai_compatible_provider!(gen_siumaibuilder_method);
}
