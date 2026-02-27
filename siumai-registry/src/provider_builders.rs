//! SiumaiBuilder Provider Methods
//!
//! This module contains all provider-specific methods for SiumaiBuilder to keep the main provider.rs clean.
//! Each provider gets its own method that sets the appropriate provider id.

use crate::provider::SiumaiBuilder;
#[cfg(feature = "openai")]
use siumai_provider_openai_compatible::siumai_for_each_openai_compatible_provider;

// Generate SiumaiBuilder methods for all OpenAI-compatible providers
// Placed at module scope so methods can be expanded inside impl blocks.
#[cfg(feature = "openai")]
macro_rules! gen_siumaibuilder_method {
    ($name:ident, $id:expr) => {
        #[cfg(feature = "openai")]
        pub fn $name(self) -> Self {
            self.provider_id($id)
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
    pub fn openai(self) -> Self {
        self.provider_id("openai")
    }

    /// Create an `OpenAI Chat` variant (explicit Chat Completions route)
    #[cfg(feature = "openai")]
    pub fn openai_chat(self) -> Self {
        self.provider_id("openai-chat")
    }

    /// Create an `OpenAI Responses` variant (explicit Responses API route)
    #[cfg(feature = "openai")]
    pub fn openai_responses(self) -> Self {
        self.provider_id("openai-responses")
    }

    /// Create an Azure OpenAI provider (Responses API by default, Vercel-aligned).
    #[cfg(feature = "azure")]
    pub fn azure(self) -> Self {
        self.provider_id("azure")
    }

    /// Create an Azure OpenAI Chat Completions variant (Vercel-aligned `azure.chat`).
    #[cfg(feature = "azure")]
    pub fn azure_chat(self) -> Self {
        self.provider_id("azure-chat")
    }

    /// Create an Anthropic provider (convenience method)
    #[cfg(feature = "anthropic")]
    pub fn anthropic(self) -> Self {
        self.provider_id("anthropic")
    }

    /// Create a Gemini provider (convenience method)
    #[cfg(feature = "google")]
    pub fn gemini(self) -> Self {
        self.provider_id("gemini")
    }

    /// Create a Google Vertex provider (convenience method)
    #[cfg(feature = "google-vertex")]
    pub fn google_vertex(self) -> Self {
        self.provider_id("vertex")
    }

    /// Alias for `google_vertex` (canonical provider id).
    #[cfg(feature = "google-vertex")]
    pub fn vertex(self) -> Self {
        self.google_vertex()
    }

    /// Create an Ollama provider (convenience method)
    #[cfg(feature = "ollama")]
    pub fn ollama(self) -> Self {
        self.provider_id("ollama")
    }

    /// Create an xAI provider (convenience method)
    #[cfg(feature = "xai")]
    pub fn xai(self) -> Self {
        self.provider_id("xai")
    }

    /// Create a Groq provider (convenience method)
    #[cfg(feature = "groq")]
    pub fn groq(self) -> Self {
        self.provider_id("groq")
    }

    /// Create a `MiniMaxi` provider (convenience method)
    #[cfg(feature = "minimaxi")]
    pub fn minimaxi(self) -> Self {
        self.provider_id("minimaxi")
    }

    // ========================================================================
    // OpenAI-Compatible Providers (generated)
    // ========================================================================

    #[cfg(feature = "openai")]
    siumai_for_each_openai_compatible_provider!(gen_siumaibuilder_method);
}
