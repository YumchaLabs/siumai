//! SiumaiBuilder Provider Methods
//!
//! This module contains all provider-specific methods for SiumaiBuilder to keep the main provider.rs clean.
//! Each provider gets its own method that sets the appropriate provider id.

use crate::provider::SiumaiBuilder;
use crate::provider::ids;
#[cfg(any(feature = "openai", feature = "deepseek"))]
use siumai_provider_openai_compatible::siumai_for_each_openai_compatible_provider;

// Generate SiumaiBuilder methods for all OpenAI-compatible providers
// Placed at module scope so methods can be expanded inside impl blocks.
#[cfg(any(feature = "openai", feature = "deepseek"))]
macro_rules! gen_siumaibuilder_method {
    (deepseek, $id:expr) => {
        #[cfg(feature = "deepseek")]
        pub fn deepseek(self) -> Self {
            self.provider_id(ids::DEEPSEEK)
        }
    };
    (cohere, $id:expr) => {
        #[cfg(all(feature = "openai", not(feature = "cohere")))]
        pub fn cohere(self) -> Self {
            self.provider_id($id)
        }
    };
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
        self.provider_id(ids::OPENAI)
    }

    /// Create an `OpenAI Chat` variant (explicit Chat Completions route)
    #[cfg(feature = "openai")]
    pub fn openai_chat(self) -> Self {
        self.provider_id(ids::OPENAI_CHAT)
    }

    /// Create an `OpenAI Responses` variant (explicit Responses API route)
    #[cfg(feature = "openai")]
    pub fn openai_responses(self) -> Self {
        self.provider_id(ids::OPENAI_RESPONSES)
    }

    /// Create an Azure OpenAI provider (Responses API by default, Vercel-aligned).
    #[cfg(feature = "azure")]
    pub fn azure(self) -> Self {
        self.provider_id(ids::AZURE)
    }

    /// Create an Azure OpenAI Chat Completions variant (Vercel-aligned `azure.chat`).
    #[cfg(feature = "azure")]
    pub fn azure_chat(self) -> Self {
        self.provider_id(ids::AZURE_CHAT)
    }

    /// Create an Anthropic provider (convenience method)
    #[cfg(feature = "anthropic")]
    pub fn anthropic(self) -> Self {
        self.provider_id(ids::ANTHROPIC)
    }

    /// Create a Cohere provider (convenience method)
    #[cfg(feature = "cohere")]
    pub fn cohere(self) -> Self {
        self.provider_id(ids::COHERE)
    }

    /// Create a Gemini provider (convenience method)
    #[cfg(feature = "google")]
    pub fn gemini(self) -> Self {
        self.provider_id(ids::GEMINI)
    }

    /// Create a Google Vertex provider (convenience method)
    #[cfg(feature = "google-vertex")]
    pub fn google_vertex(self) -> Self {
        self.provider_id(ids::VERTEX)
    }

    /// Create an Anthropic-on-Vertex provider (convenience method)
    #[cfg(feature = "google-vertex")]
    pub fn anthropic_vertex(self) -> Self {
        self.provider_id(ids::ANTHROPIC_VERTEX)
    }

    /// Alias for `google_vertex` (canonical provider id).
    #[cfg(feature = "google-vertex")]
    pub fn vertex(self) -> Self {
        self.google_vertex()
    }

    /// Create a TogetherAI provider (convenience method)
    #[cfg(feature = "togetherai")]
    pub fn togetherai(self) -> Self {
        self.provider_id(ids::TOGETHERAI)
    }

    /// Create an Amazon Bedrock provider (convenience method)
    #[cfg(feature = "bedrock")]
    pub fn bedrock(self) -> Self {
        self.provider_id(ids::BEDROCK)
    }

    /// Create an Ollama provider (convenience method)
    #[cfg(feature = "ollama")]
    pub fn ollama(self) -> Self {
        self.provider_id(ids::OLLAMA)
    }

    /// Create an xAI provider (convenience method)
    #[cfg(feature = "xai")]
    pub fn xai(self) -> Self {
        self.provider_id(ids::XAI)
    }

    /// Create a Groq provider (convenience method)
    #[cfg(feature = "groq")]
    pub fn groq(self) -> Self {
        self.provider_id(ids::GROQ)
    }

    /// Create a `MiniMaxi` provider (convenience method)
    #[cfg(feature = "minimaxi")]
    pub fn minimaxi(self) -> Self {
        self.provider_id(ids::MINIMAXI)
    }

    // ========================================================================
    // OpenAI-Compatible Providers (generated)
    // ========================================================================

    #[cfg(any(feature = "openai", feature = "deepseek"))]
    siumai_for_each_openai_compatible_provider!(gen_siumaibuilder_method);
}

#[cfg(test)]
mod tests {
    #[cfg(any(
        feature = "cohere",
        feature = "togetherai",
        feature = "bedrock",
        feature = "deepseek",
        feature = "google-vertex"
    ))]
    use super::*;

    #[test]
    #[cfg(feature = "cohere")]
    fn cohere_builder_method_sets_provider_id() {
        let builder = SiumaiBuilder::new().cohere();
        assert_eq!(builder.provider_id, Some(ids::COHERE.to_string()));
    }

    #[test]
    #[cfg(feature = "togetherai")]
    fn togetherai_builder_method_sets_provider_id() {
        let builder = SiumaiBuilder::new().togetherai();
        assert_eq!(builder.provider_id, Some(ids::TOGETHERAI.to_string()));
    }

    #[test]
    #[cfg(feature = "bedrock")]
    fn bedrock_builder_method_sets_provider_id() {
        let builder = SiumaiBuilder::new().bedrock();
        assert_eq!(builder.provider_id, Some(ids::BEDROCK.to_string()));
    }

    #[test]
    #[cfg(feature = "deepseek")]
    fn deepseek_builder_method_sets_provider_id() {
        let builder = SiumaiBuilder::new().deepseek();
        assert_eq!(builder.provider_id, Some(ids::DEEPSEEK.to_string()));
    }

    #[test]
    #[cfg(feature = "google-vertex")]
    fn anthropic_vertex_builder_method_sets_provider_id() {
        let builder = SiumaiBuilder::new().anthropic_vertex();
        assert_eq!(builder.provider_id, Some(ids::ANTHROPIC_VERTEX.to_string()));
    }
}
