//! Provider Builder Methods
//!
//! This module contains all provider-specific builder methods to keep the main builder clean.
//! Each provider gets its own builder method that returns the appropriate builder type.

use crate::builder::LlmBuilder;
use crate::siumai_for_each_openai_compatible_provider;

// Generate OpenAI-compatible builder methods from a single provider list
macro_rules! gen_llmbuilder_method {
    ($name:ident, $id:expr) => {
        #[cfg(feature = "openai")]
        pub fn $name(self) -> crate::providers::openai_compatible::OpenAiCompatibleBuilder {
            crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(self, $id)
        }
    };
}

/// OpenAI-compatible provider builder methods
impl LlmBuilder {
    // ========================================================================
    // Core Providers (with native implementations)
    // ========================================================================

    /// Create an `OpenAI` client builder.
    #[cfg(feature = "openai")]
    pub fn openai(self) -> crate::providers::openai::OpenAiBuilder {
        crate::providers::openai::OpenAiBuilder::new(self)
    }

    /// Create an `Anthropic` client builder.
    #[cfg(feature = "anthropic")]
    pub fn anthropic(self) -> crate::providers::anthropic::AnthropicBuilder {
        crate::providers::anthropic::AnthropicBuilder::new(self)
    }

    /// Create a `Google Gemini` client builder.
    #[cfg(feature = "google")]
    pub fn gemini(self) -> crate::providers::gemini::GeminiBuilder {
        crate::providers::gemini::GeminiBuilder::new(self)
    }

    /// Create an `Ollama` client builder.
    #[cfg(feature = "ollama")]
    pub fn ollama(self) -> crate::providers::ollama::OllamaBuilder {
        crate::providers::ollama::OllamaBuilder::new(self)
    }

    /// Create a `Groq` client builder (native implementation).
    #[cfg(feature = "groq")]
    pub fn groq(self) -> crate::providers::groq::GroqBuilder {
        crate::providers::groq::GroqBuilder::new(self)
    }

    /// Create an `xAI` client builder (native implementation).
    #[cfg(feature = "xai")]
    pub fn xai(self) -> crate::providers::xai::XaiBuilder {
        crate::providers::xai::XaiBuilder::new(self)
    }

    /// Create a `MiniMaxi` client builder (native implementation).
    #[cfg(feature = "minimaxi")]
    pub fn minimaxi(self) -> crate::providers::minimaxi::MinimaxiBuilder {
        crate::providers::minimaxi::MinimaxiBuilder::new(self)
    }

    // ========================================================================
    // OpenAI-Compatible Providers
    // ========================================================================

    // OpenAI-compatible providers (generated)
    siumai_for_each_openai_compatible_provider!(gen_llmbuilder_method);

    // ========================================================================
    // OpenAI-Compatible Versions of Native Providers
    // ========================================================================

    // The rest of native provider methods remain above

    // Platform aggregation provider methods are generated above by the macro
}
