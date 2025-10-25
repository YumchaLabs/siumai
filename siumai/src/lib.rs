//! # Siumai - A Unified LLM Interface Library
//!
//! Siumai is a unified LLM interface library for Rust, supporting multiple AI providers.
//! It adopts a trait-separated architectural pattern and provides a type-safe API.
//!
#![deny(unsafe_code)]

//! ## Features
//!
//! - **Capability Separation**: Uses traits to distinguish different AI capabilities (chat, audio, vision, etc.)
//! - **Shared Parameters**: AI parameters are shared as much as possible, with extension points for provider-specific parameters.
//! - **Builder Pattern**: Supports a builder pattern for chained method calls.
//! - **Type Safety**: Leverages Rust's type system to ensure compile-time safety.
//! - **HTTP Customization**: Supports passing in a reqwest client and custom HTTP configurations.
//! - **Library First**: Focuses on core library functionality, avoiding application-layer features.
//! - **Flexible Capability Access**: Capability checks serve as hints rather than restrictions, allowing users to try new model features.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create an OpenAI client
//!     let client = LlmBuilder::new()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4")
//!         .temperature(0.7)
//!         .build()
//!         .await?;
//!
//!     // Send a chat request
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!     if let Some(text) = response.content_text() {
//!         println!("Response: {}", text);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Capability Access Philosophy
//!
//! Siumai takes a **permissive and quiet approach** to capability access. It never blocks operations
//! based on static capability information, and doesn't generate noise with automatic warnings.
//! The actual API determines what's supported:
//!
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Siumai::builder()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4o")  // This model supports vision
//!         .build()
//!         .await?;
//!
//!     // Get vision capability - this always works, regardless of "official" support
//!     let vision = client.vision_capability();
//!
//!     // Optionally check support status if you want to (no automatic warnings)
//!     if !vision.is_reported_as_supported() {
//!         // You can choose to show a warning, or just proceed silently
//!         println!("Note: Vision not officially supported, but trying anyway!");
//!     }
//!
//!     // The actual operation will succeed or fail based on the model's real capabilities
//!     // No pre-emptive blocking, no automatic noise
//!     // vision.analyze_image(...).await?;
//!
//!     Ok(())
//! }
//! ```

/// Enabled providers at compile time
pub const ENABLED_PROVIDERS: &str = env!("SIUMAI_ENABLED_PROVIDERS");

/// Number of enabled providers at compile time  
pub const PROVIDER_COUNT: &str = env!("SIUMAI_PROVIDER_COUNT");

pub mod analysis;
pub mod auth;
pub mod benchmarks;
pub mod builder;
pub mod client;
pub mod core;
pub mod custom_provider;
pub mod defaults;
pub mod error;
pub mod execution;
pub mod params;
pub mod performance;
pub mod provider;
pub mod provider_builders;
pub mod provider_features;
#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq"
))]
pub mod providers;
pub mod registry;
pub mod retry;
pub mod retry_api;
pub mod streaming;
pub mod telemetry;
pub mod observability;
pub mod traits;
pub mod types;
pub mod utils;
// Cancellation helpers are in `utils::cancel`
pub mod web_search;

// Refactor modules now part of the core (no feature gates)
pub mod highlevel;
pub mod orchestrator;
pub mod provider_tools;
pub mod public;
pub mod standards;

// Re-export main types and traits
pub use error::LlmError;

// Core traits
pub use traits::{
    AudioCapability, ChatCapability, EmbeddingCapability, FileManagementCapability,
    ImageGenerationCapability, ModelListingCapability, ModerationCapability, ProviderCapabilities,
    RerankCapability, VisionCapability,
};

// Client trait
pub use client::LlmClient;

// Core types (only re-export commonly used types)
pub use types::{
    ChatMessage, ChatResponse, CommonParams, CompletionRequest, CompletionResponse,
    EmbeddingRequest, EmbeddingResponse, FinishReason, HttpConfig, ImageGenerationRequest,
    ImageGenerationResponse, MessageContent, MessageRole, ModelInfo, ModerationRequest,
    ModerationResponse, ProviderDefinedTool, ProviderType, ResponseMetadata, Tool, ToolChoice,
    Usage,
};

// Builders
pub use builder::LlmBuilder;
#[cfg(feature = "anthropic")]
pub use providers::anthropic::AnthropicBuilder;
#[cfg(feature = "google")]
pub use providers::gemini::GeminiBuilder;
#[cfg(feature = "ollama")]
pub use providers::ollama::OllamaBuilder;
#[cfg(feature = "openai")]
pub use providers::openai::OpenAiBuilder;

// Streaming
pub use streaming::ChatStreamHandle;
pub use streaming::{ChatStream, ChatStreamEvent};
// High-level object generation
pub use highlevel::object::{
    GenerateMode, GenerateObjectOptions, OutputKind, StreamObjectEvent, StreamObjectOptions,
    generate_object, stream_object,
};
#[cfg(feature = "openai")]
pub use highlevel::object::{generate_object_openai, stream_object_openai};
// Provider-agnostic auto (provider params hints)
pub use highlevel::object::{generate_object_auto, stream_object_auto};

// Web search (use types re-export)
pub use types::{WebSearchConfig, WebSearchResult};

// Performance monitoring
// Performance types are available under `crate::performance` module; no top-level re-export

// Unified retry facade
pub use retry_api::{RetryBackend, RetryOptions, retry, retry_for_provider, retry_with};

// Benchmarks
// Benchmark types are available under `crate::benchmarks` module; no top-level re-export

// Custom provider support
pub use custom_provider::{CustomProvider, CustomProviderConfig};

// Provider features
pub use provider_features::ProviderFeatures;

// Registry - unified provider access (recommended)
pub use registry::{
    EmbeddingModelHandle, ImageModelHandle, LanguageModelHandle, ProviderRegistryHandle,
    global as registry_global,
};

// Model constants (simplified access)
pub use types::models::model_constants as models;

// Model constants (detailed access)
pub use types::models::constants;

/// Convenient pre-import module
pub mod prelude {
    pub use crate::benchmarks::*;
    pub use crate::builder::*;
    pub use crate::client::*;
    pub use crate::custom_provider::*;
    pub use crate::error::LlmError;
    // Multimodal utilities are internal; integrate via provider capabilities
    // Performance helpers are available via `crate::performance` but not in prelude
    pub use crate::provider::Siumai;
    pub use crate::provider::*;
    pub use crate::provider_features::*;
    pub use crate::retry_api::*;
    pub use crate::streaming::*;
    pub use crate::traits::*;
    pub use crate::types::*;
    pub use crate::web_search::*;
    // Model constants for easy access
    pub use crate::constants;
    pub use crate::models;
    pub use crate::{Provider, assistant, provider, system, tool, user, user_with_image};
    pub use crate::{conversation, conversation_with_system, messages, quick_chat};

    // Registry - unified provider access
    /// Provider registry module for unified access to all providers
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let reg = registry::global();
    /// let model = reg.language_model("openai:gpt-4")?;
    /// let resp = model.chat(vec![user!("Hello!")]).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub mod registry {
        pub use crate::registry::{
            EmbeddingModelHandle, ImageModelHandle, LanguageModelHandle, ProviderFactory,
            ProviderRegistryHandle, create_provider_registry, create_registry_with_defaults,
            global,
        };
    }

    // Conditional provider quick functions
    #[cfg(feature = "anthropic")]
    pub use crate::{quick_anthropic, quick_anthropic_with_model};
    #[cfg(feature = "google")]
    pub use crate::{quick_gemini, quick_gemini_with_model};
    #[cfg(feature = "groq")]
    pub use crate::{quick_groq, quick_groq_with_model};
    #[cfg(feature = "openai")]
    pub use crate::{quick_openai, quick_openai_with_model};
}

/// Provider entry point for creating specific provider clients
///
/// This is the main entry point for creating provider-specific clients.
/// Use this when you need access to provider-specific features and APIs.
///
/// # Example
/// ```rust,no_run
/// use siumai::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Get a client specifically for OpenAI
///     let openai_client = Provider::openai()
///         .api_key("your-openai-key")
///         .model("gpt-4")
///         .build()
///         .await?;
///
///     // You can now call both standard and OpenAI-specific methods
///     let messages = vec![user!("Hello!")];
///     let response = openai_client.chat(messages).await?;
///     println!("OpenAI says: {}", response.text().unwrap_or_default());
///     // let assistant = openai_client.create_assistant(...).await?; // Example of specific feature
///
///     Ok(())
/// }
/// ```
pub struct Provider;

impl Provider {
    /// Create an `OpenAI` client builder
    #[cfg(feature = "openai")]
    pub fn openai() -> providers::openai::OpenAiBuilder {
        crate::builder::LlmBuilder::new().openai()
    }

    /// Create an Anthropic client builder
    #[cfg(feature = "anthropic")]
    pub fn anthropic() -> providers::anthropic::AnthropicBuilder {
        crate::builder::LlmBuilder::new().anthropic()
    }

    /// Create a Gemini client builder
    #[cfg(feature = "google")]
    pub fn gemini() -> providers::gemini::GeminiBuilder {
        crate::builder::LlmBuilder::new().gemini()
    }

    /// Create an Ollama client builder
    #[cfg(feature = "ollama")]
    pub fn ollama() -> providers::ollama::OllamaBuilder {
        crate::builder::LlmBuilder::new().ollama()
    }

    /// Create an xAI client builder
    #[cfg(feature = "xai")]
    pub fn xai() -> crate::providers::xai::XaiBuilder {
        crate::providers::xai::XaiBuilder::new(crate::builder::LlmBuilder::new())
    }

    /// Create a Groq client builder
    #[cfg(feature = "groq")]
    pub fn groq() -> crate::providers::groq::GroqBuilder {
        crate::providers::groq::GroqBuilder::new(crate::builder::LlmBuilder::new())
    }

    // Provider convenience functions are now organized in providers::convenience module
    // This keeps the main lib.rs clean and organized
}

/// Siumai unified interface entry point
///
/// This creates a unified client that can work with multiple LLM providers
/// through a single interface. Use this when you want provider-agnostic code
/// or need to switch between providers dynamically.
///
/// # Example
/// ```rust,no_run
/// use siumai::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Build a unified client, backed by Anthropic
///     let client = Siumai::builder()
///         .anthropic()
///         .api_key("your-anthropic-key")
///         .model("claude-3-sonnet-20240229")
///         .build()
///         .await?;
///
///     // Your code uses the standard Siumai interface
///     let messages = vec![user!("What is the capital of France?")];
///     let response = client.chat(messages).await?;
///
///     // If you decide to switch to OpenAI, you only change the builder.
///     // The `.chat(request)` call remains identical.
///
///     Ok(())
/// }
/// ```
impl crate::provider::Siumai {
    /// Create a new Siumai builder for unified interface
    pub fn builder() -> crate::provider::SiumaiBuilder {
        crate::provider::SiumaiBuilder::new()
    }
}

// Re-export convenience functions
#[cfg(feature = "anthropic")]
pub use crate::builder::quick_anthropic;
#[cfg(feature = "anthropic")]
pub use crate::builder::quick_anthropic_with_model;
#[cfg(feature = "google")]
pub use crate::builder::quick_gemini;
#[cfg(feature = "google")]
pub use crate::builder::quick_gemini_with_model;
#[cfg(feature = "groq")]
pub use crate::builder::quick_groq;
#[cfg(feature = "groq")]
pub use crate::builder::quick_groq_with_model;
#[cfg(feature = "openai")]
pub use crate::builder::quick_openai;
#[cfg(feature = "openai")]
pub use crate::builder::quick_openai_with_model;

// Macros moved to a dedicated module for cleanliness
mod macros;

// Re-export provider convenience functions for easy access
#[cfg(feature = "anthropic")]
pub use providers::convenience::core::anthropic;
#[cfg(feature = "google")]
pub use providers::convenience::core::gemini;
#[cfg(feature = "groq")]
pub use providers::convenience::core::groq;
#[cfg(feature = "ollama")]
pub use providers::convenience::core::ollama;
#[cfg(feature = "openai")]
pub use providers::convenience::core::openai;
#[cfg(feature = "xai")]
pub use providers::convenience::core::xai;

// Re-export all OpenAI-compatible provider functions
#[cfg(feature = "openai")]
pub use providers::convenience::openai_compatible::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Siumai;

    #[test]
    fn test_macros() {
        // Test simple macros that return ChatMessage directly
        let user_msg = user!("Hello");
        assert_eq!(user_msg.role, MessageRole::User);

        let system_msg = system!("You are helpful");
        assert_eq!(system_msg.role, MessageRole::System);

        let assistant_msg = assistant!("I can help");
        assert_eq!(assistant_msg.role, MessageRole::Assistant);

        // Test that content is correctly set
        match user_msg.content {
            MessageContent::Text(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_provider_builder() {
        let _openai_builder = Provider::openai();
        let _anthropic_builder = Provider::anthropic();
        let _siumai_builder = Siumai::builder();
        // Basic test for builder creation
        // Placeholder test
    }
}
