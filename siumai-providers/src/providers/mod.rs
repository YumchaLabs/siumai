//! Provider module.
//!
//! Contains provider implementations and their submodules.

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "google")]
pub mod gemini;
#[cfg(feature = "groq")]
pub mod groq;
#[cfg(feature = "minimaxi")]
pub mod minimaxi;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "openai")]
pub mod openai_compatible;
#[cfg(feature = "xai")]
pub mod xai;

/// Static metadata helpers for native providers.
pub mod metadata;

// Provider builder methods and convenience functions
pub mod builders;
pub mod convenience;

// Re-export main types
#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicClient;
#[cfg(feature = "google")]
pub use gemini::GeminiClient;
#[cfg(feature = "anthropic")]
pub mod anthropic_vertex;
#[cfg(feature = "groq")]
pub use groq::GroqClient;
#[cfg(feature = "minimaxi")]
pub use minimaxi::MinimaxiClient;
#[cfg(feature = "ollama")]
pub use ollama::OllamaClient;
#[cfg(feature = "openai")]
pub use openai::OpenAiClient;
#[cfg(feature = "xai")]
pub use xai::XaiClient;
