//! Provider factory implementations
//!
//! Each provider implements the ProviderFactory trait to create clients.

#[cfg(any(
    test,
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
use std::sync::Arc;

#[cfg(any(
    test,
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
use crate::client::LlmClient;
#[cfg(any(
    test,
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
use crate::error::LlmError;

#[allow(unused_imports)]
use crate::execution::http::client::build_http_client_from_config;

#[cfg(any(
    test,
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
use crate::registry::entry::ProviderFactory;
#[cfg(any(
    test,
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
use crate::traits::ProviderCapabilities;

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
use crate::registry::entry::BuildContext;

#[cfg(feature = "anthropic")]
mod anthropic;
#[cfg(feature = "google-vertex")]
mod anthropic_vertex;
#[cfg(feature = "azure")]
mod azure;
#[cfg(test)]
mod contract_tests;
#[cfg(feature = "openai")]
mod deepseek;
#[cfg(feature = "google")]
mod gemini;
#[cfg(feature = "google-vertex")]
mod google_vertex;
#[cfg(feature = "groq")]
mod groq;
#[cfg(feature = "minimaxi")]
mod minimaxi;
#[cfg(feature = "ollama")]
mod ollama;
#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
mod openai_compatible;
#[cfg(feature = "openai")]
mod openrouter;
#[cfg(test)]
mod test;
#[cfg(feature = "xai")]
mod xai;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProviderFactory;
#[cfg(feature = "google-vertex")]
pub use anthropic_vertex::AnthropicVertexProviderFactory;
#[cfg(feature = "azure")]
pub use azure::AzureOpenAiProviderFactory;
#[cfg(feature = "openai")]
pub use deepseek::DeepSeekProviderFactory;
#[cfg(feature = "google")]
pub use gemini::GeminiProviderFactory;
#[cfg(feature = "google-vertex")]
pub use google_vertex::GoogleVertexProviderFactory;
#[cfg(feature = "groq")]
pub use groq::GroqProviderFactory;
#[cfg(feature = "minimaxi")]
pub use minimaxi::MiniMaxiProviderFactory;
#[cfg(feature = "ollama")]
pub use ollama::OllamaProviderFactory;
#[cfg(feature = "openai")]
pub use openai::OpenAIProviderFactory;
#[cfg(feature = "openai")]
pub use openai_compatible::OpenAICompatibleProviderFactory;
#[cfg(feature = "openai")]
pub use openrouter::OpenRouterProviderFactory;
#[cfg(test)]
pub use test::TestProviderFactory;
#[cfg(feature = "xai")]
pub use xai::XAIProviderFactory;
