//! Provider factory implementations
//!
//! Each provider implements the ProviderFactory trait to create family model objects.

#[cfg(any(
    test,
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "cohere",
    feature = "deepinfra",
    feature = "togetherai",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi",
    feature = "bedrock"
))]
use std::sync::Arc;

#[cfg(any(
    test,
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "cohere",
    feature = "deepinfra",
    feature = "togetherai",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi",
    feature = "bedrock"
))]
use crate::client::LlmClient;
#[cfg(any(
    test,
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "cohere",
    feature = "deepinfra",
    feature = "togetherai",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi",
    feature = "bedrock"
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
    feature = "cohere",
    feature = "deepinfra",
    feature = "togetherai",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi",
    feature = "bedrock"
))]
use crate::registry::entry::ProviderFactory;
#[cfg(any(
    test,
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "cohere",
    feature = "deepinfra",
    feature = "togetherai",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi",
    feature = "bedrock"
))]
use crate::traits::ProviderCapabilities;

#[cfg(any(
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "cohere",
    feature = "deepinfra",
    feature = "togetherai",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi",
    feature = "bedrock"
))]
use crate::registry::entry::BuildContext;

#[cfg(feature = "anthropic")]
mod anthropic;
#[cfg(feature = "google-vertex")]
mod anthropic_vertex;
#[cfg(feature = "azure")]
mod azure;
#[cfg(feature = "bedrock")]
mod bedrock;
#[cfg(feature = "cohere")]
mod cohere;
#[cfg(test)]
mod contract_tests;
#[cfg(feature = "deepinfra")]
mod deepinfra;
#[cfg(feature = "deepseek")]
mod deepseek;
#[cfg(feature = "openai")]
mod fireworks;
#[cfg(feature = "google")]
mod gemini;
#[cfg(feature = "google-vertex")]
mod google_vertex;
#[cfg(feature = "google-vertex")]
mod google_vertex_xai;
#[cfg(feature = "groq")]
mod groq;
#[cfg(feature = "minimaxi")]
mod minimaxi;
#[cfg(feature = "ollama")]
mod ollama;
#[cfg(feature = "openai")]
mod openai;
#[cfg(any(feature = "openai", feature = "togetherai", feature = "deepinfra"))]
mod openai_compatible;
#[cfg(test)]
mod test;
#[cfg(feature = "togetherai")]
mod togetherai;
#[cfg(feature = "google-vertex")]
mod vertex_maas;
#[cfg(feature = "xai")]
mod xai;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProviderFactory;
#[cfg(feature = "google-vertex")]
pub use anthropic_vertex::AnthropicVertexProviderFactory;
#[cfg(feature = "azure")]
pub use azure::AzureOpenAiProviderFactory;
#[cfg(feature = "bedrock")]
pub use bedrock::BedrockProviderFactory;
#[cfg(feature = "cohere")]
pub use cohere::CohereProviderFactory;
#[cfg(feature = "deepinfra")]
pub use deepinfra::DeepInfraProviderFactory;
#[cfg(feature = "deepseek")]
pub use deepseek::DeepSeekProviderFactory;
#[cfg(feature = "openai")]
pub use fireworks::FireworksProviderFactory;
#[cfg(feature = "google")]
pub use gemini::GeminiProviderFactory;
#[cfg(feature = "google-vertex")]
pub use google_vertex::GoogleVertexProviderFactory;
#[cfg(feature = "google-vertex")]
pub use google_vertex_xai::GoogleVertexXaiProviderFactory;
#[cfg(feature = "groq")]
pub use groq::GroqProviderFactory;
#[cfg(feature = "minimaxi")]
pub use minimaxi::MiniMaxiProviderFactory;
#[cfg(feature = "ollama")]
pub use ollama::OllamaProviderFactory;
#[cfg(feature = "openai")]
pub use openai::OpenAIProviderFactory;
#[cfg(any(feature = "openai", feature = "togetherai", feature = "deepinfra"))]
pub use openai_compatible::OpenAICompatibleProviderFactory;
#[cfg(test)]
pub use test::TestProviderFactory;
#[cfg(feature = "togetherai")]
pub use togetherai::TogetherAiProviderFactory;
#[cfg(feature = "google-vertex")]
pub use vertex_maas::VertexMaasProviderFactory;
#[cfg(feature = "xai")]
pub use xai::XAIProviderFactory;
