/// Legacy provider-specific builder convenience entry type.
///
/// Prefer `siumai::prelude::unified::registry::*` or config-first provider clients for stable
/// construction. `Provider` remains a builder-style compatibility path for migration-oriented code.
///
/// # Example
/// ```rust,no_run
/// use siumai::prelude::{compat::Provider, unified::*};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let openai_client = Provider::openai()
///         .api_key("your-openai-key")
///         .model("gpt-4")
///         .build()
///         .await?;
///
///     let messages = vec![user!("Hello!")];
///     let response = openai_client.chat(messages).await?;
///     println!("OpenAI says: {}", response.text().unwrap_or_default());
///
///     Ok(())
/// }
/// ```
pub struct Provider;

use siumai_core::builder::BuilderBase;

impl Provider {
    /// Create an `OpenAI` client builder
    #[cfg(feature = "openai")]
    pub fn openai() -> siumai_provider_openai::providers::openai::OpenAiBuilder {
        siumai_provider_openai::providers::openai::OpenAiBuilder::new(BuilderBase::default())
    }

    /// Create an explicit `OpenAI Responses` client builder.
    #[cfg(feature = "openai")]
    pub fn openai_responses() -> siumai_provider_openai::providers::openai::OpenAiBuilder {
        Self::openai().use_responses_api(true)
    }

    /// Create an explicit `OpenAI Chat Completions` client builder.
    #[cfg(feature = "openai")]
    pub fn openai_chat() -> siumai_provider_openai::providers::openai::OpenAiBuilder {
        Self::openai().use_responses_api(false)
    }

    /// Create an `Azure OpenAI` unified builder (Responses API by default).
    #[cfg(feature = "azure")]
    pub fn azure() -> siumai_registry::provider::SiumaiBuilder {
        siumai_registry::provider::SiumaiBuilder::new().azure()
    }

    /// Create an `Azure OpenAI Chat Completions` unified builder.
    #[cfg(feature = "azure")]
    pub fn azure_chat() -> siumai_registry::provider::SiumaiBuilder {
        siumai_registry::provider::SiumaiBuilder::new().azure_chat()
    }

    /// Create an Anthropic client builder
    #[cfg(feature = "anthropic")]
    pub fn anthropic() -> siumai_provider_anthropic::providers::anthropic::AnthropicBuilder {
        siumai_provider_anthropic::providers::anthropic::AnthropicBuilder::new(
            BuilderBase::default(),
        )
    }

    /// Create an Amazon Bedrock client builder
    #[cfg(feature = "bedrock")]
    pub fn bedrock() -> siumai_provider_amazon_bedrock::providers::bedrock::BedrockBuilder {
        siumai_provider_amazon_bedrock::providers::bedrock::BedrockBuilder::new(
            BuilderBase::default(),
        )
    }

    /// Create a Cohere client builder
    #[cfg(feature = "cohere")]
    pub fn cohere() -> siumai_provider_cohere::providers::cohere::CohereBuilder {
        siumai_provider_cohere::providers::cohere::CohereBuilder::new(BuilderBase::default())
    }

    /// Create a Gemini client builder
    #[cfg(feature = "google")]
    pub fn gemini() -> siumai_provider_gemini::providers::gemini::GeminiBuilder {
        siumai_provider_gemini::providers::gemini::GeminiBuilder::new(BuilderBase::default())
    }

    /// Create a Google client builder alias (AI SDK package-aligned).
    #[cfg(feature = "google")]
    pub fn google() -> siumai_provider_gemini::providers::gemini::GeminiBuilder {
        Self::gemini().name("google.generative-ai")
    }

    /// Create a TogetherAI unified builder.
    ///
    /// This aligns with the AI SDK-style provider surface:
    /// chat/completion/embedding/speech/transcription route through the shared
    /// OpenAI-compatible TogetherAI path, while image and rerank use provider-owned TogetherAI
    /// implementations.
    #[cfg(feature = "togetherai")]
    pub fn togetherai() -> siumai_registry::provider::SiumaiBuilder {
        siumai_registry::provider::SiumaiBuilder::new().togetherai()
    }

    /// Create a DeepInfra unified builder.
    ///
    /// This aligns with the AI SDK-style provider surface:
    /// text/completion/embedding reuse the shared OpenAI-compatible runtime, while image
    /// generation and editing use the provider-owned DeepInfra `/inference` and
    /// `/openai/images/edits` routes.
    #[cfg(feature = "deepinfra")]
    pub fn deepinfra() -> siumai_registry::provider::SiumaiBuilder {
        siumai_registry::provider::SiumaiBuilder::new().deepinfra()
    }

    /// Create a Mistral unified builder.
    ///
    /// This mirrors the AI SDK `mistral` provider package surface while continuing to reuse the
    /// shared OpenAI-compatible runtime internally.
    #[cfg(feature = "openai")]
    pub fn mistral() -> siumai_registry::provider::SiumaiBuilder {
        siumai_registry::provider::SiumaiBuilder::new().mistral()
    }

    /// Create a Fireworks unified builder.
    ///
    /// This mirrors the AI SDK `fireworks` provider package surface: chat/completion/embedding/
    /// transcription use the shared OpenAI-compatible runtime, while image generation/edit use the
    /// provider-owned Fireworks workflow routes.
    #[cfg(feature = "openai")]
    pub fn fireworks() -> siumai_registry::provider::SiumaiBuilder {
        siumai_registry::provider::SiumaiBuilder::new().fireworks()
    }

    /// Create a Perplexity unified builder.
    ///
    /// This mirrors the AI SDK `perplexity` provider package surface while continuing to reuse the
    /// shared OpenAI-compatible runtime internally.
    #[cfg(feature = "openai")]
    pub fn perplexity() -> siumai_registry::provider::SiumaiBuilder {
        siumai_registry::provider::SiumaiBuilder::new().perplexity()
    }

    /// Create a MoonshotAI unified builder.
    ///
    /// This mirrors the AI SDK `moonshotai` provider package surface while continuing to reuse the
    /// shared OpenAI-compatible runtime internally.
    #[cfg(feature = "openai")]
    pub fn moonshotai() -> siumai_registry::provider::SiumaiBuilder {
        siumai_registry::provider::SiumaiBuilder::new().moonshotai()
    }

    /// Create an Ollama client builder
    #[cfg(feature = "ollama")]
    pub fn ollama() -> siumai_provider_ollama::providers::ollama::OllamaBuilder {
        siumai_provider_ollama::providers::ollama::OllamaBuilder::new(BuilderBase::default())
    }

    /// Create an xAI client builder
    #[cfg(feature = "xai")]
    pub fn xai() -> siumai_provider_xai::providers::xai::XaiBuilder {
        siumai_provider_xai::providers::xai::XaiBuilder::new(BuilderBase::default())
    }

    /// Create a Groq client builder
    #[cfg(feature = "groq")]
    pub fn groq() -> siumai_provider_groq::providers::groq::GroqBuilder {
        siumai_provider_groq::providers::groq::GroqBuilder::new(BuilderBase::default())
    }

    /// Create a MiniMaxi client builder
    #[cfg(feature = "minimaxi")]
    pub fn minimaxi() -> siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder {
        siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(BuilderBase::default())
    }

    /// Create a Google Vertex client builder
    #[cfg(feature = "google-vertex")]
    pub fn google_vertex() -> siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder
    {
        siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
            BuilderBase::default(),
        )
    }

    /// Create a Google Vertex client builder
    #[cfg(feature = "google-vertex")]
    pub fn vertex() -> siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder {
        Self::google_vertex()
    }

    /// Create a Google Vertex MaaS unified builder.
    ///
    /// This aligns with the AI SDK `vertexMaas` surface:
    /// chat/completion/embedding route through the shared OpenAI-compatible runtime on
    /// Vertex's `/endpoints/openapi` base URL, authenticated with Google-style Bearer tokens.
    #[cfg(feature = "google-vertex")]
    pub fn vertex_maas() -> siumai_registry::provider::SiumaiBuilder {
        siumai_registry::provider::SiumaiBuilder::new().vertex_maas()
    }

    /// Create an Anthropic on Vertex client builder
    #[cfg(feature = "google-vertex")]
    pub fn anthropic_vertex()
    -> siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder {
        siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder::new(
            BuilderBase::default(),
        )
    }

    /// AI SDK package-aligned alias for `anthropic_vertex()`.
    #[cfg(feature = "google-vertex")]
    pub fn vertex_anthropic()
    -> siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder {
        Self::anthropic_vertex()
    }

    /// Create a DeepSeek client builder
    #[cfg(feature = "deepseek")]
    pub fn deepseek() -> siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder {
        siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(BuilderBase::default())
    }
}
