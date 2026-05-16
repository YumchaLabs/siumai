//! Static metadata for native (non OpenAI-compatible) providers.
//!
//! This module centralizes provider identifiers, human-readable names,
//! descriptions, default base URLs, default model policy, and declared capabilities so that:
//! - The registry can register built-in providers from a single source.
//! - Documentation helpers (e.g., `get_supported_providers`) can reuse the
//!   same metadata without re-encoding strings or URLs.

#[allow(unused_imports)]
use crate::provider::ids;
use crate::traits::ProviderCapabilities;

/// Default model behavior for public convenience construction.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeProviderDefaultModelPolicy {
    /// Use this model when public compatibility construction omits `.model(...)`.
    Default(&'static str),
    /// Reject model-less public compatibility construction with this message.
    ExplicitRequired(&'static str),
    /// No native default policy is known for this provider.
    None,
}

impl NativeProviderDefaultModelPolicy {
    pub fn default_model(self) -> Option<&'static str> {
        match self {
            Self::Default(model) => Some(model),
            Self::ExplicitRequired(_) | Self::None => None,
        }
    }

    pub fn explicit_required_message(self) -> Option<&'static str> {
        match self {
            Self::ExplicitRequired(message) => Some(message),
            Self::Default(_) | Self::None => None,
        }
    }
}

/// Static metadata for a native provider.
///
/// OpenAI-compatible providers are configured via `openai_compatible::config`
/// and are not covered by this type.
#[derive(Debug, Clone)]
pub struct NativeProviderMetadata {
    /// Canonical provider identifier (e.g., `"openai"`, `"anthropic"`).
    pub id: &'static str,
    /// Human-readable provider name.
    pub name: &'static str,
    /// Short description suitable for docs and introspection.
    pub description: &'static str,
    /// Default base URL used when no override is provided.
    ///
    /// `None` means the provider has no single canonical HTTP endpoint
    /// (for example, wrappers like `"anthropic-vertex"`).
    pub default_base_url: Option<&'static str>,
    /// Public compatibility default model policy.
    pub default_model_policy: NativeProviderDefaultModelPolicy,
    /// Declared provider-level capabilities.
    pub capabilities: ProviderCapabilities,
}

/// Return the native default model policy for an enabled provider id.
pub fn native_provider_default_model_policy(
    provider_id: &str,
) -> Option<NativeProviderDefaultModelPolicy> {
    native_providers_metadata()
        .into_iter()
        .find(|meta| meta.id == provider_id)
        .map(|meta| meta.default_model_policy)
}

/// Return metadata for all native providers enabled in this build.
///
/// Most entries are registered into the built-in provider catalog by default.
/// A small set may be **metadata-only** (feature-gated, but not yet backed by
/// built-in factories). Those should be excluded from the default catalog until
/// their factories exist.
#[allow(clippy::vec_init_then_push)]
pub fn native_providers_metadata() -> Vec<NativeProviderMetadata> {
    #[allow(unused_mut)]
    let mut out = Vec::new();

    // OpenAI
    #[cfg(feature = "openai")]
    out.push(NativeProviderMetadata {
        id: "openai",
        name: "OpenAI",
        description: "OpenAI GPT models including GPT-4, GPT-3.5, and specialized models",
        default_base_url: Some("https://api.openai.com/v1"),
        default_model_policy: NativeProviderDefaultModelPolicy::Default(
            siumai_provider_openai::providers::openai::model_constants::gpt_4o::GPT_4O,
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_completion()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_audio()
            .with_file_management()
            .with_custom_feature("skills", true)
            .with_image_generation(),
    });

    // Azure OpenAI (OpenAI-compatible endpoints hosted on Azure).
    #[cfg(feature = "azure")]
    out.push(NativeProviderMetadata {
        id: "azure",
        name: "Azure OpenAI",
        description: "Azure OpenAI deployments via OpenAI-compatible endpoints",
        // Requires resource name or explicit base_url; see AZURE_RESOURCE_NAME.
        default_base_url: None,
        default_model_policy: NativeProviderDefaultModelPolicy::ExplicitRequired(
            "Azure OpenAI requires an explicit model (deployment id)",
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_completion()
            .with_streaming()
            .with_tools()
            .with_embedding()
            .with_audio()
            .with_file_management()
            .with_image_generation(),
    });

    // Anthropic
    #[cfg(feature = "anthropic")]
    {
        out.push(NativeProviderMetadata {
            id: "anthropic",
            name: "Anthropic",
            description: "Anthropic Claude models with advanced reasoning capabilities",
            default_base_url: Some("https://api.anthropic.com"),
            default_model_policy: NativeProviderDefaultModelPolicy::Default(
                siumai_provider_anthropic::providers::anthropic::model_constants::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20241022,
            ),
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_custom_feature("skills", true)
                .with_custom_feature("thinking", true),
        });
    }

    // Google Gemini
    #[cfg(feature = "google")]
    out.push(NativeProviderMetadata {
        id: ids::GEMINI,
        name: "Google Gemini",
        description: "Google Gemini models with multimodal capabilities and code execution",
        default_base_url: Some("https://generativelanguage.googleapis.com/v1beta"),
        default_model_policy: NativeProviderDefaultModelPolicy::Default(
            siumai_provider_gemini::providers::gemini::model_constants::gemini_2_5_flash::GEMINI_2_5_FLASH,
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_file_management()
            .with_image_generation()
            .with_custom_feature("thinking", true)
            .with_custom_feature("video", true),
    });

    // Google Vertex AI (Gemini + Imagen + Veo via Vertex).
    #[cfg(feature = "google-vertex")]
    {
        // Anthropic on Vertex AI (wrapper around Anthropic served via Vertex).
        out.push(NativeProviderMetadata {
            id: ids::ANTHROPIC_VERTEX,
            name: "Anthropic on Vertex",
            description: "Anthropic Claude models served via Google Vertex AI",
            default_base_url: None,
            default_model_policy: NativeProviderDefaultModelPolicy::ExplicitRequired(
                "Anthropic on Vertex requires an explicit model id",
            ),
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools(),
        });

        out.push(NativeProviderMetadata {
            id: ids::VERTEX,
            name: "Google Vertex AI",
            description: "Google Vertex AI models (e.g., Gemini, Imagen, Veo) served via Vertex endpoints",
            // Requires project/location; use `base_url_for_vertex` or explicit `base_url`.
            default_base_url: None,
            default_model_policy: NativeProviderDefaultModelPolicy::ExplicitRequired(
                "Google Vertex requires an explicit model id",
            ),
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_embedding()
                .with_image_generation()
                .with_custom_feature("video", true),
        });

        out.push(NativeProviderMetadata {
            id: ids::VERTEX_MAAS,
            name: "Google Vertex MaaS",
            description:
                "Google Vertex AI MaaS partner and open models served through the OpenAI-compatible /endpoints/openapi surface",
            // Requires project/location or an explicit OpenAPI base URL.
            default_base_url: None,
            default_model_policy: NativeProviderDefaultModelPolicy::ExplicitRequired(
                "Google Vertex MaaS requires an explicit model id",
            ),
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_completion()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_embedding(),
        });
    }

    // Groq
    #[cfg(feature = "groq")]
    out.push(NativeProviderMetadata {
        id: "groq",
        name: "Groq",
        description: "Groq models with ultra-fast inference",
        default_base_url: Some("https://api.groq.com/openai/v1"),
        default_model_policy: NativeProviderDefaultModelPolicy::Default(
            siumai_provider_openai_compatible::providers::openai_compatible::providers::models::groq::LLAMA_3_1_70B,
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_audio(),
    });

    // DeepSeek
    #[cfg(feature = "deepseek")]
    out.push(NativeProviderMetadata {
        id: ids::DEEPSEEK,
        name: "DeepSeek",
        description: "DeepSeek chat and reasoning models exposed through a dedicated registry factory",
        default_base_url: Some("https://api.deepseek.com"),
        default_model_policy: NativeProviderDefaultModelPolicy::Default(
            siumai_provider_deepseek::providers::deepseek::models::CHAT,
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("thinking", true),
    });

    // DeepInfra
    #[cfg(feature = "deepinfra")]
    out.push(NativeProviderMetadata {
        id: ids::DEEPINFRA,
        name: "DeepInfra",
        description: "DeepInfra unified provider surface via OpenAI-compatible text endpoints plus provider-owned image generation and edit routes",
        default_base_url: Some("https://api.deepinfra.com/v1"),
        default_model_policy: NativeProviderDefaultModelPolicy::Default(
            siumai_provider_openai_compatible::providers::openai_compatible::providers::models::deepinfra::CHAT,
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_completion()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_image_generation(),
    });

    // Fireworks
    #[cfg(feature = "openai")]
    out.push(NativeProviderMetadata {
        id: ids::FIREWORKS,
        name: "Fireworks AI",
        description: "Fireworks unified provider surface via OpenAI-compatible chat, completion, embedding, and transcription endpoints plus provider-owned image generation and edit routes",
        default_base_url: Some("https://api.fireworks.ai/inference/v1"),
        default_model_policy: NativeProviderDefaultModelPolicy::None,
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_completion()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_image_generation()
            .with_transcription(),
    });

    // xAI
    #[cfg(feature = "xai")]
    out.push(NativeProviderMetadata {
        id: "xai",
        name: "xAI",
        description: "xAI Grok models with reasoning, vision, provider-owned image generation, speech, and video task APIs",
        default_base_url: Some("https://api.x.ai/v1"),
        default_model_policy: NativeProviderDefaultModelPolicy::Default(
            siumai_provider_xai::providers::xai::models::legacy::GROK_BETA,
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_image_generation()
            .with_speech()
            .with_custom_feature("video", true),
    });

    // Ollama
    #[cfg(feature = "ollama")]
    out.push(NativeProviderMetadata {
        id: "ollama",
        name: "Ollama",
        description: "Local Ollama models with full control and privacy",
        default_base_url: Some("http://localhost:11434"),
        default_model_policy: NativeProviderDefaultModelPolicy::Default(
            siumai_provider_ollama::providers::ollama::model_constants::llama_3_2::LLAMA_3_2,
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding(),
    });

    // MiniMaxi
    #[cfg(feature = "minimaxi")]
    out.push(NativeProviderMetadata {
        id: "minimaxi",
        name: "MiniMaxi",
        description: "MiniMaxi models with multi-modal capabilities (text, speech, video, music)",
        default_base_url: Some("https://api.minimaxi.com/anthropic"),
        default_model_policy: NativeProviderDefaultModelPolicy::Default(
            siumai_provider_minimaxi::providers::minimaxi::MinimaxiConfig::DEFAULT_MODEL,
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_speech()
            .with_file_management()
            .with_custom_feature("video", true)
            .with_image_generation()
            .with_custom_feature("music", true),
    });

    // Cohere
    #[cfg(feature = "cohere")]
    out.push(NativeProviderMetadata {
        id: "cohere",
        name: "Cohere",
        description: "Cohere native chat, embedding, and reranking via the v2 API",
        default_base_url: Some("https://api.cohere.com/v2"),
        default_model_policy: NativeProviderDefaultModelPolicy::ExplicitRequired(
            "Cohere requires an explicit model id",
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_embedding()
            .with_rerank(),
    });

    // TogetherAI (OpenAI-compatible text/audio/image + native rerank)
    #[cfg(feature = "togetherai")]
    out.push(NativeProviderMetadata {
        id: "togetherai",
        name: "TogetherAI",
        description:
            "TogetherAI unified provider surface via OpenAI-compatible chat/completion/audio/image endpoints plus native /v1/rerank",
        default_base_url: Some("https://api.together.xyz/v1"),
        default_model_policy: NativeProviderDefaultModelPolicy::Default(
            siumai_provider_togetherai::providers::togetherai::models::CHAT,
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_completion()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_image_generation()
            .with_speech()
            .with_transcription()
            .with_audio()
            .with_rerank(),
    });

    // Amazon Bedrock (Converse + embedding + image + Rerank)
    #[cfg(feature = "bedrock")]
    out.push(NativeProviderMetadata {
        id: "bedrock",
        name: "Amazon Bedrock",
        description:
            "Amazon Bedrock models via Converse, invoke-based embedding/image APIs, and Agent Runtime reranking",
        // Requires region + service selection (bedrock-runtime vs bedrock-agent-runtime).
        default_base_url: None,
        default_model_policy: NativeProviderDefaultModelPolicy::ExplicitRequired(
            "Amazon Bedrock requires an explicit model id",
        ),
        capabilities: ProviderCapabilities::new()
            .with_chat()
            .with_embedding()
            .with_image_generation()
            .with_streaming()
            .with_tools()
            .with_rerank(),
    });

    out
}
