//! Static metadata for native (non OpenAI-compatible) providers.
//!
//! This module centralizes provider identifiers, human-readable names,
//! descriptions, default base URLs, and declared capabilities so that:
//! - The registry can register built-in providers from a single source.
//! - Documentation helpers (e.g., `get_supported_providers`) can reuse the
//!   same metadata without re-encoding strings or URLs.

#[allow(unused_imports)]
use crate::provider::ids;
use crate::traits::ProviderCapabilities;

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
    /// Declared provider-level capabilities.
    pub capabilities: ProviderCapabilities,
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

    // Google Vertex AI (Imagen via Vertex).
    #[cfg(feature = "google-vertex")]
    {
        // Anthropic on Vertex AI (wrapper around Anthropic served via Vertex).
        out.push(NativeProviderMetadata {
            id: ids::ANTHROPIC_VERTEX,
            name: "Anthropic on Vertex",
            description: "Anthropic Claude models served via Google Vertex AI",
            default_base_url: None,
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools(),
        });

        out.push(NativeProviderMetadata {
            id: ids::VERTEX,
            name: "Google Vertex AI",
            description: "Google Vertex AI models (e.g., Imagen) served via Vertex endpoints",
            // Requires project/location; use `base_url_for_vertex` or explicit `base_url`.
            default_base_url: None,
            capabilities: ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_embedding()
                .with_image_generation(),
        });

        out.push(NativeProviderMetadata {
            id: ids::VERTEX_MAAS,
            name: "Google Vertex MaaS",
            description:
                "Google Vertex AI MaaS partner and open models served through the OpenAI-compatible /endpoints/openapi surface",
            // Requires project/location or an explicit OpenAPI base URL.
            default_base_url: None,
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
