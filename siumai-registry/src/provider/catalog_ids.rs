//! Provider-id-first catalog classification for built-in provider metadata.
//!
//! This module is intentionally registry-owned. It keeps catalog routing on open provider ids
//! instead of using the public compatibility `ProviderType` enum as the primary switch.

use crate::provider::ids;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum CatalogProviderId {
    OpenAi,
    Azure,
    Anthropic,
    Gemini,
    Vertex,
    AnthropicVertex,
    VertexMaas,
    GoogleVertexXai,
    Ollama,
    DeepSeek,
    DeepInfra,
    Cohere,
    TogetherAi,
    Bedrock,
    Gateway,
    Mistral,
    Fireworks,
    Perplexity,
    Xai,
    Groq,
    MiniMaxi,
    Custom,
}

impl CatalogProviderId {
    pub(crate) fn parse(provider_id: &str) -> Self {
        match provider_id {
            ids::OPENAI | ids::OPENAI_CHAT | ids::OPENAI_RESPONSES => Self::OpenAi,
            ids::AZURE | ids::AZURE_CHAT => Self::Azure,
            ids::ANTHROPIC => Self::Anthropic,
            ids::GEMINI => Self::Gemini,
            ids::VERTEX | ids::GOOGLE_VERTEX_ALIAS => Self::Vertex,
            ids::ANTHROPIC_VERTEX | "google-vertex-anthropic" => Self::AnthropicVertex,
            ids::VERTEX_MAAS
            | ids::GOOGLE_VERTEX_MAAS_ALIAS
            | ids::GOOGLE_VERTEX_MAAS_DOTTED_ALIAS
            | "vertexMaas" => Self::VertexMaas,
            ids::GOOGLE_VERTEX_XAI
            | ids::GOOGLE_VERTEX_XAI_DOTTED_ALIAS
            | ids::GOOGLE_VERTEX_XAI_SHORT_ALIAS => Self::GoogleVertexXai,
            ids::OLLAMA => Self::Ollama,
            ids::DEEPSEEK => Self::DeepSeek,
            ids::DEEPINFRA => Self::DeepInfra,
            ids::COHERE => Self::Cohere,
            ids::TOGETHERAI => Self::TogetherAi,
            ids::GATEWAY => Self::Gateway,
            ids::BEDROCK => Self::Bedrock,
            "mistral" => Self::Mistral,
            ids::FIREWORKS => Self::Fireworks,
            "perplexity" => Self::Perplexity,
            ids::XAI => Self::Xai,
            ids::GROQ => Self::Groq,
            ids::MINIMAXI => Self::MiniMaxi,
            _ => Self::Custom,
        }
    }

    pub(crate) fn canonical_provider_id(self) -> Option<&'static str> {
        match self {
            Self::OpenAi => Some(ids::OPENAI),
            Self::Azure => Some(ids::AZURE),
            Self::Anthropic => Some(ids::ANTHROPIC),
            Self::Gemini => Some(ids::GEMINI),
            Self::Vertex => Some(ids::VERTEX),
            Self::AnthropicVertex => Some(ids::ANTHROPIC_VERTEX),
            Self::VertexMaas => Some(ids::VERTEX_MAAS),
            Self::GoogleVertexXai => Some(ids::GOOGLE_VERTEX_XAI),
            Self::Ollama => Some(ids::OLLAMA),
            Self::DeepSeek => Some(ids::DEEPSEEK),
            Self::DeepInfra => Some(ids::DEEPINFRA),
            Self::Cohere => Some(ids::COHERE),
            Self::Gateway => Some(ids::GATEWAY),
            Self::TogetherAi => Some(ids::TOGETHERAI),
            Self::Bedrock => Some(ids::BEDROCK),
            Self::Mistral => Some("mistral"),
            Self::Fireworks => Some(ids::FIREWORKS),
            Self::Perplexity => Some("perplexity"),
            Self::Xai => Some(ids::XAI),
            Self::Groq => Some(ids::GROQ),
            Self::MiniMaxi => Some(ids::MINIMAXI),
            Self::Custom => None,
        }
    }
}
