//! Provider id constants and explicit variant parsing.
//!
//! Centralizing ids avoids "stringly-typed" routing scattered across multiple layers.

/// OpenAI provider id (Responses API by default).
pub(crate) const OPENAI: &str = "openai";
/// OpenAI variant that forces Chat Completions API.
pub(crate) const OPENAI_CHAT: &str = "openai-chat";
/// OpenAI variant that forces Responses API (explicit).
pub(crate) const OPENAI_RESPONSES: &str = "openai-responses";

/// Azure OpenAI provider id (Responses API by default).
pub(crate) const AZURE: &str = "azure";
/// Azure OpenAI variant that forces Chat Completions API.
pub(crate) const AZURE_CHAT: &str = "azure-chat";

pub(crate) const ANTHROPIC: &str = "anthropic";
pub(crate) const ANTHROPIC_VERTEX: &str = "anthropic-vertex";

pub(crate) const GEMINI: &str = "gemini";
pub(crate) const VERTEX: &str = "vertex";

pub(crate) const OLLAMA: &str = "ollama";
pub(crate) const XAI: &str = "xai";
pub(crate) const GROQ: &str = "groq";
pub(crate) const MINIMAXI: &str = "minimaxi";

/// Alias id for registry convenience (canonical id is `vertex`).
pub(crate) const GOOGLE_VERTEX_ALIAS: &str = "google-vertex";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinProviderId {
    OpenAi,
    OpenAiChat,
    OpenAiResponses,
    Azure,
    AzureChat,
    Anthropic,
    AnthropicVertex,
    Gemini,
    Vertex,
    Ollama,
    Xai,
    Groq,
    MiniMaxi,
}

impl BuiltinProviderId {
    pub(crate) fn parse(provider_id: &str) -> Option<Self> {
        match provider_id {
            OPENAI => Some(Self::OpenAi),
            OPENAI_CHAT => Some(Self::OpenAiChat),
            OPENAI_RESPONSES => Some(Self::OpenAiResponses),
            AZURE => Some(Self::Azure),
            AZURE_CHAT => Some(Self::AzureChat),
            ANTHROPIC => Some(Self::Anthropic),
            ANTHROPIC_VERTEX => Some(Self::AnthropicVertex),
            GEMINI => Some(Self::Gemini),
            VERTEX => Some(Self::Vertex),
            OLLAMA => Some(Self::Ollama),
            XAI => Some(Self::Xai),
            GROQ => Some(Self::Groq),
            MINIMAXI => Some(Self::MiniMaxi),
            _ => None,
        }
    }
}

pub(crate) fn is_openai_family(provider_id: &str) -> bool {
    matches!(
        BuiltinProviderId::parse(provider_id),
        Some(
            BuiltinProviderId::OpenAi
                | BuiltinProviderId::OpenAiChat
                | BuiltinProviderId::OpenAiResponses
        )
    )
}

pub(crate) fn is_azure_family(provider_id: &str) -> bool {
    matches!(
        BuiltinProviderId::parse(provider_id),
        Some(BuiltinProviderId::Azure | BuiltinProviderId::AzureChat)
    )
}

#[cfg(feature = "openai-websocket")]
pub(crate) fn is_openai_responses_variant(provider_id: &str) -> bool {
    matches!(
        BuiltinProviderId::parse(provider_id),
        Some(BuiltinProviderId::OpenAi | BuiltinProviderId::OpenAiResponses)
    )
}
