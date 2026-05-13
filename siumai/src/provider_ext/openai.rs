#[cfg(feature = "openai-websocket")]
pub use siumai_provider_openai::providers::openai::OpenAiIncrementalWebSocketSession;
#[cfg(feature = "openai-websocket")]
pub use siumai_provider_openai::providers::openai::OpenAiWebSocketRecoveryConfig;
#[cfg(feature = "openai-websocket")]
pub use siumai_provider_openai::providers::openai::OpenAiWebSocketSession;
#[cfg(feature = "openai-websocket")]
pub use siumai_provider_openai::providers::openai::OpenAiWebSocketTransport;
pub use siumai_provider_openai::providers::openai::{
    OpenAIProviderSettings, OpenAiBuilder, OpenAiClient, OpenAiConfig, VERSION,
};

/// Create the OpenAI provider builder.
pub fn openai() -> OpenAiBuilder {
    crate::Provider::openai()
}

/// Create the OpenAI provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createOpenAI()`.
pub fn create_openai() -> OpenAiBuilder {
    openai()
}

/// Provider tool factories that return `Tool` directly (Vercel-aligned).
///
/// This mirrors the Vercel AI SDK `{ type: "provider", id, name, args }` convention.
pub mod tools {
    pub use crate::tools::openai::*;
}

/// Provider-executed tool builders (typed args).
///
/// Prefer this module when you need provider-specific configuration helpers and want `.build()`.
pub mod hosted_tools {
    pub use crate::hosted_tools::openai::*;
}

/// Compatibility alias for older imports.
///
/// Prefer `siumai::providers::openai::tools`.
pub mod provider_tools {
    pub use crate::tools::openai::*;
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["openai"]`).
pub mod metadata {
    pub use siumai_provider_openai::provider_metadata::openai::{
        OpenAiChatResponseExt, OpenAiContentPartExt, OpenAiContentPartMetadata, OpenAiMetadata,
        OpenAiSource, OpenAiSourceExt, OpenAiSourceMetadata,
    };
}
pub use metadata::{
    OpenAiChatResponseExt, OpenAiContentPartExt, OpenAiContentPartMetadata, OpenAiMetadata,
    OpenAiSource, OpenAiSourceExt, OpenAiSourceMetadata,
};

/// Typed provider options (`provider_options_map["openai"]`).
pub mod options {
    pub use siumai_provider_openai::provider_options::openai::{
        ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
        ChatCompletionModalities, InputAudio, InputAudioFormat, OpenAIContextManagementConfig,
        OpenAIContextManagementType, OpenAIEmbeddingModelOptions, OpenAIFilesOptions,
        OpenAILanguageModelChatOptions, OpenAILanguageModelCompletionOptions,
        OpenAILanguageModelResponsesOptions, OpenAISpeechModelOptions,
        OpenAITranscriptionModelOptions, OpenAiOptions, OpenAiWebSearchOptions, PredictionContent,
        PredictionContentData, ReasoningEffort, ResponsesApiConfig, ServiceTier, SystemMessageMode,
        TextVerbosity, Truncation, UserLocationWrapper, WebSearchLocation,
    };
    #[allow(deprecated)]
    pub use siumai_provider_openai::provider_options::openai::{
        OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions,
    };
    pub use siumai_provider_openai::providers::openai::ext::OpenAiChatRequestExt;
    pub use siumai_provider_openai::providers::openai::ext::audio_options::{
        OpenAiSttOptions, OpenAiSttRequestExt, OpenAiTtsOptions,
    };
    pub use siumai_provider_openai::providers::openai::types::{
        OpenAiEmbeddingOptions, OpenAiEmbeddingRequestExt,
    };
}
pub use options::{
    ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
    ChatCompletionModalities, InputAudio, InputAudioFormat, OpenAIContextManagementConfig,
    OpenAIContextManagementType, OpenAIEmbeddingModelOptions, OpenAIFilesOptions,
    OpenAILanguageModelChatOptions, OpenAILanguageModelCompletionOptions,
    OpenAILanguageModelResponsesOptions, OpenAISpeechModelOptions, OpenAITranscriptionModelOptions,
    OpenAiChatRequestExt, OpenAiEmbeddingOptions, OpenAiEmbeddingRequestExt, OpenAiOptions,
    OpenAiSttOptions, OpenAiSttRequestExt, OpenAiTtsOptions, OpenAiWebSearchOptions,
    PredictionContent, PredictionContentData, ReasoningEffort, ResponsesApiConfig, ServiceTier,
    SystemMessageMode, TextVerbosity, Truncation, UserLocationWrapper, WebSearchLocation,
};
#[allow(deprecated)]
pub use options::{OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions};

/// Non-unified OpenAI extension APIs (streaming helpers, moderation/files resources, etc.).
pub mod ext {
    pub use siumai_provider_openai::providers::openai::ext::{
        moderation, responses, speech_streaming, transcription_streaming,
    };
    pub use siumai_provider_openai::providers::openai::responses::OpenAiResponsesEventConverter;
}

/// Provider-specific resources not covered by the unified families.
pub mod resources {
    pub use siumai_provider_openai::providers::openai::{
        OpenAiFiles, OpenAiModels, OpenAiModeration, OpenAiRerank, OpenAiSkills,
    };
}

/// Legacy OpenAI parameter structs (client-level defaults).
///
/// Prefer request-level provider options (`OpenAiOptions`) for new code.
pub mod legacy_params {
    pub use siumai_provider_openai::params::openai::{
        FunctionChoice, OpenAiParams, OpenAiParamsBuilder, ResponseFormat, ToolChoice,
    };
}
