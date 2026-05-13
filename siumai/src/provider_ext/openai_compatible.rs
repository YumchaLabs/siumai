pub use siumai_provider_openai_compatible::providers::openai_compatible::{
    AlibabaProviderSettings, ConfigurableAdapter, DeepInfraProviderSettings, DeepSeekChatModelId,
    DeepSeekClient, DeepSeekConfig, DeepSeekProviderSettings, FireworksProviderSettings,
    GoogleVertexMaasProviderSettings, GroqChatModelId, GroqClient, GroqConfig,
    GroqProviderSettings, GroqTranscriptionModelId, MetadataExtractor, MistralProviderSettings,
    MoonshotAIProviderSettings, OPENAI_COMPATIBLE_VERSION as VERSION, OpenAICompatibleChatModelId,
    OpenAICompatibleClient, OpenAICompatibleCompletionModelId, OpenAICompatibleConfig,
    OpenAICompatibleEmbeddingModelId, OpenAICompatibleErrorData, OpenAICompatibleImageModelId,
    OpenAICompatibleProviderSettings, OpenAICompatibleRequestSettings, OpenAiCompatibleChatModelId,
    OpenAiCompatibleClient, OpenAiCompatibleCompletionModelId, OpenAiCompatibleConfig,
    OpenAiCompatibleEmbeddingModelId, OpenAiCompatibleErrorData, OpenAiCompatibleImageModelId,
    OpenAiCompatibleRequestSettings, PerplexityProviderSettings, ProviderAdapter,
    ProviderCompatibility, ProviderConfig, ProviderErrorStructure, RequestBodyTransformer,
    ResponseMetadataExtractor, TogetherAIChatModelId, TogetherAIClient,
    TogetherAICompletionModelId, TogetherAIConfig, TogetherAIEmbeddingModelId,
    TogetherAIImageModelId, TogetherAIProviderSettings, TogetherAIRerankingModelId, XaiChatModelId,
    XaiClient, XaiConfig, XaiImageModelId, XaiProviderSettings, XaiResponsesModelId,
    XaiVideoModelId, deepinfra, deepseek, fireworks, generic_provider_config, get_provider_config,
    groq, list_provider_ids, moonshot, moonshotai, openrouter, provider_supports_capability,
    siliconflow, together, togetherai, vertex_maas, xai,
};

/// Typed generic OpenAI-compatible provider options (`provider_options_map["openaiCompatible"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_openai_compatible::provider_options::{
        OpenAICompatibleCompletionProviderOptions, OpenAICompatibleEmbeddingModelOptions,
        OpenAICompatibleEmbeddingProviderOptions, OpenAICompatibleLanguageModelChatOptions,
        OpenAICompatibleLanguageModelCompletionOptions, OpenAICompatibleProviderOptions,
        OpenAiCompatibleCompletionProviderOptions, OpenAiCompatibleEmbeddingModelOptions,
        OpenAiCompatibleEmbeddingProviderOptions, OpenAiCompatibleLanguageModelChatOptions,
        OpenAiCompatibleLanguageModelCompletionOptions, OpenAiCompatibleProviderOptions,
    };
    pub use siumai_provider_openai_compatible::providers::openai_compatible::ext::{
        OpenAiCompatibleChatRequestExt, OpenAiCompatibleCompletionRequestExt,
        OpenAiCompatibleEmbeddingRequestExt,
    };
}

#[allow(deprecated)]
pub use options::{
    OpenAICompatibleCompletionProviderOptions, OpenAICompatibleEmbeddingModelOptions,
    OpenAICompatibleEmbeddingProviderOptions, OpenAICompatibleLanguageModelChatOptions,
    OpenAICompatibleLanguageModelCompletionOptions, OpenAICompatibleProviderOptions,
    OpenAiCompatibleChatRequestExt, OpenAiCompatibleCompletionProviderOptions,
    OpenAiCompatibleCompletionRequestExt, OpenAiCompatibleEmbeddingModelOptions,
    OpenAiCompatibleEmbeddingProviderOptions, OpenAiCompatibleEmbeddingRequestExt,
    OpenAiCompatibleLanguageModelChatOptions, OpenAiCompatibleLanguageModelCompletionOptions,
    OpenAiCompatibleProviderOptions,
};
