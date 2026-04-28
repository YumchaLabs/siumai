//! Provider-owned typed option structs for OpenAI-compatible vendors.

pub mod alibaba;
pub mod deepseek;
pub mod fireworks;
pub mod groq;
pub mod mistral;
pub mod moonshotai;
pub mod openai_compatible;
pub mod openrouter;
pub mod perplexity;
pub mod togetherai;
pub mod xai;

#[allow(deprecated)]
pub use alibaba::{
    AlibabaCacheControl, AlibabaChatOptions, AlibabaLanguageModelOptions, AlibabaProviderOptions,
    AlibabaVideoModelOptions, AlibabaVideoProviderOptions, QwenChatOptions,
    QwenLanguageModelOptions, QwenProviderOptions,
};
#[allow(deprecated)]
pub use deepseek::{
    DeepSeekChatOptions, DeepSeekLanguageModelOptions, DeepSeekProviderOptions,
    DeepSeekThinkingConfig, DeepSeekThinkingType,
};
#[allow(deprecated)]
pub use fireworks::{
    FireworksChatOptions, FireworksEmbeddingModelOptions, FireworksEmbeddingProviderOptions,
    FireworksLanguageModelOptions, FireworksProviderOptions, FireworksReasoningHistory,
    FireworksThinkingConfig, FireworksThinkingType,
};
#[allow(deprecated)]
pub use groq::{
    GroqChatOptions, GroqLanguageModelOptions, GroqProviderOptions, GroqReasoningEffort,
    GroqReasoningFormat, GroqServiceTier, GroqTranscriptionModelOptions,
};
pub use mistral::{MistralChatOptions, MistralLanguageModelOptions, MistralReasoningEffort};
#[allow(deprecated)]
pub use moonshotai::{
    MoonshotAIChatOptions, MoonshotAILanguageModelOptions, MoonshotAIProviderOptions,
    MoonshotAIReasoningHistory, MoonshotAIThinkingConfig, MoonshotAIThinkingType,
};
#[allow(deprecated)]
pub use openai_compatible::{
    OpenAICompatibleCompletionProviderOptions, OpenAICompatibleEmbeddingModelOptions,
    OpenAICompatibleEmbeddingProviderOptions, OpenAICompatibleLanguageModelChatOptions,
    OpenAICompatibleLanguageModelCompletionOptions, OpenAICompatibleProviderOptions,
    OpenAiCompatibleCompletionProviderOptions, OpenAiCompatibleEmbeddingModelOptions,
    OpenAiCompatibleEmbeddingProviderOptions, OpenAiCompatibleLanguageModelChatOptions,
    OpenAiCompatibleLanguageModelCompletionOptions, OpenAiCompatibleProviderOptions,
};
pub use openrouter::{OpenRouterOptions, OpenRouterTransform};
pub use perplexity::{
    PerplexityOptions, PerplexitySearchContextSize, PerplexitySearchMode,
    PerplexitySearchRecencyFilter, PerplexityUserLocation, PerplexityWebSearchOptions,
};
#[allow(deprecated)]
pub use togetherai::{
    TogetherAIImageModelOptions, TogetherAIImageProviderOptions, TogetherAIRerankingModelOptions,
    TogetherAIRerankingOptions, TogetherAiImageModelOptions, TogetherAiImageProviderOptions,
    TogetherAiRerankingModelOptions, TogetherAiRerankingOptions,
};
#[allow(deprecated)]
pub use xai::{
    NewsSearchSource, RssSearchSource, SearchMode, SearchSource, WebSearchSource, XSearchSource,
    XaiChatReasoningEffort, XaiFilesOptions, XaiImageModelOptions, XaiImageProviderOptions,
    XaiImageQuality, XaiImageResolution, XaiLanguageModelChatOptions,
    XaiLanguageModelResponsesOptions, XaiOptions, XaiProviderOptions, XaiReasoningSummary,
    XaiResponseInclude, XaiResponsesProviderOptions, XaiResponsesReasoningEffort,
    XaiSearchParameters, XaiVideoMode, XaiVideoModelOptions, XaiVideoProviderOptions,
    XaiVideoResolution,
};
