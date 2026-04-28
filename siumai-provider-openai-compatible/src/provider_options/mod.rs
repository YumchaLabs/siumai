//! Provider-owned typed option structs for OpenAI-compatible vendors.

pub mod alibaba;
pub mod fireworks;
pub mod mistral;
pub mod moonshotai;
pub mod openai_compatible;
pub mod openrouter;
pub mod perplexity;

#[allow(deprecated)]
pub use alibaba::{
    AlibabaCacheControl, AlibabaChatOptions, AlibabaLanguageModelOptions, AlibabaProviderOptions,
    QwenChatOptions, QwenLanguageModelOptions, QwenProviderOptions,
};
#[allow(deprecated)]
pub use fireworks::{
    FireworksChatOptions, FireworksEmbeddingModelOptions, FireworksEmbeddingProviderOptions,
    FireworksLanguageModelOptions, FireworksProviderOptions, FireworksReasoningHistory,
    FireworksThinkingConfig, FireworksThinkingType,
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
