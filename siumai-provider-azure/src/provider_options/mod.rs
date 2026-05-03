//! Azure provider-owned typed request options.

pub mod azure;
pub mod openai;

pub use azure::{AzureOpenAiOptions, AzureReasoningEffort, AzureResponsesApiConfig};
#[allow(deprecated)]
pub use openai::{OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions};
pub use openai::{
    OpenAIContextManagementConfig, OpenAIContextManagementType, OpenAILanguageModelChatOptions,
    OpenAILanguageModelResponsesOptions, PromptCacheRetention, ReasoningEffort, ResponsesLogprobs,
    ServiceTier, SystemMessageMode, TextVerbosity, Truncation,
};
