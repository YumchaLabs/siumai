//! Azure provider-owned typed request options.

pub mod azure;

pub use azure::{AzureOpenAiOptions, AzureReasoningEffort, AzureResponsesApiConfig};
#[allow(deprecated)]
pub use siumai_provider_openai::provider_options::openai::{
    OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions,
};
pub use siumai_provider_openai::provider_options::openai::{
    OpenAIContextManagementConfig, OpenAIContextManagementType, OpenAILanguageModelChatOptions,
    OpenAILanguageModelResponsesOptions, SystemMessageMode,
};
