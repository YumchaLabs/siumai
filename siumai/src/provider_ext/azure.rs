pub use siumai_provider_azure::providers::azure_openai::{
    AzureChatMode, AzureOpenAIProviderSettings, AzureOpenAiBuilder, AzureOpenAiClient,
    AzureOpenAiConfig, AzureOpenAiSpec, AzureUrlConfig, VERSION,
};
use siumai_registry::provider::SiumaiBuilder;

/// Create the unified Azure provider builder.
pub fn azure() -> SiumaiBuilder {
    SiumaiBuilder::new().azure()
}

/// Create the unified Azure provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createAzure()`.
pub fn create_azure() -> SiumaiBuilder {
    azure()
}

/// Typed provider options (`provider_options_map["azure"]`).
pub mod options {
    pub use siumai_provider_azure::provider_options::{
        AzureOpenAiOptions, AzureReasoningEffort, AzureResponsesApiConfig,
        OpenAIContextManagementConfig, OpenAIContextManagementType, OpenAILanguageModelChatOptions,
        OpenAILanguageModelResponsesOptions, PromptCacheRetention, ReasoningEffort,
        ResponsesLogprobs, ServiceTier, SystemMessageMode, TextVerbosity, Truncation,
    };
    #[allow(deprecated)]
    pub use siumai_provider_azure::provider_options::{
        OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions,
    };
    pub use siumai_provider_azure::providers::azure_openai::AzureOpenAiChatRequestExt;
}
pub use options::{
    AzureOpenAiChatRequestExt, AzureOpenAiOptions, AzureReasoningEffort, AzureResponsesApiConfig,
    OpenAIContextManagementConfig, OpenAIContextManagementType, OpenAILanguageModelChatOptions,
    OpenAILanguageModelResponsesOptions, PromptCacheRetention, ReasoningEffort, ResponsesLogprobs,
    ServiceTier, SystemMessageMode, TextVerbosity, Truncation,
};
#[allow(deprecated)]
pub use options::{OpenAIChatLanguageModelOptions, OpenAIResponsesProviderOptions};

/// Typed response metadata helpers (`ChatResponse.provider_metadata["azure"]`).
pub mod metadata {
    pub use siumai_provider_azure::provider_metadata::azure::{
        AzureChatResponseExt, AzureContentPartExt, AzureContentPartMetadata, AzureMetadata,
        AzureResponsesProviderMetadata, AzureResponsesReasoningProviderMetadata,
        AzureResponsesSourceDocumentProviderMetadata, AzureResponsesTextProviderMetadata,
        AzureSource, AzureSourceExt, AzureSourceMetadata,
    };
}
pub use metadata::{
    AzureChatResponseExt, AzureContentPartExt, AzureContentPartMetadata, AzureMetadata,
    AzureResponsesProviderMetadata, AzureResponsesReasoningProviderMetadata,
    AzureResponsesSourceDocumentProviderMetadata, AzureResponsesTextProviderMetadata, AzureSource,
    AzureSourceExt, AzureSourceMetadata,
};
