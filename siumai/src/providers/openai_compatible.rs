//! Curated OpenAI-compatible provider facade.

pub use siumai_provider_openai_compatible::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
    OpenAiCompatibleConfigError, OpenAiCompatibleCredential, OpenAiCompatibleLanguageModel,
    OpenAiCompatibleProfile, OpenAiCompatibleProvider, OpenAiCompatibleProviderBuilder,
    RetiredModelBehavior,
};

pub mod options {
    pub use siumai_provider_openai_compatible::provider_options::{
        DeepSeekLanguageModelOptions, DeepSeekThinkingConfig, DeepSeekThinkingType,
        OpenAiCompatibleEmbeddingModelOptions, OpenAiCompatibleLanguageModelChatOptions,
        OpenAiCompatibleLanguageModelCompletionOptions,
    };
}

pub mod profiles {
    pub mod deepseek {
        pub use siumai_provider_openai_compatible::profiles::deepseek::{
            CHAT, FLASH, OFFICIAL_SOURCE, PRO, REASONER, VERIFIED_ON, profile,
        };
    }
}
