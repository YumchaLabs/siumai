//! Curated DeepSeek language provider facade.

pub use siumai_provider_deepseek::{
    CredentialSourceError, DeepSeekAssistantPrefix, DeepSeekConfigError, DeepSeekCredential,
    DeepSeekLanguageApi, DeepSeekLanguageModel, DeepSeekProfileError, DeepSeekProvider,
    DeepSeekProviderBuilder, DynamicCredentialSource,
};

pub mod models {
    pub use siumai_provider_deepseek::models::{
        ALL, ALL_CHAT, ALL_RESPONSES, DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO, MODEL_CATALOG_SOURCE,
        MODEL_CATALOG_VERIFIED_ON,
    };
}

pub mod options {
    pub use siumai_provider_deepseek::{
        DeepSeekChatOptions, DeepSeekReasoningEffort, DeepSeekResponsesOptions,
        DeepSeekResponsesTool, DeepSeekThinkingConfig, DeepSeekThinkingType,
    };
}
