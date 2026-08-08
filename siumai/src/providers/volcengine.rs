//! Curated Volcengine provider facade for the ARK platform.

pub use siumai_provider_volcengine::{
    BearerCredential, CHAT_SOURCE, CredentialRequest, CredentialSourceError, DEFAULT_BASE_URL,
    DynamicCredentialSource, MODEL_SOURCE, PLATFORM_ID, PROVIDER_ID, RESPONSES_SOURCE, VERIFIED_ON,
    VolcengineConfigError, VolcengineCredential, VolcengineLanguageApi, VolcengineLanguageModel,
    VolcengineProfileError, VolcengineProvider, VolcengineProviderBuilder,
};

pub mod models {
    pub use siumai_provider_volcengine::models::*;
}

pub mod options {
    pub use siumai_provider_volcengine::{
        ARK_BETA_IMAGE_PROCESS_HEADER, ARK_BETA_KNOWLEDGE_SEARCH_HEADER, ArkCaching,
        ArkCachingType, ArkChatOptions, ArkResponsesOptions, ArkResponsesTool, ArkThinking,
        ArkThinkingType,
    };
}
