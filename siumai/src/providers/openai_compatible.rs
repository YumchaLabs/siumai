//! Curated OpenAI-compatible provider facade.

pub use siumai_provider_openai_compatible::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
    OpenAiCompatibleConfigError, OpenAiCompatibleCredential, OpenAiCompatibleLanguageModel,
    OpenAiCompatibleProfile, OpenAiCompatibleProvider, OpenAiCompatibleProviderBuilder,
};

pub mod options {
    pub use siumai_provider_openai_compatible::provider_options::{
        DeepSeekLanguageModelOptions, DeepSeekThinkingConfig, DeepSeekThinkingType,
        KimiLanguageOptions, KimiReasoningEffort, KimiThinking, KimiThinkingMode,
        KimiThinkingRetention, OpenAiCompatibleEmbeddingModelOptions,
        OpenAiCompatibleLanguageModelChatOptions, OpenAiCompatibleLanguageModelCompletionOptions,
    };
}

pub mod profiles {
    pub mod deepseek {
        pub use siumai_provider_openai_compatible::profiles::deepseek::{
            CHAT, FLASH, OFFICIAL_SOURCE, PRO, REASONER, VERIFIED_ON, profile,
        };
    }

    pub mod moonshotai {
        pub use siumai_provider_openai_compatible::profiles::moonshotai::{
            CHAT, KIMI_K2_5, KIMI_K2_6, KIMI_K2_7_CODE, KIMI_K2_7_CODE_HIGHSPEED, KIMI_K3,
            MODEL_SOURCE, OFFICIAL_SOURCE, VERIFIED_ON, profile,
        };
    }
}
