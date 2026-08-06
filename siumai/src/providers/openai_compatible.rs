//! Curated OpenAI-compatible provider facade.

pub use siumai_openai_compatible::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
    OpenAiCompatibleApiMode, OpenAiCompatibleConfigError, OpenAiCompatibleCredential,
    OpenAiCompatibleLanguageModel, OpenAiCompatibleProfile, OpenAiCompatibleProvider,
    OpenAiCompatibleProviderBuilder,
};

pub mod options {
    pub use siumai_openai_compatible::provider_options::{
        ArkCaching, ArkCachingType, ArkChatOptions, ArkResponsesOptions, ArkResponsesTool,
        ArkThinking, ArkThinkingType, KimiLanguageOptions, KimiReasoningEffort, KimiThinking,
        KimiThinkingMode, KimiThinkingRetention,
    };
}

pub mod profiles {
    pub mod ark {
        pub use siumai_openai_compatible::profiles::ark::{
            ArkProfileError, CHAT_SOURCE, DOUBAO_SEED_2_1_PRO_260628, MODEL_SOURCE,
            OFFICIAL_BASE_URL, PLATFORM_ID, PROVIDER_ID, RESPONSES_SOURCE, VERIFIED_ON, profile,
        };
    }

    pub mod moonshotai {
        pub use siumai_openai_compatible::profiles::moonshotai::{
            CHAT, KIMI_K2_5, KIMI_K2_6, KIMI_K2_7_CODE, KIMI_K2_7_CODE_HIGHSPEED, KIMI_K3,
            MODEL_SOURCE, OFFICIAL_SOURCE, VERIFIED_ON, profile,
        };
    }
}
