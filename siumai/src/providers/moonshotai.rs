//! Curated Moonshot AI provider facade for the Kimi product surface.

pub use siumai_provider_moonshotai::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
    KimiAssistantPartial, KimiFile, KimiFileDeleteResult, KimiFileList, KimiFileUpload,
    KimiFileUploadPurpose, KimiFiles, MoonshotConfigError, MoonshotCredential,
    MoonshotLanguageModel, MoonshotProvider, MoonshotProviderBuilder,
};

pub mod models {
    pub use siumai_provider_moonshotai::{
        API_OVERVIEW_SOURCE, CHAT, KIMI_K2_5, KIMI_K2_6, KIMI_K2_7_CODE, KIMI_K2_7_CODE_HIGHSPEED,
        KIMI_K3, MODEL_SOURCE, OFFICIAL_SOURCE, VERIFIED_ON,
    };
}

pub mod options {
    pub use siumai_provider_moonshotai::{
        KimiLanguageOptions, KimiReasoningEffort, KimiThinking, KimiThinkingMode,
        KimiThinkingRetention,
    };
}
