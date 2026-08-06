//! Provider-owned typed option structs for OpenAI-compatible vendors.

pub mod ark;
pub mod moonshotai;

pub use ark::{
    ARK_BETA_IMAGE_PROCESS_HEADER, ARK_BETA_KNOWLEDGE_SEARCH_HEADER, ArkCaching, ArkCachingType,
    ArkChatOptions, ArkResponsesOptions, ArkResponsesTool, ArkThinking, ArkThinkingType,
};
pub use moonshotai::{
    KimiLanguageOptions, KimiReasoningEffort, KimiThinking, KimiThinkingMode, KimiThinkingRetention,
};
