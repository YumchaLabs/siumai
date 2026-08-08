//! Volcengine ARK provider integration for Siumai.
//!
//! This crate owns Volcengine identity, credentials, endpoint selection, typed ARK options, model
//! advisories, and the verified ARK OpenAI dialect. Network execution is delegated to the shared
//! OpenAI-compatible runtime without exposing that engine as the public provider owner.
#![deny(unsafe_code)]

mod language;
pub mod models;
pub mod options;
mod provider;

pub use language::{
    CHAT_SOURCE, DEFAULT_BASE_URL, MODEL_SOURCE, PLATFORM_ID, PROVIDER_ID, RESPONSES_SOURCE,
    VERIFIED_ON, VolcengineProfileError,
};
pub use options::{
    ARK_BETA_IMAGE_PROCESS_HEADER, ARK_BETA_KNOWLEDGE_SEARCH_HEADER, ArkCaching, ArkCachingType,
    ArkChatOptions, ArkResponsesOptions, ArkResponsesTool, ArkThinking, ArkThinkingType,
};
pub use provider::{
    VolcengineConfigError, VolcengineCredential, VolcengineLanguageApi, VolcengineLanguageModel,
    VolcengineProvider, VolcengineProviderBuilder,
};
pub use siumai_openai_compatible::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
