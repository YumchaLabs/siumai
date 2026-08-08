//! Rust-first Moonshot AI provider for the Kimi language-model product surface.
//!
//! [`MoonshotProvider`] is long-lived and model-independent. It owns Moonshot AI identity,
//! credentials, technical endpoint selection, Kimi model advisories, typed options, and the
//! verified Kimi Chat Completions dialect. Network execution is delegated to the reusable
//! OpenAI-compatible engine without exposing Kimi as a compatibility-engine profile.

#![deny(unsafe_code)]

mod language;
pub mod options;
mod provider;

pub use language::{
    API_OVERVIEW_SOURCE, CHAT, DEFAULT_BASE_URL, KIMI_K2_5, KIMI_K2_6, KIMI_K2_7_CODE,
    KIMI_K2_7_CODE_HIGHSPEED, KIMI_K3, MODEL_SOURCE, OFFICIAL_SOURCE, PLATFORM_ID, PROVIDER_ID,
    VERIFIED_ON,
};
pub use options::{
    KimiLanguageOptions, KimiReasoningEffort, KimiThinking, KimiThinkingMode, KimiThinkingRetention,
};
pub use provider::{
    MoonshotConfigError, MoonshotCredential, MoonshotLanguageModel, MoonshotProvider,
    MoonshotProviderBuilder,
};
pub use siumai_openai_compatible::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
