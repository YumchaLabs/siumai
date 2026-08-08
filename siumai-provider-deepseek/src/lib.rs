//! DeepSeek provider integration for Siumai.
//!
//! The crate owns DeepSeek identity, credentials, endpoint selection, typed options, model
//! advisories, DeepSeek-specific OpenAI dialect policy, and the official Anthropic-compatible
//! Messages surface. Network execution is delegated to the shared configured protocol runtimes.
#![deny(unsafe_code)]

mod annotations;
mod language;
pub mod models;
pub mod options;
mod provider;

pub use annotations::DeepSeekAssistantPrefix;
pub use language::DeepSeekProfileError;
pub use options::{
    DeepSeekChatOptions, DeepSeekReasoningEffort, DeepSeekResponsesOptions, DeepSeekResponsesTool,
    DeepSeekThinkingConfig, DeepSeekThinkingType,
};
pub use provider::{
    DeepSeekConfigError, DeepSeekCredential, DeepSeekLanguageApi, DeepSeekLanguageModel,
    DeepSeekProvider, DeepSeekProviderBuilder,
};
pub use siumai_openai_compatible::{CredentialSourceError, DynamicCredentialSource};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
