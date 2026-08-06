//! Rust-first xAI provider for Siumai.
//!
//! The configured provider is long-lived and model-independent. Language handles default to the
//! xAI Responses API, while Chat Completions remains explicit.
#![deny(unsafe_code)]

pub mod provider_options;
pub mod providers;
pub mod tools;

pub use provider_options::{
    NewsSearchSource, RssSearchSource, SearchMode, SearchSource, WebSearchSource, XSearchSource,
    XaiChatOptions, XaiChatReasoningEffort, XaiReasoningSummary, XaiResponseInclude,
    XaiResponsesOptions, XaiResponsesReasoningEffort, XaiSearchParameters,
};
pub use providers::xai::{
    XaiConfigError, XaiCredential, XaiErrorData, XaiErrorPayload, XaiLanguageApi, XaiLanguageModel,
    XaiProvider, XaiProviderBuilder,
};
pub use siumai_openai_compatible::{CredentialSourceError, DynamicCredentialSource};
