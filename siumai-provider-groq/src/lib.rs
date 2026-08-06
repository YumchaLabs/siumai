//! Rust-first Groq provider for Chat Completions, Responses, and final-result transcription.
//!
//! [`GroqProvider`] is long-lived and model-independent. Language and transcription model handles
//! are lightweight, synchronously constructed, and accept open model IDs. Groq-specific controls
//! are expressed through typed provider options rather than a universal client or arbitrary JSON
//! settings map.

#![deny(unsafe_code)]

mod language;
mod metadata;
pub mod models;
mod options;
mod provider;
pub mod tools;
mod transcription;

pub use language::{
    CHAT_SOURCE, DEFAULT_BASE_URL, DEPRECATIONS_SOURCE, GroqProfileError, MODEL_CATALOG_SOURCE,
    PLATFORM_ID, PROVIDER_ID, RESPONSES_SOURCE, VERIFIED_ON,
};
pub use metadata::{
    GroqLanguageMetadata, GroqLanguageResponseExt, GroqTranscriptionMetadata,
    GroqTranscriptionResponseExt,
};
pub use options::{
    GroqLanguageOptions, GroqReasoningEffort, GroqReasoningFormat, GroqResponsesOptions,
    GroqResponsesServiceTier, GroqServiceTier, GroqTimestampGranularity, GroqTranscriptionOptions,
    GroqTranscriptionResponseFormat,
};
pub use provider::{
    GroqConfigError, GroqCredential, GroqLanguageApi, GroqLanguageModel, GroqProvider,
    GroqProviderBuilder,
};
pub use siumai_openai_compatible::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
};
pub use transcription::{
    GroqTranscriptionModel, TRANSCRIPTION_API_MODE_ID, TRANSCRIPTION_PROTOCOL_ID,
    TRANSCRIPTION_SOURCE,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
