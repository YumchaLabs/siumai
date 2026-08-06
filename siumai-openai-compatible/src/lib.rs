//! Reusable OpenAI-compatible provider runtime for Siumai.
//!
//! This crate composes `siumai-protocol-openai` and `siumai-transport` into a configurable
//! execution engine. It owns verified compatible profiles and the explicit custom-provider escape
//! hatch. Branded provider crates may reuse its bounded codec hooks without depending on another
//! branded provider package.
#![deny(unsafe_code)]

mod configured;

pub use configured::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
    OpenAiCompatibleApiMode, OpenAiCompatibleConfigError, OpenAiCompatibleCredential,
    OpenAiCompatibleLanguageModel, OpenAiCompatibleProfile, OpenAiCompatibleProvider,
    OpenAiCompatibleProviderBuilder, profiles,
};

/// Low-level bounded hooks for branded provider crates that reuse the compatible runtime.
///
/// These hooks may shape and decode protocol payloads, but they cannot replace endpoint,
/// authentication, transport, retry, identity, or stream lifecycle policy.
#[doc(hidden)]
pub mod extension {
    pub use crate::configured::{
        ChatCodecPolicy, CompatibleStreamDecoder, PreparedChatCall, PreparedResponsesCall,
        ResponsesCodecPolicy,
    };
}

/// Provider-owned typed option structs for OpenAI-compatible vendors.
pub mod provider_options;
