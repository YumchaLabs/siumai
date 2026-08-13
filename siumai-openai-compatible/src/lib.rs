//! Reusable OpenAI-compatible provider runtime for Siumai.
//!
//! This crate composes `siumai-protocol-openai` and `siumai-transport` into a configurable
//! execution engine. It owns explicit generic/custom provider construction and a bounded,
//! versioned codec seam. Branded provider crates own their profiles, options, evidence, and model
//! advisories while reusing this engine for protocol execution.
#![deny(unsafe_code)]

mod configured;

pub use configured::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
    OpenAiCompatibleApiMode, OpenAiCompatibleConfigError, OpenAiCompatibleCredential,
    OpenAiCompatibleLanguageModel, OpenAiCompatibleProfile, OpenAiCompatibleProvider,
    OpenAiCompatibleProviderBuilder, ResponsesWireDialect,
};

/// Versioned, provider-neutral composition hooks for branded provider crates.
///
/// These hooks may shape and decode protocol payloads, but they cannot replace endpoint,
/// authentication, transport, retry, identity, or stream lifecycle policy. Application code
/// should use a branded provider crate or the configured compatible provider instead.
pub mod extension {
    /// First version of the bounded compatible-provider composition contract.
    pub mod v1 {
        pub use crate::configured::{
            ChatCodecPolicy, CompatibleStreamDecoder, PreparedChatCall, PreparedResponsesCall,
            ResponsesCodecPolicy,
        };
    }
}
