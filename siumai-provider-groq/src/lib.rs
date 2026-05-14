//! siumai-provider-groq
//!
//! Groq provider implementation built on the OpenAI-like protocol standard.
#![deny(unsafe_code)]

// Keep provider-agnostic core modules available only to this crate's implementation.
// Provider crates must not publicly mirror `siumai-core`.
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, error, execution, hosted_tools, observability, params,
    retry, retry_api, streaming, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

pub mod providers;
pub mod standards;

pub use siumai_core::types::{ChatResponse, CommonParams};

/// Provider-owned typed response metadata (`ChatResponse.provider_metadata["groq"]`).
pub mod provider_metadata;
/// Provider-owned typed option structs (Groq-specific).
pub mod provider_options;
