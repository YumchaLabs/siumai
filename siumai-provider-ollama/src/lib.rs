//! siumai-provider-ollama
//!
//! Ollama provider implementation + shared Ollama protocol standard.
//!
//! This crate owns:
//! - the Ollama provider implementation (client + builder + extensions)
//! - the Ollama protocol mapping and streaming helpers used by the provider
#![deny(unsafe_code)]

// Keep provider-agnostic core modules available only to this crate's implementation.
// Provider crates must not publicly mirror `siumai-core`.
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, error, execution, observability, params, retry,
    retry_api, streaming, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

pub mod providers;
pub mod standards;

pub use siumai_core::types::{ChatResponse, CommonParams};

/// Provider-owned typed option structs (Ollama-specific).
pub mod provider_options;

/// Provider-owned typed response metadata.
pub mod provider_metadata;
