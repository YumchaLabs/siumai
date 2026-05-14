//! siumai-provider-cohere
//!
//! Cohere provider implementation for siumai.
//!
//! This crate exposes Cohere's native unified provider surface:
//! - chat (`/v2/chat`)
//! - embedding (`/v2/embed`)
//! - rerank (`/v2/rerank`)
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

/// Provider-owned typed option structs (Cohere-specific).
pub mod provider_options;
