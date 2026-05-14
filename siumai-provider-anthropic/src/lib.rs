//! siumai-provider-anthropic
//!
//! Anthropic provider implementation.
//!
//! This crate owns:
//! - the Anthropic provider implementation (client + builder + extensions)
//! - provider-owned typed options/metadata and extension traits
//!
//! The reusable Anthropic Messages protocol mapping lives in `siumai-protocol-anthropic`
//! and is re-exported under `crate::standards` for compatibility.
#![deny(unsafe_code)]

// Keep provider-agnostic core modules available only to this crate's implementation.
// Provider crates must not publicly mirror `siumai-core`.
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, error, execution, hosted_tools, observability, retry,
    retry_api, streaming, tools, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

/// Provider-owned legacy parameter types.
pub mod params;

// Provider-owned typed options and typed metadata re-exports (kept out of `siumai-core`).
pub mod provider_metadata;
pub mod provider_options;

pub mod providers;
pub mod standards;

pub use siumai_core::types::{ChatResponse, CommonParams};
