//! siumai-provider-gemini
//!
//! Google Gemini provider implementation.
//!
//! This crate owns:
//! - the Gemini provider implementation (client + builder + extensions)
//! - provider-owned typed options/metadata and extension traits
//!
//! Protocol mapping and streaming helpers live in `siumai-protocol-gemini` and are re-exported
//! from this crate under `crate::standards` for compatibility.
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

// Provider-owned typed options and metadata (kept out of `siumai-core`).
pub mod provider_metadata;
pub mod provider_options;

pub mod providers;
pub use siumai_protocol_gemini::standards;

pub use siumai_core::types::{ChatResponse, CommonParams};
