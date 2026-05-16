//! siumai-protocol-anthropic
//!
//! Anthropic Messages protocol mapping for siumai.
//!
//! This crate owns the vendor-agnostic protocol layer for Anthropic Messages:
//! - Messages API request/response mapping
//! - Streaming event conversion
//! - Prompt caching + thinking helpers (protocol-level)
//!
//! Provider crates should depend on this crate and keep vendor-specific quirks behind
//! provider-owned presets/wrappers.
#![deny(unsafe_code)]

// Keep provider-agnostic core modules available only to this crate's implementation.
// Protocol crates must not publicly mirror `siumai-core`.
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, encoding, error, execution, observability, retry,
    retry_api, streaming, tools, traits, types, utils,
};

pub mod hosted_tools;

/// Builder utilities shared across provider crates.
pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

/// Protocol-owned typed metadata views.
pub mod provider_metadata;

pub mod standards;

pub use siumai_core::types::{ChatResponse, CommonParams};
