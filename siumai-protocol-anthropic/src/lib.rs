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

// Re-export the provider-agnostic core modules required by the standard implementation.
// This preserves existing internal-style module paths in migrated code (e.g. `crate::types::*`).
pub use siumai_core::{
    LlmError, auth, client, core, defaults, encoding, error, execution, hosted_tools,
    observability, retry, retry_api, streaming, tools, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub mod builder {
    pub use siumai_core::builder::*;
}

/// Protocol-owned typed metadata views.
pub mod provider_metadata;

pub mod standards;

pub use types::{ChatResponse, CommonParams};
