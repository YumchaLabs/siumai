//! siumai-provider-anthropic
//!
//! Anthropic provider implementation.
//!
//! This crate owns:
//! - the Anthropic provider implementation (client + builder + extensions)
//! - provider-owned typed options/metadata and extension traits
//!
//! The reusable Anthropic Messages protocol mapping lives in `siumai-provider-anthropic-compatible`
//! and is re-exported under `crate::standards` for compatibility.
#![deny(unsafe_code)]

// Re-export the provider-agnostic core modules required by the provider implementation.
// This preserves existing internal module paths in migrated code (e.g. `crate::types::*`).
pub use siumai_core::{
    LlmError, auth, client, core, defaults, error, execution, hosted_tools, observability, retry,
    retry_api, streaming, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub mod builder {
    pub use siumai_core::builder::*;
}

/// Provider-owned legacy parameter types.
pub mod params;

// Provider-owned typed options and typed metadata re-exports (kept out of `siumai-core`).
pub mod provider_metadata;
pub mod provider_options;

pub mod providers;
pub mod standards;

pub use types::{ChatResponse, CommonParams};
