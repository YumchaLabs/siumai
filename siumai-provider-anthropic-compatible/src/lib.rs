//! siumai-provider-anthropic-compatible
//!
//! Anthropic-compatible protocol standard (Vercel-aligned):
//! - Anthropic Messages API mapping (Chat + streaming)
//! - Prompt caching + thinking helpers (protocol-level)
//!
//! Note: This crate intentionally does **not** include a native Anthropic provider client.
#![deny(unsafe_code)]

// Re-export the provider-agnostic core modules required by the standard implementation.
// This preserves existing internal-style module paths in migrated code (e.g. `crate::types::*`).
pub use siumai_core::{
    LlmError, auth, client, core, defaults, error, execution, hosted_tools, observability, retry,
    retry_api, streaming, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub mod builder {
    pub use siumai_core::builder::*;
}

/// Protocol-owned typed metadata views.
pub mod provider_metadata;

pub mod standards;

pub use types::{ChatResponse, CommonParams};
