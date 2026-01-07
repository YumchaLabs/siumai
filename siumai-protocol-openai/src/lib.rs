//! siumai-protocol-openai
//!
//! OpenAI(-like) protocol family mapping for siumai.
//!
//! This crate owns the vendor-agnostic protocol layer for the OpenAI family:
//! - OpenAI-like protocol mapping (Chat/Embedding/Image/Rerank)
//! - OpenAI-compatible adapters + vendor routing utilities
//! - OpenAI Responses API mapping (feature-gated)
//!
//! Provider crates (e.g. `siumai-provider-openai`, `siumai-provider-xai`) should depend on this
//! crate, and keep provider-specific quirks behind provider-owned presets/wrappers.
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

pub mod providers;
pub mod standards;
