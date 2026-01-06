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

// Provider-owned typed options and metadata (kept out of `siumai-core`).
pub mod provider_metadata;
pub mod provider_options;

pub mod providers;
pub use siumai_protocol_gemini::standards;

pub use types::{ChatResponse, CommonParams};
