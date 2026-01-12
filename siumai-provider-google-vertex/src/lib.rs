//! siumai-provider-google-vertex
//!
//! Google Vertex AI provider implementation.
//!
//! This crate owns:
//! - Vertex AI provider implementation (clients + specs)
//! - provider-owned typed options and extension traits
//!
//! Vertex protocol mapping modules (e.g. Imagen via `:predict`) live in `crate::standards`.
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

pub mod provider_options;
pub mod providers;
pub mod standards;
pub mod tools;
