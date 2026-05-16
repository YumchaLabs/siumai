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

// Keep provider-agnostic core modules available only to this crate's implementation.
// Provider crates must not publicly mirror `siumai-core`.
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, client, core, defaults, error, execution, observability, retry, retry_api, streaming,
    traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

pub mod auth;
pub mod hosted_tools;
pub mod provider_metadata;
pub mod provider_options;
pub mod providers;
pub mod standards;
pub mod tools;
