//! siumai-protocol-gemini
//!
//! Google Gemini protocol standard for siumai:
//! request/response mapping, streaming conversion, and protocol-local helpers.
#![deny(unsafe_code)]

// Keep provider-agnostic core modules available only to this crate's implementation.
// Protocol crates must not publicly mirror `siumai-core`.
#[allow(unused_imports)]
pub(crate) use siumai_provider_utils as provider_utils;

#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, encoding, error, execution, observability, retry,
    retry_api, streaming, traits, types, utils,
};

pub mod hosted_tools;

/// Typed response metadata owned by the Google/Gemini protocol family.
pub mod provider_metadata;

/// Provider-defined tool catalog owned by the Google/Gemini protocol family.
pub mod tool_catalog;

/// Builder utilities shared across workspace crates.
pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

pub mod standards;

pub use siumai_core::types::{ChatResponse, CommonParams};
