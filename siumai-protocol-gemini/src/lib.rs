//! siumai-protocol-gemini
//!
//! Google Gemini protocol standard for siumai:
//! request/response mapping, streaming conversion, and protocol-local helpers.
#![deny(unsafe_code)]

// Keep provider-agnostic core modules available only to this crate's implementation.
// Protocol crates must not publicly mirror `siumai-core`.
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, encoding, error, execution, hosted_tools,
    observability, retry, retry_api, streaming, tools, traits, types, utils,
};

/// Builder utilities shared across workspace crates.
pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

pub mod standards;

pub use siumai_core::types::{ChatResponse, CommonParams};
