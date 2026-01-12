//! siumai-protocol-gemini
//!
//! Google Gemini protocol standard for siumai:
//! request/response mapping, streaming conversion, and protocol-local helpers.
#![deny(unsafe_code)]

// Re-export the provider-agnostic core modules required by the protocol implementation.
// This preserves existing internal module paths in migrated code (e.g. `crate::types::*`).
pub use siumai_core::{
    LlmError, auth, client, core, defaults, encoding, error, execution, hosted_tools,
    observability, retry, retry_api, streaming, tools, traits, types, utils,
};

/// Builder utilities shared across workspace crates.
pub mod builder {
    pub use siumai_core::builder::*;
}

pub mod standards;

pub use types::{ChatResponse, CommonParams};
