//! siumai-provider-openai
//!
//! OpenAI provider implementation.
//!
//! This crate owns:
//! - the OpenAI provider implementation (client + builder + extensions)
//! - the OpenAI-compatible vendor provider implementation (configuration presets + adapter wiring)
//!
//! The reusable OpenAI-like protocol mapping lives in `siumai-protocol-openai` and is re-exported
//! under `crate::standards` for compatibility.
#![deny(unsafe_code)]

// Re-export the provider-agnostic core modules required by the standard implementation.
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

pub mod providers;
pub mod standards;

/// Provider-owned typed option structs (OpenAI-specific).
pub mod provider_options;

/// Provider-owned typed metadata structs (OpenAI-specific).
pub mod provider_metadata;
