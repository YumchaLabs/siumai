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

// Keep provider-agnostic core modules available only to this crate's implementation.
// Provider crates must not publicly mirror `siumai-core`.
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, error, execution, observability, retry, retry_api,
    streaming, tools, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

/// Provider-owned legacy parameter types.
pub mod params;

pub mod hosted_tools;
pub mod providers;
pub mod standards;

/// Provider-owned typed option structs (OpenAI-specific).
pub mod provider_options;

/// Provider-owned typed metadata structs (OpenAI-specific).
pub mod provider_metadata;
