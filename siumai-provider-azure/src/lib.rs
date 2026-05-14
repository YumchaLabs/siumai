//! siumai-provider-azure
//!
//! Azure provider implementation for siumai.
//!
//! This crate currently focuses on Azure OpenAI's OpenAI-compatible endpoints
//! (e.g. `/openai/v1/responses?api-version=...`) and is intended to mirror
//! Vercel AI SDK's `@ai-sdk/azure` granularity.
#![deny(unsafe_code)]

// Keep provider-agnostic core modules available only to this crate's implementation.
// Provider crates must not publicly mirror `siumai-core`.
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, error, execution, hosted_tools, observability, retry,
    retry_api, streaming, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

/// Provider-owned typed option structs (Azure-specific).
pub mod provider_options;

/// Provider-owned typed metadata structs (Azure-specific).
pub mod provider_metadata;

pub mod providers;
pub mod standards;
