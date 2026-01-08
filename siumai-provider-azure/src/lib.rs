//! siumai-provider-azure
//!
//! Azure provider implementation for siumai.
//!
//! This crate currently focuses on Azure OpenAI's OpenAI-compatible endpoints
//! (e.g. `/openai/v1/responses?api-version=...`) and is intended to mirror
//! Vercel AI SDK's `@ai-sdk/azure` granularity.
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

pub mod providers;
pub mod standards;
