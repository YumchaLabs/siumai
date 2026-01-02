//! siumai-provider-openai-compatible
//!
//! OpenAI-compatible provider foundation (Vercel-aligned):
//! - OpenAI-like protocol standard mapping (Chat/Embedding/Image/Rerank)
//! - Vendor presets + adapter wiring for OpenAI-compatible endpoints
//!
//! Note: This crate intentionally does **not** include the native OpenAI provider.
#![deny(unsafe_code)]

// Re-export the provider-agnostic core modules required by the standard implementation.
// This preserves existing internal-style module paths in migrated code (e.g. `crate::types::*`).
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
