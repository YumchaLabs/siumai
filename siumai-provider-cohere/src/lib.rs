//! siumai-provider-cohere
//!
//! Cohere provider implementation for siumai.
//!
//! Currently, this crate focuses on Cohere's reranking endpoint (`/v2/rerank`) to align
//! with the Vercel AI SDK `@ai-sdk/cohere` behavior.
#![deny(unsafe_code)]

// Re-export the provider-agnostic core modules required by the provider implementation.
// This preserves existing internal module paths in migrated code (e.g. `crate::types::*`).
pub use siumai_core::{
    LlmError, auth, client, core, defaults, error, execution, hosted_tools, observability, params,
    retry, retry_api, streaming, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub mod builder {
    pub use siumai_core::builder::*;
}

pub mod standards;
