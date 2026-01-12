//! siumai-provider-togetherai
//!
//! TogetherAI provider implementation for siumai.
//!
//! Currently, this crate focuses on TogetherAI's reranking endpoint (`/v1/rerank`) to align
//! with the Vercel AI SDK `@ai-sdk/togetherai` behavior.
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
