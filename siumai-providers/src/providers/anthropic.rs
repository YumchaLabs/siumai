//! Anthropic provider (re-export).
//!
//! The Anthropic implementation lives in the provider crate `siumai-provider-anthropic`.

pub use siumai_provider_anthropic::providers::anthropic::*;

// Provider-owned typed options and metadata (kept out of `siumai-core`).
pub use siumai_provider_anthropic::provider_metadata::anthropic::*;
pub use siumai_provider_anthropic::provider_options::anthropic::*;
