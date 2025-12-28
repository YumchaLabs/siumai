//! OpenAI provider (re-export).
//!
//! The OpenAI implementation lives in the provider crate `siumai-provider-openai`.

pub use siumai_provider_openai::providers::openai::*;

// Provider-owned typed options and metadata (kept out of `siumai-core`).
pub use siumai_provider_openai::provider_metadata::openai::*;
pub use siumai_provider_openai::provider_options::openai::*;
