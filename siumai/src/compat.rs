//! Compatibility surface for legacy, method-style APIs.
//!
//! This module exists to keep older code building while the recommended invocation style
//! moves to the Rust-first model-family APIs:
//! - `siumai::text::*`
//! - `siumai::embedding::*`
//! - `siumai::image::*`
//! - `siumai::rerank::*`
//! - `siumai::speech::*`
//! - `siumai::transcription::*`
//!
//! Recommended construction for new code is registry/config-first:
//! - `registry::global().language_model("openai:gpt-4o-mini")?`
//! - `OpenAiClient::from_config(OpenAiConfig { .. })?`
//!
//! Builder-style construction (`Siumai::builder()` / `Provider::<provider>()`) remains available
//! as a compatibility convenience.

/// Legacy unified interface entry type.
pub use crate::provider::Siumai;

/// Legacy unified builder (provider-agnostic construction).
pub use siumai_registry::provider::SiumaiBuilder;

/// Legacy builder base types (provider builder internals).
///
/// Prefer using provider builders directly (e.g. `Provider::openai()`) unless you are
/// implementing a custom provider.
pub mod builder {
    pub use crate::builder::*;
}
