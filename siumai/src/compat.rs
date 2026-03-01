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
//! Construction is still typically done via `Siumai::builder()` or `Provider::<provider>()`,
//! but new code should prefer calling the family APIs for actual inference.

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
