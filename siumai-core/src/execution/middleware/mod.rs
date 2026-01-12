//! Middleware module (model-level)
//!
//! This module defines model-level middleware abstractions. They operate above
//! HTTP interceptors and below high-level orchestrators. The primary initial
//! goal is to support parameter transformation prior to provider-specific
//! request mapping. Wrapping generate/stream calls can be added progressively.

pub mod auto;
pub mod builder;
pub mod presets;
pub mod samples;

// Group language-model middlewares under `lm` namespace for clarity
pub mod lm {
    pub mod language_model;
    pub mod named;
    pub mod tag_extractor;

    pub use language_model::*;
    pub use named::*;
    pub use tag_extractor::*;
}

pub use auto::{MiddlewareConfig, build_auto_middlewares, build_auto_middlewares_vec};
pub use builder::*;
// Backward-compatible module re-exports
pub use lm::language_model;
pub use lm::named;
pub use lm::tag_extractor;
// Backward-compatible item re-exports
pub use lm::language_model::*;
pub use lm::named::*;
pub use lm::tag_extractor::*;
