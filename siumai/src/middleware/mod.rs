//! Middleware module (model-level)
//!
//! This module defines model-level middleware abstractions. They operate above
//! HTTP interceptors and below high-level orchestrators. The primary initial
//! goal is to support parameter transformation prior to provider-specific
//! request mapping. Wrapping generate/stream calls can be added progressively.

pub mod auto;
pub mod builder;
pub mod language_model;
pub mod named;
pub mod presets;
pub mod samples;
pub mod tag_extractor;

pub use auto::{MiddlewareConfig, build_auto_middlewares, build_auto_middlewares_vec};
pub use builder::*;
pub use language_model::*;
pub use named::*;
pub use tag_extractor::*;
