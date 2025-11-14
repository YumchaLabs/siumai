//! Siumai Core
//!
//! Core traits, types and execution primitives.

pub mod error;
pub mod traits;
pub mod types;
// Expose execution transformers for standards crates
pub mod execution;
// Provider specification traits for standards/providers
pub mod provider_spec;
// Core utilities shared across crates (e.g., MIME helpers)
pub mod utils;

/// Version of the core crate (semantic handoff)
pub const VERSION: &str = "0.0.1";

/// Minimal prelude
pub mod prelude {
    pub use crate::VERSION as CORE_VERSION;
    pub use crate::error::*;
}
