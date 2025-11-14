//! Error module wrapper
//!
//! Delegates core error types and conversions to `siumai-core`, while keeping
//! user-facing helpers and handlers within the aggregator crate.

mod conversions;
pub mod handlers;
pub mod helpers;
pub mod types;

pub use handlers::*;
pub use helpers::*;
pub use types::*;
