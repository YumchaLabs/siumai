//! Siumai OpenAI-Compatible Provider (extracted)
//!
//! This crate hosts the adapter/types for OpenAI-compatible providers
//! to allow the aggregator to re-export them under feature gates.

pub const VERSION: &str = "0.0.1";

pub mod adapter;
pub mod helpers;
pub mod registry;
pub mod types;

pub use adapter::*;
pub use registry::*;
pub use types::*;
