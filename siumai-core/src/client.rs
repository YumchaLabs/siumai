//! Legacy generic-client compatibility alias.
//!
//! The physical `LlmClient` implementation lives under `siumai_core::compat::client`.
//! This module is kept as a migration alias for older imports.

pub use crate::compat::client::*;
