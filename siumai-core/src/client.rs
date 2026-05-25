//! Legacy generic-client compatibility alias.
//!
//! The physical `LlmClient` implementation lives under `siumai_core::compat::client`.
//! This module is kept as a migration alias for older imports.
//!
//! Prefer `siumai_core::compat::client` for new compatibility-focused code. This alias is planned
//! for removal after ADR-0007's family-native migration conditions are met.

#[deprecated(
    since = "0.11.0-beta.8",
    note = "use siumai_core::compat::client instead; this lower-level migration alias is planned for removal after ADR-0007 conditions are met"
)]
pub use crate::compat::client::*;
