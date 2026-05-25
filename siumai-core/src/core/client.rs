//! LlmClient compatibility alias.
//!
//! This module re-exports the generic client trait from the explicit compatibility namespace.
//! Prefer `siumai_core::compat::client::LlmClient` for new compatibility-focused code. This alias
//! is planned for removal after ADR-0007's family-native migration conditions are met.

#[deprecated(
    since = "0.11.0-beta.8",
    note = "use siumai_core::compat::client::LlmClient instead; this lower-level migration alias is planned for removal after ADR-0007 conditions are met"
)]
pub use crate::compat::client::LlmClient;
