//! OpenAI-compatible ProviderSpec re-exports.
//!
//! Canonical protocol implementations live under provider crates (for the OpenAI-like family,
//! `siumai-provider-openai::standards::*`).
//! This module keeps the historical `siumai::providers::openai_compatible::spec::*` path stable.

pub use crate::standards::openai::compat::spec::*;
