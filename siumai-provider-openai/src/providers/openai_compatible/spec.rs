//! OpenAI-compatible ProviderSpec re-exports.
//!
//! Canonical protocol implementations live under `siumai-providers::standards::*`
//! (and the OpenAI-like family crate `siumai-provider-openai`).
//! This module keeps the historical `siumai::providers::openai_compatible::spec::*` path stable.

pub use crate::standards::openai::compat::spec::*;

