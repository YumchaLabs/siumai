//! OpenAI-compatible adapter + config + streaming protocol layer.
//!
//! Note: The protocol-level building blocks (`adapter`, `openai_config`, `provider_registry`,
//! `types`) live in `siumai-core` to reduce cross-crate coupling. This module keeps the historical
//! `siumai_provider_openai::standards::openai::compat::*` path by re-exporting them.

pub use siumai_core::standards::openai::compat::{
    adapter, openai_config, provider_registry, types,
};

pub mod spec;
pub use siumai_core::standards::openai::compat::{streaming, transformers};
