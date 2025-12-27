//! siumai-registry
//!
//! Provider registry, factories, and handles.
#![deny(unsafe_code)]

pub use siumai_core::{
    LlmError, auth, client, core, custom_provider, defaults, error, execution, hosted_tools,
    observability, params, retry, retry_api, standards, streaming, traits, types, utils,
};

// Backward-compatible re-exports for builds that include built-in providers.
#[cfg(feature = "builtins")]
pub use siumai_providers::{builder, constants, models, providers};

pub mod provider;
pub mod provider_builders;
pub mod registry;

// Built-in provider catalog helpers (requires `siumai-providers`).
#[cfg(feature = "builtins")]
pub mod provider_catalog;
