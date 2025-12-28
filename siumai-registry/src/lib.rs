//! siumai-registry
//!
//! Provider registry, factories, and handles.
#![deny(unsafe_code)]

// Keep a small stable surface; avoid leaking provider-agnostic internals by default.
pub use siumai_core::{LlmError, custom_provider, error, hosted_tools, retry_api, streaming, traits, types};

// Compatibility / internal re-exports (hidden to reduce accidental coupling).
#[doc(hidden)]
pub use siumai_core::{auth, client, core, defaults, execution, observability, params, retry, utils};

// Backward-compatible re-exports for builds that include built-in providers.
#[cfg(feature = "builtins")]
pub use siumai_providers::{builder, constants, models, providers};

/// Protocol standards are available only when built-in providers are enabled.
#[cfg(feature = "builtins")]
pub use siumai_providers::standards;

pub mod provider;
pub mod provider_builders;
pub mod registry;

// Built-in provider catalog helpers (requires `siumai-providers`).
#[cfg(feature = "builtins")]
pub mod provider_catalog;
