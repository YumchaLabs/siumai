//! siumai-registry
//!
//! Provider registry, factories, and handles.
#![deny(unsafe_code)]

// Keep a small stable surface; avoid leaking provider-agnostic internals by default.
pub use siumai_core::client::LlmClient;
pub use siumai_core::{
    LlmError, custom_provider, error, hosted_tools, retry_api, streaming, traits, types,
};

// Internal aliases for registry implementation (not part of the public API).
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    auth, client, core, defaults, execution, observability, params, retry, utils,
};

/// Experimental low-level APIs (advanced use only).
///
/// This module exposes lower-level building blocks from `siumai-core` without
/// making them part of the stable surface of `siumai-registry`.
pub mod experimental {
    pub use siumai_core::core::*;
    pub use siumai_core::{
        auth, client, core, defaults, execution, observability, params, retry, utils,
    };
}

// Note: `siumai-registry` intentionally does not re-export provider crates.
// Use the `siumai` facade for stable entry points (`provider_ext`, `prelude::unified`, etc.).

pub mod provider;
pub mod provider_builders;
pub mod registry;

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(feature = "builtins")]
mod native_provider_metadata;

// Built-in provider catalog helpers (feature-gated; depends on provider crates).
#[cfg(feature = "builtins")]
pub mod provider_catalog;
