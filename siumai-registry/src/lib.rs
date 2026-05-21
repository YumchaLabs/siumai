//! siumai-registry
//!
//! Provider registry, factories, and handles.
#![deny(unsafe_code)]

// Keep a small stable surface; avoid leaking provider-agnostic internals by default.
pub use siumai_core::{LlmError, error, streaming, text, traits, types};

/// Explicit compatibility surface for legacy generic-client paths.
pub mod compat {
    /// Generic client compatibility imports.
    pub mod client {
        pub use siumai_core::compat::client::{ClientWrapper, LlmClient};
    }
}

// Internal aliases for registry implementation (not part of the public API).
#[allow(unused_imports)]
pub(crate) use siumai_provider_utils as provider_utils;

#[allow(unused_imports)]
pub(crate) use siumai_core::{
    auth, compat as core_compat, core, defaults, embedding, execution, image, observability,
    params, retry, retry_api, utils, video,
};

/// Experimental low-level APIs (advanced use only).
///
/// This module exposes lower-level building blocks from `siumai-core` without
/// making them part of the stable surface of `siumai-registry`.
pub mod experimental {
    pub use siumai_core::core::*;
    pub use siumai_core::{
        auth, compat as core_compat, core, defaults, execution, observability, params, retry, utils,
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
