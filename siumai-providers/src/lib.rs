//! siumai-providers
//!
//! Provider implementations and client construction helpers.
#![deny(unsafe_code)]

#[macro_use]
mod macros;

// Keep a small stable surface; avoid leaking provider-agnostic internals by default.
pub use siumai_core::{LlmError, custom_provider, error, hosted_tools, retry_api, streaming, traits, types};

// Compatibility / internal re-exports (hidden to reduce accidental coupling).
#[doc(hidden)]
pub use siumai_core::{auth, client, defaults, execution, observability, params, retry, utils};

#[doc(hidden)]
pub mod standards;

pub mod builder;
#[doc(hidden)]
pub mod core;
pub mod model_catalog;
pub mod providers;

pub use model_catalog::constants;
pub use model_catalog::model_constants as models;

pub use builder::LlmBuilder;

pub use types::{ChatResponse, CommonParams};
