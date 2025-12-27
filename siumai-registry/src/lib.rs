//! siumai-registry
//!
//! Provider registry, factories, and handles.
#![deny(unsafe_code)]

pub use siumai_core::{
    LlmError, auth, client, core, custom_provider, defaults, error, execution, hosted_tools,
    observability, params, retry, retry_api, standards, streaming, traits, types, utils,
};

pub use siumai_providers::builder;
pub use siumai_providers::providers;
pub use siumai_providers::{constants, models};

pub mod provider;
pub mod provider_builders;
pub mod provider_catalog;
pub mod registry;
