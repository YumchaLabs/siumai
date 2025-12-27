//! siumai-providers
//!
//! Provider implementations and client construction helpers.
#![deny(unsafe_code)]

#[macro_use]
mod macros;

pub use siumai_core::{
    LlmError, auth, client, custom_provider, defaults, error, execution, hosted_tools,
    observability, params, retry, retry_api, standards, streaming, traits, types, utils,
};

pub mod builder;
pub mod core;
pub mod model_catalog;
pub mod providers;

pub use model_catalog::constants;
pub use model_catalog::model_constants as models;

pub use builder::LlmBuilder;

pub use types::{ChatResponse, CommonParams};
