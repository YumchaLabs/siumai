//! Vercel AI Gateway provider.

pub use siumai_core::{
    builder, compat as core_compat, core, defaults, embedding, error, execution, retry_api,
    streaming, text, traits, types,
};
pub use siumai_provider_utils as provider_utils;

pub mod provider_options;
pub mod providers;

pub use provider_options::*;
pub use providers::gateway::*;
