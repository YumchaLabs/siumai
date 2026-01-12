//! siumai-provider-amazon-bedrock
//!
//! Amazon Bedrock provider implementation for siumai.
//!
//! This crate mirrors the Vercel AI SDK `@ai-sdk/amazon-bedrock` granularity.
#![deny(unsafe_code)]

pub use siumai_core::{
    LlmError, auth, client, core, defaults, error, execution, hosted_tools, observability, retry,
    retry_api, streaming, traits, types, utils,
};

/// Builder utilities shared across provider crates.
pub mod builder {
    pub use siumai_core::builder::*;
}

pub mod standards;
