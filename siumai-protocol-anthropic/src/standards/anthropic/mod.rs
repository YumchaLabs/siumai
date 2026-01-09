//! Anthropic Messages API Standard
//!
//! This module implements the Anthropic Messages API format.
//!
//! This crate is a shared protocol layer used by multiple providers during the
//! alpha.5 split-crate refactor to avoid provider→provider dependencies.

pub mod cache;
pub mod chat;
pub mod errors;
pub mod params;
pub mod provider_metadata;
pub mod server_tools;
pub mod streaming;
pub mod thinking;
pub mod transformers;
pub mod types;
pub mod utils;

#[cfg(test)]
mod chat_adapter_sse_tests;

pub use chat::{AnthropicChatAdapter, AnthropicChatStandard};
