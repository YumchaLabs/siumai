//! Anthropic API Standard
//!
//! This module implements the Anthropic Messages API format.
//!
//! Note: MiniMaxi uses Anthropic-style chat mapping, but this module is kept
//! provider-owned (inside `siumai-provider-minimaxi`) to avoid provider→provider
//! dependencies during the alpha.5 split-crate refactor.

pub mod cache;
pub mod chat;
pub mod errors;
pub mod params;
pub mod provider_metadata;
pub mod streaming;
pub mod thinking;
pub mod transformers;
pub mod types;
pub mod utils;

#[cfg(test)]
mod chat_adapter_sse_tests;

pub use chat::{AnthropicChatAdapter, AnthropicChatStandard};
