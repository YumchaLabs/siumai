//! Siumai Anthropic Standard
//!
//! This crate hosts standard request/response/streaming transformers and
//! adapters for the Anthropic Messages API. It acts as a standard layer on
//! top of `siumai-core` and does not depend on concrete provider implementations.

pub const VERSION: &str = "0.0.1";

pub mod anthropic;

pub use anthropic::chat::{
    AnthropicChatAdapter, AnthropicChatStandard, AnthropicDefaultChatAdapter,
};
