//! Anthropic API Standard
//!
//! This module implements the Anthropic Messages API format.
//!
//! ## Supported Providers
//!
//! - Anthropic (native)
//! - DeepSeek (via Anthropic-compatible endpoint)
//! - Some proxy services
//!
//! ## Capabilities
//!
//! - Messages API (Chat)
//! - Prompt Caching (Anthropic-specific extension)
//! - Thinking Mode (Anthropic-specific extension)

pub mod cache;
pub mod chat;
pub mod errors;
pub mod streaming;
pub mod thinking;
pub mod transformers;
pub mod types;
pub mod utils;

#[cfg(test)]
mod chat_adapter_sse_tests;

// Re-export main types
pub use chat::{AnthropicChatAdapter, AnthropicChatStandard};
