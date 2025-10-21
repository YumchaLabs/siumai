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

pub mod chat;

// Re-export main types
pub use chat::{AnthropicChatAdapter, AnthropicChatStandard};
