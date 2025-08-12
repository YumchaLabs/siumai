//! Anthropic Provider Module
//!
//! Modular implementation of Anthropic Claude API client with capability separation.

pub mod cache;
pub mod chat;
pub mod client;
pub mod models;
pub mod request;
pub mod streaming;
pub mod thinking;
pub mod types;
pub mod utils;

// Re-export main types for backward compatibility
pub use client::AnthropicClient;
pub use types::*;

// Re-export capability implementations
pub use chat::AnthropicChatCapability;
pub use models::AnthropicModels;
