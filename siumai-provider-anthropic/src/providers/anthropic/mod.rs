//! Anthropic Provider Module
//!
//! Modular implementation of Anthropic Claude API client with capability separation.

pub mod builder;
pub mod cache;
pub mod client;
pub mod config;
/// Anthropic extension APIs (non-unified surface)
pub mod ext;
pub mod message_batches;
pub mod middleware;
pub mod model_constants;
pub mod models;
pub mod spec;
pub mod streaming;
pub mod thinking;
pub mod tokens;
pub mod transformers;
pub mod types;
pub mod utils;

// Re-export main types for backward compatibility
pub use builder::AnthropicBuilder;
pub use client::AnthropicClient;
pub use config::*;
pub use message_batches::{
    AnthropicCreateMessageBatchRequest, AnthropicListMessageBatchesResponse, AnthropicMessageBatch,
    AnthropicMessageBatchRequest, AnthropicMessageBatches,
};
pub use middleware::AnthropicToolWarningsMiddleware;
pub use tokens::{AnthropicCountTokensResponse, AnthropicTokens};
pub use types::*;

// Provider-owned typed options (kept out of `siumai-core`).
pub use crate::provider_options::anthropic::{
    AnthropicCacheControl, AnthropicCacheType, AnthropicOptions, AnthropicResponseFormat,
    PromptCachingConfig, ThinkingModeConfig,
};

// Typed provider metadata views (protocol-owned; re-exported via this provider for ergonomics).
pub use crate::provider_metadata::anthropic::{
    AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock, AnthropicMetadata,
    AnthropicServerToolUse, AnthropicSource,
};

// Re-export capability implementations
pub use models::AnthropicModels;
