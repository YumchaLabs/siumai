//! `MiniMax` Provider Module
//!
//! Modular implementation of MiniMax API client with multi-modal capabilities.
//! This module follows the design pattern of separating different AI capabilities
//! into distinct modules while providing a unified client interface.
//!
//! MiniMax provides multiple AI capabilities:
//! - Text generation (M3 and M2 family) - native, OpenAI, and Anthropic compatible
//! - Speech synthesis (Speech 2.8/2.6 HD and Turbo)
//! - Image generation (image-01, image-01-live)
//! - Video generation (Hailuo 2.3 & 2.3 Fast)
//! - Music generation (Music 2.6 and cover variants)
//!
//! # Architecture
//! - `client.rs` - Main MiniMax client that aggregates all capabilities
//! - `config.rs` - Configuration structures and validation
//! - `builder.rs` - Builder pattern implementation for client creation
//! - `audio.rs` - Audio (TTS) capability implementation
//! - `image.rs` - Image generation capability implementation
//! - `spec.rs` - ProviderSpec implementation (chat uses Anthropic standard)
//! - `types.rs` - MiniMax-specific type definitions
//! - `models.rs` - Curated model-family constants for facade/catalog alignment
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai_core::{
//!     builder::BuilderBase,
//!     traits::ChatCapability,
//!     types::{ChatMessage, ChatRequest},
//! };
//! use siumai_provider_minimax::providers::minimax::{
//!     MinimaxBuilder,
//!     models::chat::MINIMAX_M3,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = MinimaxBuilder::new(BuilderBase::default())
//!         .api_key("your-api-key")
//!         .model(MINIMAX_M3)
//!         .build().await?;
//!
//!     let request = ChatRequest::new(vec![ChatMessage::user("Hello, world!").build()]);
//!     let response = client.chat_request(request).await?;
//!
//!     Ok(())
//! }
//! ```

// Core modules
pub mod builder;
pub mod client;
pub mod config;
/// MiniMax extension APIs (non-unified surface)
pub mod ext;
pub mod files;
pub mod spec;
pub mod transformers;
pub mod types;
mod utils;

// Capability modules
pub mod audio;
pub mod image;
pub mod models;
pub mod music;
pub mod video;

// Re-export main types for convenience
pub use crate::provider_options::{
    MinimaxOptions, MinimaxServiceTier, MinimaxThinking, MinimaxTtsOptions, MinimaxVideoOptions,
};
pub use builder::MinimaxBuilder;
pub use client::MinimaxClient;
pub use config::MinimaxConfig;
pub use files::MinimaxFiles;
pub use spec::MinimaxSpec;
pub use types::*;

// Typed provider metadata views (provider-owned; re-exported via this provider for ergonomics).
pub use crate::provider_metadata::minimax::{
    MinimaxChatResponseExt, MinimaxCitation, MinimaxCitationsBlock, MinimaxContentPartExt,
    MinimaxMetadata, MinimaxServerToolUse, MinimaxSource, MinimaxToolCallMetadata,
    MinimaxToolCaller,
};

// Re-export chat capability implementation
// (Removed) `MinimaxChatCapability`: chat is implemented directly on `MinimaxClient`.

// Tests module
#[cfg(test)]
mod tests;
