//! Google Gemini Provider
//!
//! This module provides integration with Google's Gemini AI models.
//!
//! # Architecture
//! - `client.rs` - Main Gemini client that aggregates all capabilities
//! - `types.rs` - Gemini-specific type definitions based on `OpenAPI` spec
//! - `chat.rs` - Chat completion capability implementation
//! - `models.rs` - Model listing capability implementation
//! - `files.rs` - File management capability implementation
//! - `code_execution.rs` - Code execution feature implementation
//! - `streaming.rs` - Streaming functionality with JSON buffering
//! - Embeddings are executed via Executors + Transformers (no standalone HTTP module)
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmBuilder::new()
//!         .gemini()
//!         .api_key("your-api-key")
//!         .model("gemini-1.5-flash")
//!         .build()
//!         .await?;
//!
//!     // Use chat capability
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat_with_tools(messages, None).await?;
//!
//!     Ok(())
//! }
//! ```

// Core modules
pub mod chat;
pub mod client;
pub mod config;
pub mod context;
pub mod convert;
/// Gemini extension APIs (non-unified surface)
pub mod ext;
pub mod file_search_stores;
pub mod files;
pub mod middleware;
pub mod model_constants;
pub mod models;
pub mod spec;
pub mod streaming;
pub mod transformers;
pub mod types;

// Feature modules
pub mod builder;
pub mod cached_contents;
pub mod code_execution;
pub mod tokens;
pub mod video;

// Re-export main types for convenience
pub use builder::GeminiBuilder;
pub use cached_contents::GeminiCachedContents;
pub use chat::GeminiChatCapability;
pub use client::GeminiClient;
pub use file_search_stores::GeminiFileSearchStores;
pub use files::GeminiFiles;
pub use middleware::GeminiToolWarningsMiddleware;
pub use models::GeminiModels;
pub use tokens::{GeminiCountTokensResponse, GeminiTokens};
pub use types::*;
pub use video::GeminiVideo;
