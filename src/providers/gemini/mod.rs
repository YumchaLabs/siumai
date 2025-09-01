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
//! - `embeddings.rs` - Text embedding capability implementation
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
pub mod embeddings;
pub mod files;
pub mod model_constants;
pub mod models;
pub mod request;
pub mod streaming;
pub mod types;

// Feature modules
pub mod builder;
pub mod code_execution;

// Re-export main types for convenience
pub use builder::GeminiBuilder;
pub use chat::GeminiChatCapability;
pub use client::GeminiClient;
pub use embeddings::GeminiEmbeddings;
pub use files::GeminiFiles;
pub use models::GeminiModels;
pub use types::*;
