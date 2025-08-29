//! `OpenAI` Provider Module
//!
//! Modular implementation of `OpenAI` API client with capability separation.
//! This module follows the design pattern of separating different AI capabilities
//! into distinct modules while providing a unified client interface.
//!
//! # Architecture
//! - `client.rs` - Main `OpenAI` client that aggregates all capabilities
//! - `config.rs` - Configuration structures and validation
//! - `builder.rs` - Builder pattern implementation for client creation
//! - `chat.rs` - Chat completion capability implementation
//! - `audio.rs` - Audio processing (TTS/STT) capability implementation
//! - `embeddings.rs` - Text embedding capability implementation
//! - `images.rs` - Image generation capability implementation
//! - `files.rs` - File management capability implementation
//! - `models.rs` - Model listing capability implementation (future)
//! - `moderation.rs` - Content moderation capability implementation
//! - `types.rs` - OpenAI-specific type definitions
//! - `utils.rs` - Utility functions and helpers
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmBuilder::new()
//!         .openai()
//!         .api_key("your-api-key")
//!         .model("gpt-4")
//!         .build()
//!         .await?;
//!
//!     // Use chat capability
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!
//!     // Use audio capability (if available)
//!     // let audio_data = client.speech("Hello, world!").await?;
//!
//!     // Use embedding capability (if available)
//!     // let embeddings = client.embed(vec!["Hello, world!".to_string()]).await?;
//!
//!     Ok(())
//! }
//! ```

// Core modules
pub mod builder;
pub mod client;
pub mod config;
pub mod types;
pub mod utils;

#[cfg(test)]
mod thinking_utils_test;

// Capability modules
pub mod audio;
pub mod chat;
pub mod embeddings;
pub mod files;
pub mod images;
pub mod rerank;
pub mod responses;
pub mod streaming;
pub mod structured_output;

// Request building module
pub mod request;

// Future capability modules (placeholders)
pub mod models;
pub mod moderation;

// Model constants module
pub mod model_constants;

// Re-export main types for convenience
pub use builder::OpenAiBuilder;
pub use client::OpenAiClient;
pub use config::OpenAiConfig;
pub use types::*;

// Re-export capability implementations
pub use audio::OpenAiAudio;
pub use chat::OpenAiChatCapability;
pub use embeddings::OpenAiEmbeddings;
pub use files::OpenAiFiles;
pub use images::OpenAiImages;
pub use models::OpenAiModels;
pub use moderation::OpenAiModeration;
pub use rerank::OpenAiRerank;
pub use responses::{
    ListResponsesQuery, OpenAiResponses, ResponseMetadata, ResponseStatus, ResponsesApiCapability,
};

// Re-export parameter enums for convenience
pub use crate::params::openai::{IncludableItem, SortOrder, TruncationStrategy};
