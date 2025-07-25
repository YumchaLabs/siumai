//! `Groq` Provider Module
//!
//! Modular implementation of `Groq` API client with capability separation.
//! This module follows the design pattern of separating different AI capabilities
//! into distinct modules while providing a unified client interface.
//!
//! Groq provides OpenAI-compatible API endpoints with high-performance inference.
//!
//! # Architecture
//! - `client.rs` - Main `Groq` client that aggregates all capabilities
//! - `config.rs` - Configuration structures and validation
//! - `builder.rs` - Builder pattern implementation for client creation
//! - `chat.rs` - Chat completion capability implementation
//! - `audio.rs` - Audio processing (TTS/STT) capability implementation
//! - `files.rs` - File management capability implementation
//! - `models.rs` - Model listing capability implementation
//! - `types.rs` - Groq-specific type definitions
//! - `utils.rs` - Utility functions and helpers
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmBuilder::new()
//!         .groq()
//!         .api_key("your-api-key")
//!         .model("llama-3.3-70b-versatile")
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
//!     Ok(())
//! }
//! ```

// Core modules
pub mod builder;
pub mod client;
pub mod config;
pub mod types;
pub mod utils;

// Capability modules
pub mod audio;
pub mod chat;
pub mod files;
pub mod models;
pub mod streaming;

// Re-export main types for convenience
pub use builder::GroqBuilder;
pub use client::GroqClient;
pub use config::GroqConfig;
pub use types::*;

// Re-export capability implementations
pub use audio::GroqAudio;
pub use chat::GroqChatCapability;
pub use files::GroqFiles;
pub use models::GroqModels;

// Tests module
#[cfg(test)]
mod tests;
