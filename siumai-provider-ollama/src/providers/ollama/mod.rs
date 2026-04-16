//! Ollama Provider Module
//!
//! Modular implementation of Ollama API client with capability separation.
//! This module follows the design pattern of separating different AI capabilities
//! into distinct modules while providing a unified client interface.
//!
//! # Architecture
//! - `client.rs` - Main Ollama client that aggregates all capabilities
//! - `config.rs` - Configuration structures and validation
//! - `chat.rs` - Chat completion capability implementation
//! - `embeddings.rs` - Text embedding capability implementation
//! - `model_listing.rs` - Remote model management capability implementation
//! - `models.rs` - Curated model constants for the public facade/catalog
//! - `types.rs` - Ollama-specific type definitions
//! - `utils.rs` - Utility functions and helpers
//! - `streaming.rs` - Streaming functionality with line buffering
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmBuilder::new()
//!         .ollama()
//!         .base_url("http://localhost:11434")
//!         .model("llama3.2")
//!         .build()
//!         .await?;
//!
//!     // Use chat capability
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!
//!     // Use embedding capability
//!     let embeddings = client.embed(vec!["Hello, world!".to_string()]).await?;
//!
//!     Ok(())
//! }
//! ```

// Core modules
pub mod client;
pub mod config;
/// Ollama extension APIs (non-unified surface)
pub mod ext;
pub mod model_constants;
pub mod model_listing;
pub mod spec;
pub mod transformers;
pub mod types;
pub mod utils;

// Capability modules
pub mod builder;
pub mod chat;
pub mod embeddings;
pub mod models;
pub mod streaming;

// Re-export main types
pub use builder::OllamaBuilder;
pub use client::OllamaClient;
pub use config::{OllamaConfig, OllamaConfigBuilder};
pub use types::*;

/// Default Ollama models
pub fn get_default_models() -> Vec<String> {
    models::all_models()
        .into_iter()
        .map(str::to_string)
        .collect()
}
