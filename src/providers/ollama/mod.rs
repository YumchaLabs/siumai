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
//! - `completion.rs` - Text completion capability implementation
//! - `embeddings.rs` - Text embedding capability implementation
//! - `models.rs` - Model management capability implementation
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
pub mod model_constants;
pub mod transformers;
pub mod types;
pub mod utils;

// Capability modules
pub mod builder;
pub mod chat;
pub mod completion;
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
    vec![
        "llama3.2:latest".to_string(),
        "llama3.2:3b".to_string(),
        "llama3.2:1b".to_string(),
        "llama3.1:latest".to_string(),
        "llama3.1:8b".to_string(),
        "llama3.1:70b".to_string(),
        "mistral:latest".to_string(),
        "mistral:7b".to_string(),
        "codellama:latest".to_string(),
        "codellama:7b".to_string(),
        "codellama:13b".to_string(),
        "codellama:34b".to_string(),
        "phi3:latest".to_string(),
        "phi3:mini".to_string(),
        "phi3:medium".to_string(),
        "gemma:latest".to_string(),
        "gemma:2b".to_string(),
        "gemma:7b".to_string(),
        "qwen2:latest".to_string(),
        "qwen2:0.5b".to_string(),
        "qwen2:1.5b".to_string(),
        "qwen2:7b".to_string(),
        "qwen2:72b".to_string(),
        "deepseek-coder:latest".to_string(),
        "deepseek-coder:6.7b".to_string(),
        "deepseek-coder:33b".to_string(),
        "nomic-embed-text:latest".to_string(),
        "all-minilm:latest".to_string(),
    ]
}
