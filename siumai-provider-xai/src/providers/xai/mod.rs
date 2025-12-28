//! `xAI` Provider Module
//!
//! Modular implementation of `xAI` API client with capability separation.
//! This module follows the design pattern of separating different AI capabilities
//! into distinct modules while providing a unified client interface.
//!
//! # Architecture
//! - `client.rs` - Main `xAI` client that aggregates all capabilities
//! - `config.rs` - Configuration structures and validation
//! - `builder.rs` - Builder pattern implementation for client creation
//! - `chat.rs` - Chat completion capability implementation
//! - `types.rs` - xAI-specific type definitions
//! - `api.rs` - Model listing/retrieval capability
//! - `spec.rs` - ProviderSpec wiring (delegates protocol to `standards::openai`)
//! - `models.rs` - Built-in model catalog (fallback)
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::models;
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Provider::xai()
//!         .api_key("your-api-key")
//!         .model(models::xai::GROK_3_LATEST)
//!         .build()
//!         .await?;
//!
//!     // Use chat capability
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!
//!     Ok(())
//! }
//! ```

// Core modules
pub mod api;
pub mod builder;
pub mod client;
pub mod config;
pub mod models;
pub mod spec;
pub mod types;

// Capability modules
pub mod chat;

// Re-export main types for convenience
pub use api::XaiModels;
pub use builder::XaiBuilder;
pub use client::XaiClient;
pub use config::XaiConfig;
pub use types::*;

// Re-export capability implementations
pub use chat::XaiChatCapability;
