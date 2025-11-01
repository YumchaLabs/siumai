//! `MiniMaxi` Provider Module
//!
//! Modular implementation of MiniMaxi API client with multi-modal capabilities.
//! This module follows the design pattern of separating different AI capabilities
//! into distinct modules while providing a unified client interface.
//!
//! MiniMaxi provides multiple AI capabilities:
//! - Text generation (M2 model) - OpenAI and Anthropic compatible
//! - Speech synthesis (Speech 2.6 HD/Turbo)
//! - Image generation (image-01, image-01-live)
//! - Video generation (Hailuo 2.3 & 2.3 Fast)
//! - Music generation (Music 2.0)
//!
//! # Architecture
//! - `client.rs` - Main MiniMaxi client that aggregates all capabilities
//! - `config.rs` - Configuration structures and validation
//! - `builder.rs` - Builder pattern implementation for client creation
//! - `chat.rs` - Chat completion capability implementation
//! - `audio.rs` - Audio (TTS) capability implementation
//! - `image.rs` - Image generation capability implementation
//! - `spec.rs` - ProviderSpec implementation (uses OpenAI standard)
//! - `types.rs` - MiniMaxi-specific type definitions
//! - `model_constants.rs` - Model name constants
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::models;
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmBuilder::new()
//!         .minimaxi()
//!         .api_key("your-api-key")
//!         .model("MiniMax-M2")
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
pub mod builder;
pub mod client;
pub mod config;
pub mod spec;
pub mod transformers;
pub mod types;

// Capability modules
pub mod audio;
pub mod chat;
pub mod image;
pub mod model_constants;
pub mod music;
pub mod video;

// Re-export main types for convenience
pub use builder::MinimaxiBuilder;
pub use client::MinimaxiClient;
pub use config::MinimaxiConfig;
pub use spec::MinimaxiSpec;
pub use types::*;

// Re-export chat capability implementation
pub use chat::MinimaxiChatCapability;

// Tests module
#[cfg(test)]
mod tests;
