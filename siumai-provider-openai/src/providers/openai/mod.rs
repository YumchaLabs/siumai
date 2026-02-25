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
//! - Chat/Embedding/Image/Audio are executed via Executors + Transformers (client.rs)
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
pub mod adapter;
pub mod builder;
pub mod client;
pub mod config;
pub mod types;
pub mod utils;

#[cfg(test)]
mod thinking_utils_test;

// Capability modules
pub mod files;
pub mod rerank;
// NOTE: Expose Responses event converter publicly for fixture-driven tests and
// compatibility checks. This keeps the client surface unchanged while allowing
// integration tests to validate SSE parsing behavior directly.
/// OpenAI extension APIs (non-unified surface)
pub mod ext;
pub mod responses;
pub mod structured_output;
pub mod transformers;
// pub mod streaming; // removed after test migration to compat converter

// Request building module (removed; Transformers handle mapping/validation)

// Future capability modules (placeholders)
#[cfg(feature = "openai-websocket")]
pub mod incremental_session;
pub mod middleware;
pub mod models;
pub mod moderation;
pub mod spec;
#[cfg(feature = "openai-websocket")]
pub mod websocket_session;
#[cfg(feature = "openai-websocket")]
pub mod websocket_transport;

// Model constants module
pub mod model_constants;

// Re-export main types for convenience
pub use builder::OpenAiBuilder;
pub use client::OpenAiClient;
pub use config::OpenAiConfig;
#[cfg(feature = "openai-websocket")]
pub use incremental_session::OpenAiIncrementalWebSocketSession;
pub use middleware::OpenAiResponsesInputWarningsMiddleware;
pub use types::*;
#[cfg(feature = "openai-websocket")]
pub use websocket_session::{OpenAiWebSocketRecoveryConfig, OpenAiWebSocketSession};
#[cfg(feature = "openai-websocket")]
pub use websocket_transport::OpenAiWebSocketTransport;

// Provider-owned typed options (kept out of `siumai-core`).
pub use crate::provider_options::openai::{
    ChatCompletionAudio, ChatCompletionAudioFormat, ChatCompletionAudioVoice,
    ChatCompletionModalities, InputAudio, InputAudioFormat, OpenAiOptions, OpenAiWebSearchOptions,
    PredictionContent, PredictionContentData, ReasoningEffort, ResponsesApiConfig, ServiceTier,
    TextVerbosity, Truncation, UserLocationWrapper, WebSearchLocation,
};

// Typed provider metadata views (protocol-owned; re-exported via this provider for ergonomics).
pub use crate::provider_metadata::openai::{OpenAiChatResponseExt, OpenAiMetadata, OpenAiSource};

/// OpenAI-compatible vendors (OpenAI-like providers).
///
/// This is a thin re-export layer over `providers::openai_compatible` to keep the public
/// mental model "OpenAI is the protocol family" while vendors are configuration/presets.
#[cfg(feature = "openai")]
pub mod vendors;

// Re-export capability implementations
pub use files::OpenAiFiles;
pub use models::OpenAiModels;
pub use moderation::OpenAiModeration;
pub use rerank::OpenAiRerank;
// Responses API client/types are no longer re-exported; use unified OpenAiClient

// Re-export parameter enums for convenience
pub use crate::params::openai::{IncludableItem, SortOrder, TruncationStrategy};

// Test modules
#[cfg(test)]
mod tests {
    pub mod thinking_priority_tests;
}
