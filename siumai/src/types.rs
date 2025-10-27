//! Core Data Type Definitions
//!
//! This module contains all data structures used in the LLM library, organized by functionality.
//!
//! ## Module Organization
//!
//! - **`chat/`** - Chat-related types (messages, requests, responses, content)
//! - **`common`** - Common parameters and types shared across providers
//! - **`embedding`** - Embedding request/response types
//! - **`image`** - Image generation types
//! - **`audio`** - Audio transcription/generation types
//! - **`tools`** - Tool/function calling types
//! - **`streaming`** - Streaming response types
//! - **`provider_options/`** - Provider-specific configuration options
//! - **`provider_metadata/`** - Provider-specific response metadata
//!
//! ## Usage Guidelines
//!
//! ### For Application Developers
//!
//! Most types are re-exported at the module root for convenience:
//!
//! ```rust
//! use siumai::types::{ChatMessage, ChatRequest, ChatResponse, CommonParams};
//! ```
//!
//! Or use the prelude for common types:
//!
//! ```rust
//! use siumai::prelude::*;
//! ```
//!
//! ### For Library Developers
//!
//! When adding new types:
//! - **Common types** → `common.rs`
//! - **Chat-related** → `chat/` subdirectory
//! - **Provider-specific options** → `provider_options/<provider>.rs`
//! - **Provider-specific metadata** → `provider_metadata/<provider>.rs`
//!
//! ## Type Categories
//!
//! ### Request Types
//! - `ChatRequest` - Chat completion requests
//! - `EmbeddingRequest` - Embedding generation requests
//! - `ImageGenRequest` - Image generation requests
//! - `AudioRequest` - Audio transcription/generation requests
//!
//! ### Response Types
//! - `ChatResponse` - Chat completion responses
//! - `EmbeddingResponse` - Embedding vectors
//! - `ImageResponse` - Generated images
//! - `AudioResponse` - Audio transcription/generation results
//!
//! ### Common Types
//! - `CommonParams` - Parameters shared across all providers (temperature, max_tokens, etc.)
//! - `Usage` - Token usage information
//! - `FinishReason` - Completion finish reasons
//!
//! ### Provider-Specific Types
//! - `OpenAiOptions` - OpenAI-specific options
//! - `AnthropicOptions` - Anthropic-specific options
//! - `GeminiOptions` - Gemini-specific options
//! - etc.

pub mod audio;
pub mod chat;
pub mod common;
pub mod completion;
pub mod embedding;
pub mod files;
pub mod image;
pub mod models;
pub mod moderation;
pub mod provider_metadata;
pub mod provider_options;
pub mod rerank;
pub mod schema;
pub mod streaming;
pub mod tools;
pub mod web_search;

// Re-export all types for backward compatibility
pub use audio::*;
pub use chat::*;
pub use common::*;
pub use completion::*;
pub use embedding::*;
pub use files::*;
pub use image::*;
pub use models::*;
pub use moderation::*;
pub use provider_options::*;
pub use rerank::*;
pub use schema::*;
pub use streaming::*;
pub use tools::*;
pub use web_search::*;

// Re-export provider metadata types (not wildcard to avoid conflicts)
pub use provider_metadata::{AnthropicMetadata, GeminiMetadata, OpenAiMetadata};
