//! Core Data Type Definitions
//!
//! This module contains all data structures used in the LLM library, organized by
//! functionality. The public API is surfaced from this root module; internal
//! submodules like `params`, `http`, and `usage` are implementation details.
//!
//! ## Module Organization
//!
//! - **`chat/`** - Chat-related types (messages, requests, responses, content)
//! - **`common`** - Common enums/metadata shared across providers
//! - **`params`** - Common AI parameters (model, temperature, max_tokens, etc.)
//! - **`http`** - HTTP configuration (`HttpConfig` and builder)
//! - **`usage`** - Token usage and detailed usage breakdown
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
//! - **Common enums/metadata** → `common.rs`
//! - **Shared AI parameters** → `params.rs`
//! - **HTTP configuration** → `http.rs`
//! - **Usage accounting** → `usage.rs`
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
//! - `CommonParams` - Parameters shared across all providers
//! - `HttpConfig` - HTTP configuration shared across providers
//! - `Usage` - Token usage information
//! - `FinishReason` - Completion finish reasons
//! - `ProviderType` - Provider identifier enum
//! - `ResponseMetadata` - Shared response metadata
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
pub mod http;
pub mod image;
pub mod models;
pub mod moderation;
pub mod music;
pub mod params;
pub mod provider_metadata;
pub mod provider_options;
pub mod rerank;
pub mod schema;
pub mod streaming;
pub mod tools;
pub mod usage;
pub mod video;

// Re-export all types for convenience
pub use audio::*;
pub use chat::*;
pub use common::*;
pub use completion::*;
pub use embedding::*;
pub use files::*;
pub use http::*;
pub use image::*;
pub use models::*;
pub use moderation::*;
pub use music::*;
pub use params::*;
pub use provider_options::*;
pub use rerank::*;
pub use schema::*;
pub use streaming::*;
pub use tools::*;
pub use usage::*;
pub use video::*;

// Re-export provider metadata types (not wildcard to avoid conflicts)
pub use provider_metadata::{AnthropicMetadata, GeminiMetadata, OpenAiMetadata};
