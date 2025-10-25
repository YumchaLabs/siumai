//! Execution Layer - Unified Public API
//!
//! **This is the recommended entry point for execution-related types.**
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use siumai::execution::{
//!     executors::chat::HttpChatExecutor,
//!     transformers::request::RequestTransformer,
//!     middleware::LanguageModelMiddleware,
//!     http::HttpInterceptor,
//! };
//! ```
//!
//! ## Module Organization
//!
//! - **`executors`** - HTTP executors for different capabilities (chat, embedding, image, etc.)
//!   - Handles HTTP execution with middleware and retry logic
//!   - Provides executor traits and implementations
//!
//! - **`transformers`** - Request/response/stream format conversion
//!   - Provider-agnostic transformation traits
//!   - Converts between unified types and provider-specific formats
//!
//! - **`middleware`** - Model-level parameter transformation
//!   - Pre/post processing hooks for requests and responses
//!   - Middleware chain composition
//!
//! - **`http`** - HTTP utilities
//!   - Client configuration, headers, interceptors, retry policies
//!   - Re-exports from `crate::utils` and `crate::retry`
//!
//! ## Architecture
//!
//! ```text
//! Provider Client
//!     ↓
//! Executor (HTTP execution)
//!     ↓
//! Middleware (parameter transformation)
//!     ↓
//! Transformer (format conversion)
//!     ↓
//! HTTP Client (with interceptors & retry)
//! ```
//!
//! ## For Library Developers
//!
//! This module provides a clean, organized public API. All implementations
//! are contained within this module for better encapsulation.

// Actual implementation modules
pub mod executors;
pub mod http;
pub mod middleware;
pub mod streaming;
pub mod transformers;

// Re-export commonly used types for convenience
pub use executors::{
    BeforeSendHook,
    audio::{AudioExecutor, HttpAudioExecutor},
    chat::{ChatExecutor, ChatExecutorBuilder, HttpChatExecutor},
    embedding::{EmbeddingExecutor, HttpEmbeddingExecutor},
    files::{FilesExecutor, HttpFilesExecutor},
    image::{HttpImageExecutor, ImageExecutor},
    rerank::{HttpRerankExecutor, RerankExecutor},
};

pub use transformers::{
    audio::{AudioHttpBody, AudioTransformer},
    files::{FilesHttpBody, FilesTransformer},
    request::{ImageHttpBody, RequestTransformer},
    rerank_request::RerankRequestTransformer,
    rerank_response::RerankResponseTransformer,
    response::ResponseTransformer,
    stream::StreamChunkTransformer,
};

pub use middleware::{LanguageModelMiddleware, MiddlewareBuilder, NamedMiddleware};
