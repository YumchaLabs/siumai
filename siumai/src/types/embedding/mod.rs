//! Embedding Types and Structures
//!
//! This module defines all types related to text embedding functionality,
//! organized into submodules for better maintainability.
//!
//! ## Module Organization
//!
//! - **`request`** - Request types (`EmbeddingRequest`, `BatchEmbeddingRequest`)
//! - **`response`** - Response types (`EmbeddingResponse`, `BatchEmbeddingResponse`)
//! - **`common`** - Common types (formats, task types, usage, model info)
//!
//! ## Usage
//!
//! All types are re-exported at the module root for convenience:
//!
//! ```rust
//! use siumai::types::embedding::{EmbeddingRequest, EmbeddingResponse};
//! ```

pub mod common;
pub mod request;
pub mod response;

// Re-export all types for convenience
pub use common::{EmbeddingFormat, EmbeddingModelInfo, EmbeddingTaskType, EmbeddingUsage};
pub use request::{BatchEmbeddingRequest, BatchOptions, EmbeddingRequest};
pub use response::{BatchEmbeddingResponse, EmbeddingResponse};

