//! Execution Layer
//!
//! This module contains all execution-related components:
//! - **executor**: HTTP executors for different capabilities (chat, embedding, etc.)
//! - **transformer**: Request/response/stream transformers
//! - **middleware**: Model-level parameter transformation
//! - **http**: HTTP utilities (client, headers, interceptors, retry)
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

pub mod executor;
pub mod http;
pub mod middleware;
pub mod transformer;

// Re-export main types for convenience
pub use executor::*;
pub use http::*;
pub use middleware::*;
pub use transformer::*;
