//! Error Handling Module
//!
//! This module provides comprehensive error handling for the LLM library, including:
//! - Core error types (`LlmError`, `ErrorCategory`)
//! - User-facing error helpers and summaries
//! - Type conversions from common error types
//!
//! # Example
//!
//! ```rust,ignore
//! use siumai::error::{LlmError, ErrorCategory};
//!
//! let error = LlmError::api_error(404, "Not found");
//! assert_eq!(error.category(), ErrorCategory::Client);
//! assert!(!error.is_retryable());
//! ```

// Module declarations
mod conversions;
pub mod helpers;
pub mod types;

// Re-exports for public API
pub use helpers::*;
pub use types::*;
