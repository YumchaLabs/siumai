//! Retry Mechanisms
//!
//! Curated re-exports for retry helpers to avoid ambiguous glob exports.

// Public facade (recommended)
pub use crate::retry_api::{RetryBackend, RetryOptions, classify_http_error, retry, retry_with};

// Selected core types for advanced usage
pub use crate::retry::{BackoffRetryExecutor, RetryPolicy, retry_with_backoff};
