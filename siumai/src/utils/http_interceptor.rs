//! HTTP Interceptor Utilities
//!
//! **DEPRECATED**: This module has been moved to `crate::execution::http::interceptor`.
//! This re-export is maintained for backward compatibility.
//!
//! Please update your imports:
//! ```rust,ignore
//! // Old (deprecated)
//! use siumai::utils::http_interceptor::*;
//!
//! // New (recommended)
//! use siumai::execution::http::interceptor::*;
//! ```

pub use crate::execution::http::interceptor::*;
