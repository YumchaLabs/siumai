//! HTTP Header Utilities
//!
//! **DEPRECATED**: This module has been moved to `crate::execution::http::headers`.
//! This re-export is maintained for backward compatibility.
//!
//! Please update your imports:
//! ```rust,ignore
//! // Old (deprecated)
//! use siumai::utils::http_headers::*;
//!
//! // New (recommended)
//! use siumai::execution::http::headers::*;
//! ```

pub use crate::execution::http::headers::*;
