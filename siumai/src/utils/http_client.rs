//! HTTP Client Utilities
//!
//! **DEPRECATED**: This module has been moved to `crate::execution::http::client`.
//! This re-export is maintained for backward compatibility.
//!
//! Please update your imports:
//! ```rust,ignore
//! // Old (deprecated)
//! use siumai::utils::http_client::*;
//!
//! // New (recommended)
//! use siumai::execution::http::client::*;
//! ```

pub use crate::execution::http::client::*;
