//! Utility modules for siumai
//!
//! This module contains various utility functions and types used throughout the library.
//!
//! ## Module Organization
//!
//! ### Core Utilities (Recommended)
//! - [`cancel`] - Cancellation token utilities
//! - [`mime`] - MIME type handling
//! - [`url`] - URL utilities
//! - [`utf8_decoder`] - UTF-8 stream decoding
//!
//! ### Deprecated Modules (Use New Locations)
//! - [`http_client`] - **DEPRECATED**: Use [`crate::execution::http::client`] instead
//! - [`http_headers`] - **DEPRECATED**: Use [`crate::execution::http::headers`] instead
//! - [`http_interceptor`] - **DEPRECATED**: Use [`crate::execution::http::interceptor`] instead
//! - [`vertex`] - **DEPRECATED**: Use [`crate::auth::vertex`] instead
//!
//! ## Migration Guide
//!
//! If you're using the deprecated modules, please update your imports:
//!
//! ```rust,ignore
//! // Old (deprecated)
//! use siumai::utils::http_headers::*;
//! use siumai::utils::vertex::*;
//!
//! // New (recommended)
//! use siumai::execution::http::headers::*;
//! use siumai::auth::vertex::*;
//! ```

pub mod cancel;
pub mod mime;
pub mod url;
pub mod utf8_decoder;

// Deprecated modules (re-exports for backward compatibility)
#[deprecated(
    since = "0.11.1",
    note = "Use `crate::execution::http::client` instead"
)]
pub mod http_client;
#[deprecated(
    since = "0.11.1",
    note = "Use `crate::execution::http::headers` instead"
)]
pub mod http_headers;
#[deprecated(
    since = "0.11.1",
    note = "Use `crate::execution::http::interceptor` instead"
)]
pub mod http_interceptor;
#[deprecated(since = "0.11.1", note = "Use `crate::auth::vertex` instead")]
pub mod vertex;

// Re-exports for convenience
pub use mime::*;
pub use url::*;
pub use utf8_decoder::Utf8StreamDecoder;

// Deprecated re-exports
#[allow(deprecated)]
pub use http_client::*;
#[allow(deprecated)]
pub use http_interceptor::*;
#[allow(deprecated)]
pub use vertex::*;
