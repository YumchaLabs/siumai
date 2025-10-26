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
//! - [`vertex`] - **DEPRECATED**: Use [`crate::auth::vertex`] instead
//!
//! ## Migration Guide
//!
//! If you're using the deprecated modules, please update your imports:
//!
//! ```rust,ignore
//! // Old (deprecated)
//! use siumai::utils::vertex::*;
//!
//! // New (recommended)
//! use siumai::auth::vertex::*;
//! ```

pub mod cancel;
pub mod mime;
pub mod model_alias;
pub mod url;
pub mod utf8_decoder;

#[deprecated(since = "0.11.1", note = "Use `crate::auth::vertex` instead")]
pub mod vertex;

// Re-exports for convenience
pub use mime::*;
pub use model_alias::*;
pub use url::*;
pub use utf8_decoder::Utf8StreamDecoder;

// Deprecated re-exports
#[allow(deprecated)]
pub use vertex::*;
