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
//! use siumai::experimental::utils::vertex::*;
//!
//! // New (recommended)
//! use siumai::experimental::auth::vertex::*;
//! ```

pub mod builder_helpers;
pub mod cancel;
pub mod chat_request;
pub mod data;
pub mod download;
pub mod error_message;
pub mod headers;
pub mod id;
pub mod json_instruction;
pub mod json_parse;
pub mod mime;
pub mod model_alias;
pub mod option;
pub mod provider_options;
pub mod provider_reference;
pub mod reasoning;
pub mod runtime;
pub mod serial_job;
pub mod settings;
pub mod streaming_tool_call;
pub mod url;
pub mod utf8_decoder;
pub mod validate_types;

#[deprecated(since = "0.11.1", note = "Use `crate::auth::vertex` instead")]
pub mod vertex;

// Re-exports for convenience
pub use data::*;
pub use download::*;
pub use error_message::*;
pub use headers::*;
pub use id::*;
pub use json_instruction::*;
pub use json_parse::*;
pub use mime::*;
pub use model_alias::*;
pub use option::*;
pub use provider_options::*;
pub use provider_reference::*;
pub use reasoning::*;
pub use runtime::*;
pub use serial_job::*;
pub use settings::*;
pub use streaming_tool_call::*;
pub use url::*;
pub use utf8_decoder::Utf8StreamDecoder;
pub use validate_types::*;

// Deprecated re-exports
#[allow(deprecated)]
pub use vertex::*;
