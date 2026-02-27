//! Error handling (re-export).
//!
//! The canonical error types live in `siumai-spec` and are re-exported here.

pub mod helpers;
pub use helpers::*;
pub use siumai_spec::error::*;
