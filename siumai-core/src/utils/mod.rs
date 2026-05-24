//! Core-owned utility modules.
//!
//! Generic provider/protocol helpers live in `siumai-provider-utils`. This module only keeps
//! utilities that depend directly on core runtime stream types plus explicit compatibility helpers.
pub mod cancel;
pub mod streaming_tool_call;

pub use cancel::{delay, is_abort_error};
pub use streaming_tool_call::*;
