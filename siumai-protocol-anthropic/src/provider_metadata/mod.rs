//! Typed response metadata (protocol-owned).
//!
//! This module contains typed views over `ChatResponse.provider_metadata` that are owned by the
//! shared Anthropic Messages protocol crate. Provider crates may re-export these types via
//! `siumai::provider_ext::anthropic::*` for convenience.

pub mod anthropic;
