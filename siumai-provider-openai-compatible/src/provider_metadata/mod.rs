//! Typed response metadata (protocol-owned).
//!
//! This module contains typed views over `ChatResponse.provider_metadata` that are owned by the
//! shared OpenAI-like protocol crate. Provider crates may re-export these types via
//! `siumai::provider_ext::openai::*` for convenience.

pub mod openai;
