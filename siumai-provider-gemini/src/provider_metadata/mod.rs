//! Provider-owned typed response metadata.
//!
//! This module contains typed views over `ChatResponse.provider_metadata` that are owned by the
//! provider crate to avoid coupling `siumai-core` to provider-specific response shapes.

pub mod gemini;

