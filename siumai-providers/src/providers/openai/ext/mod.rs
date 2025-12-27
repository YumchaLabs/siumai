//! OpenAI provider extension APIs (non-unified surface)
//!
//! These APIs are intentionally *not* part of the Vercel-aligned unified model families.
//! Use them when you need OpenAI-specific endpoints/resources beyond the unified surface.

pub mod hosted_tools;
pub mod responses;
pub mod speech_streaming;
pub mod transcription_streaming;
