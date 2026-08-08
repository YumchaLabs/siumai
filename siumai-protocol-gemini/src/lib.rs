//! Google Gemini protocol codecs.
#![deny(unsafe_code)]

/// Stable Gemini v1 embedding request and response codecs.
pub mod embedding;

/// Stable-v1 Generate Content request, response, and SSE codecs.
pub mod generate_content;

/// Gemini Interactions request and response codecs.
pub mod interactions;
