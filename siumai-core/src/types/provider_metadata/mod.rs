//! Provider-specific metadata types
//!
//! This module provides strongly-typed structures for provider-specific metadata
//! returned in responses. While the response stores metadata as a nested HashMap
//! for flexibility, these types provide type-safe access to common provider metadata.

pub mod anthropic;
pub mod gemini;
pub mod openai;

pub use anthropic::AnthropicMetadata;
pub use gemini::GeminiMetadata;
pub use openai::OpenAiMetadata;

/// Helper trait for converting HashMap metadata to typed structures
pub trait FromMetadata: Sized {
    /// Try to parse metadata from a HashMap
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self>;
}
