//! Provider-specific metadata types
//!
//! This module provides strongly-typed structures for provider-specific metadata
//! returned in responses. While the response stores metadata as a nested HashMap
//! for flexibility, these types provide type-safe access to common provider metadata.

// Provider-specific typed metadata types are intentionally owned by provider crates.

/// Helper trait for converting HashMap metadata to typed structures
pub trait FromMetadata: Sized {
    /// Try to parse metadata from a HashMap
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self>;
}
