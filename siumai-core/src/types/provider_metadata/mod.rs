//! Provider metadata helpers.
//!
//! Provider-specific typed metadata types are intentionally owned by provider crates to
//! reduce coupling and compile cost in `siumai-core`.

// Provider-specific typed metadata types are intentionally owned by provider crates.

/// Helper trait for converting HashMap metadata to typed structures
pub trait FromMetadata: Sized {
    /// Try to parse metadata from a HashMap
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self>;
}
