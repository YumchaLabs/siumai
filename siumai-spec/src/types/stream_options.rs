//! Runtime-only stream request options.
//!
//! These options affect local stream processing behavior and must not be
//! serialized onto provider wire payloads.

use serde::{Deserialize, Serialize};

/// Runtime-only stream options carried on stable request types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct StreamRequestOptions {
    /// Include provider raw chunks on the semantic stream part lane.
    #[serde(default, rename = "includeRawChunks")]
    pub include_raw_chunks: bool,
}

impl StreamRequestOptions {
    /// Create a new runtime stream-options object.
    pub const fn new() -> Self {
        Self {
            include_raw_chunks: false,
        }
    }

    /// Enable or disable raw chunk emission.
    pub const fn with_include_raw_chunks(mut self, include_raw_chunks: bool) -> Self {
        self.include_raw_chunks = include_raw_chunks;
        self
    }

    /// Whether all options are still at their defaults.
    pub const fn is_default(&self) -> bool {
        !self.include_raw_chunks
    }
}
