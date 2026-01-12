//! Unified response format hints (Vercel-aligned).
//!
//! This module models request-level structured output hints such as
//! `responseFormat: { type: "json", schema: ... }`.

use serde::{Deserialize, Serialize};

/// Response format hint for the model output.
///
/// This is aligned with Vercel AI SDK's `responseFormat` option on generate calls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ResponseFormat {
    /// Request JSON output that conforms to the given JSON schema.
    Json { schema: serde_json::Value },
}
