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
    Json {
        /// JSON schema describing the expected output.
        schema: serde_json::Value,

        /// Optional schema name (provider-dependent).
        ///
        /// For OpenAI, this maps to the `json_schema.name` field.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,

        /// Optional schema description (provider-dependent).
        ///
        /// For OpenAI, this maps to the `json_schema.description` field.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,

        /// Optional strictness hint (provider-dependent).
        ///
        /// For OpenAI, this maps to `json_schema.strict`.
        /// When unset, provider-specific defaults apply.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

impl ResponseFormat {
    /// Create a JSON schema response format hint with provider-agnostic defaults.
    pub fn json_schema(schema: serde_json::Value) -> Self {
        Self::Json {
            schema,
            name: None,
            description: None,
            strict: None,
        }
    }

    /// Set schema name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        match &mut self {
            Self::Json { name: n, .. } => {
                *n = Some(name.into());
            }
        }
        self
    }

    /// Set schema description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        match &mut self {
            Self::Json { description: d, .. } => {
                *d = Some(description.into());
            }
        }
        self
    }

    /// Set strictness hint.
    pub fn with_strict(mut self, strict: bool) -> Self {
        match &mut self {
            Self::Json { strict: s, .. } => {
                *s = Some(strict);
            }
        }
        self
    }
}
