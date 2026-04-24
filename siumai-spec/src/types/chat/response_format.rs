//! Unified response format hints (Vercel-aligned).
//!
//! This module models request-level structured output hints such as
//! `responseFormat: { type: "json", schema: ... }`.

use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Response format hint for the model output.
///
/// This is aligned with Vercel AI SDK's `responseFormat` option on generate calls.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    /// Request JSON output without a provider-facing JSON Schema.
    ///
    /// This corresponds to AI SDK `responseFormat: { type: "json" }`.
    JsonObject {
        /// Optional output name (provider-dependent).
        name: Option<String>,

        /// Optional output description (provider-dependent).
        description: Option<String>,
    },

    /// Request JSON output that conforms to the given JSON schema.
    Json {
        /// JSON schema describing the expected output.
        schema: serde_json::Value,

        /// Optional schema name (provider-dependent).
        ///
        /// For OpenAI, this maps to the `json_schema.name` field.
        name: Option<String>,

        /// Optional schema description (provider-dependent).
        ///
        /// For OpenAI, this maps to the `json_schema.description` field.
        description: Option<String>,

        /// Optional strictness hint (provider-dependent).
        ///
        /// For OpenAI, this maps to `json_schema.strict`.
        /// When unset, provider-specific defaults apply.
        strict: Option<bool>,
    },
}

impl ResponseFormat {
    /// Create a schema-less JSON response format hint.
    pub const fn json_object() -> Self {
        Self::JsonObject {
            name: None,
            description: None,
        }
    }

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
            Self::JsonObject { name: n, .. } => {
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
            Self::JsonObject { description: d, .. } => {
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
            Self::JsonObject { .. } => {}
        }
        self
    }
}

impl Serialize for ResponseFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", "json")?;

        match self {
            Self::JsonObject { name, description } => {
                if let Some(name) = name {
                    map.serialize_entry("name", name)?;
                }
                if let Some(description) = description {
                    map.serialize_entry("description", description)?;
                }
            }
            Self::Json {
                schema,
                name,
                description,
                strict,
            } => {
                map.serialize_entry("schema", schema)?;
                if let Some(name) = name {
                    map.serialize_entry("name", name)?;
                }
                if let Some(description) = description {
                    map.serialize_entry("description", description)?;
                }
                if let Some(strict) = strict {
                    map.serialize_entry("strict", strict)?;
                }
            }
        }

        map.end()
    }
}

impl<'de> Deserialize<'de> for ResponseFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireResponseFormat {
            #[serde(rename = "type")]
            ty: String,
            #[serde(default)]
            schema: Option<serde_json::Value>,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            description: Option<String>,
            #[serde(default)]
            strict: Option<bool>,
        }

        let wire = WireResponseFormat::deserialize(deserializer)?;
        if wire.ty != "json" {
            return Err(D::Error::custom(format!(
                "unsupported response format type {:?}",
                wire.ty
            )));
        }

        Ok(match wire.schema {
            Some(schema) => Self::Json {
                schema,
                name: wire.name,
                description: wire.description,
                strict: wire.strict,
            },
            None => Self::JsonObject {
                name: wire.name,
                description: wire.description,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ResponseFormat;

    #[test]
    fn json_object_serializes_like_ai_sdk_schema_less_json() {
        let value = serde_json::to_value(
            ResponseFormat::json_object()
                .with_name("payload")
                .with_description("Any JSON payload"),
        )
        .expect("serialize response format");

        assert_eq!(
            value,
            serde_json::json!({
                "type": "json",
                "name": "payload",
                "description": "Any JSON payload"
            })
        );
    }

    #[test]
    fn json_schema_serializes_with_schema_and_strictness() {
        let schema = serde_json::json!({ "type": "object" });
        let value = serde_json::to_value(
            ResponseFormat::json_schema(schema.clone())
                .with_name("payload")
                .with_strict(false),
        )
        .expect("serialize response format");

        assert_eq!(
            value,
            serde_json::json!({
                "type": "json",
                "schema": schema,
                "name": "payload",
                "strict": false
            })
        );
    }

    #[test]
    fn response_format_deserializes_by_schema_presence() {
        let schema_less: ResponseFormat =
            serde_json::from_value(serde_json::json!({ "type": "json" }))
                .expect("deserialize schema-less response format");
        assert_eq!(schema_less, ResponseFormat::json_object());

        let schema = serde_json::json!({ "type": "object" });
        let schema_format: ResponseFormat =
            serde_json::from_value(serde_json::json!({ "type": "json", "schema": schema }))
                .expect("deserialize schema response format");
        assert!(matches!(schema_format, ResponseFormat::Json { .. }));
    }
}
