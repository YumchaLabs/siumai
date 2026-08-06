//! Model-visible tool contracts, ownership, and typed outcomes.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::annotations::{
    ProviderAnnotationError, ToolAnnotationTarget, ToolAnnotations, TypedProviderAnnotation,
};
use crate::provider::ProviderId;

/// Invalid model-visible tool definition.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid tool definition: {0}")]
pub struct InvalidToolSpec(String);

/// A tool definition sent to a model. It contains no executable callback.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolSpec {
    name: String,
    description: Option<String>,
    input_schema: Value,
    #[serde(
        default,
        rename = "providerAnnotations",
        skip_serializing_if = "ToolAnnotations::is_empty"
    )]
    provider_annotations: ToolAnnotations,
}

/// Owned components returned when decomposing a [`ToolSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpecParts {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
    pub provider_annotations: ToolAnnotations,
}

impl ToolSpec {
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        input_schema: Value,
    ) -> Result<Self, InvalidToolSpec> {
        Self::from_parts(ToolSpecParts {
            name: name.into(),
            description,
            input_schema,
            provider_annotations: ToolAnnotations::default(),
        })
    }

    /// Rebuild a tool definition while preserving previously validated annotations.
    pub fn from_parts(parts: ToolSpecParts) -> Result<Self, InvalidToolSpec> {
        let ToolSpecParts {
            name,
            description,
            input_schema,
            provider_annotations,
        } = parts;
        if name.is_empty()
            || name.len() > 128
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(InvalidToolSpec(
                "name must be 1..=128 ASCII letters, digits, '-' or '_'".to_string(),
            ));
        }
        if !input_schema.is_object() && !input_schema.is_boolean() {
            return Err(InvalidToolSpec(
                "input_schema must be a JSON Schema object or boolean".to_string(),
            ));
        }
        Ok(Self {
            name,
            description,
            input_schema,
            provider_annotations,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn input_schema(&self) -> &Value {
        &self.input_schema
    }

    pub fn annotations(&self) -> &ToolAnnotations {
        &self.provider_annotations
    }

    pub fn with_provider_annotation<T>(
        mut self,
        annotation: &T,
    ) -> Result<Self, ProviderAnnotationError>
    where
        T: TypedProviderAnnotation<Target = ToolAnnotationTarget>,
    {
        self.provider_annotations.insert(annotation)?;
        Ok(self)
    }

    pub fn into_parts(self) -> ToolSpecParts {
        ToolSpecParts {
            name: self.name,
            description: self.description,
            input_schema: self.input_schema,
            provider_annotations: self.provider_annotations,
        }
    }
}

#[derive(Deserialize)]
struct ToolSpecWire {
    name: String,
    description: Option<String>,
    input_schema: Value,
    #[serde(default, rename = "providerAnnotations")]
    provider_annotations: ToolAnnotations,
}

impl<'de> Deserialize<'de> for ToolSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ToolSpecWire::deserialize(deserializer)?;
        Self::from_parts(ToolSpecParts {
            name: wire.name,
            description: wire.description,
            input_schema: wire.input_schema,
            provider_annotations: wire.provider_annotations,
        })
        .map_err(serde::de::Error::custom)
    }
}

/// The runtime that is allowed to execute a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExecutionOwner {
    Local,
    Provider { provider: ProviderId },
}

/// Immutable identity of a local executable binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolBindingIdentity {
    pub name: String,
    pub fingerprint: String,
}

/// A model-emitted call with explicit execution ownership.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    pub owner: ExecutionOwner,
}

/// A successful or unsuccessful outcome returned to a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolOutcome {
    Success {
        value: Value,
    },
    Denied {
        reason: String,
    },
    ExecutionFailed {
        message: String,
        retryable: bool,
        /// Optional provider-native or integration-specific diagnostic data.
        ///
        /// The runtime never interprets this value as a successful result. It
        /// is bounded by the owning integration before it reaches a model or
        /// a durable snapshot.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
    },
    Cancelled {
        reason: String,
    },
}

/// A tool result retains its call identity and typed outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub name: String,
    pub outcome: ToolOutcome,
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct HostedToolAnnotation {
        hosted_tool: String,
    }

    impl TypedProviderAnnotation for HostedToolAnnotation {
        type Target = ToolAnnotationTarget;

        const NAMESPACE: &'static str = "openai";
        const API_MODE: Option<&'static str> = Some("responses");
    }

    #[test]
    fn denied_and_failed_outcomes_are_not_success_json() {
        let denied = ToolOutcome::Denied {
            reason: "approval denied".to_string(),
        };
        let failed = ToolOutcome::ExecutionFailed {
            message: "database unavailable".to_string(),
            retryable: true,
            details: None,
        };

        assert!(!matches!(denied, ToolOutcome::Success { .. }));
        assert!(!matches!(failed, ToolOutcome::Success { .. }));
        assert_ne!(
            serde_json::to_value(denied).unwrap(),
            json!("approval denied")
        );
    }

    #[test]
    fn deserialization_cannot_bypass_tool_name_validation() {
        assert!(
            serde_json::from_value::<ToolSpec>(json!({
                "name": "invalid name",
                "description": null,
                "input_schema": {"type": "object"}
            }))
            .is_err()
        );
    }

    #[test]
    fn tool_spec_parts_retain_provider_annotations() {
        let expected = HostedToolAnnotation {
            hosted_tool: "web_search".to_string(),
        };
        let tool = ToolSpec::new(
            "search",
            Some("Search the web".to_string()),
            json!({"type": "object"}),
        )
        .unwrap()
        .with_provider_annotation(&expected)
        .unwrap();

        let decoded: ToolSpec =
            serde_json::from_value(serde_json::to_value(&tool).unwrap()).unwrap();
        let parts = decoded.into_parts();

        assert_eq!(parts.name, "search");
        let rebuilt = ToolSpec::from_parts(parts).unwrap();
        assert_eq!(
            rebuilt
                .annotations()
                .decode::<HostedToolAnnotation>()
                .unwrap(),
            Some(expected)
        );
    }
}
