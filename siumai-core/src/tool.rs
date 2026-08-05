//! Model-visible tool contracts, ownership, and typed outcomes.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;

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
}

impl ToolSpec {
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        input_schema: Value,
    ) -> Result<Self, InvalidToolSpec> {
        let name = name.into();
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

    pub fn into_parts(self) -> (String, Option<String>, Value) {
        (self.name, self.description, self.input_schema)
    }
}

#[derive(Deserialize)]
struct ToolSpecWire {
    name: String,
    description: Option<String>,
    input_schema: Value,
}

impl<'de> Deserialize<'de> for ToolSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ToolSpecWire::deserialize(deserializer)?;
        Self::new(wire.name, wire.description, wire.input_schema).map_err(serde::de::Error::custom)
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
    use serde_json::json;

    use super::*;

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
}
