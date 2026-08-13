//! Model-visible tool contracts, ownership, and typed outcomes.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::annotations::{
    ProviderAnnotationError, ToolAnnotationTarget, ToolAnnotations, TypedProviderAnnotation,
};
use crate::provider::ProviderId;

/// Default maximum encoded size of one semantic tool input.
pub const DEFAULT_TOOL_INPUT_BYTE_LIMIT: usize = 1024 * 1024;

const MAX_TOOL_CALL_ID_BYTES: usize = 512;

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
        if !is_valid_tool_name(&name) {
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

/// Invalid canonical tool input.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InvalidToolInput {
    #[error("tool input could not be encoded as JSON")]
    Encoding,
    #[error("tool input is {actual} encoded bytes; maximum is {maximum}")]
    TooLarge { actual: usize, maximum: usize },
}

/// One parsed JSON value used as the semantic input of an executable tool call.
///
/// Provider protocols may carry this value as encoded JSON text. Protocol decoders parse that text
/// exactly once before constructing this type. A JSON string stored here is therefore a semantic
/// string value, not an unparsed object or array.
#[derive(Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ToolInput {
    value: Value,
    #[serde(skip)]
    encoded_json_bytes: usize,
}

impl ToolInput {
    pub fn new(value: Value) -> Result<Self, InvalidToolInput> {
        Self::with_byte_limit(value, DEFAULT_TOOL_INPUT_BYTE_LIMIT)
    }

    pub fn with_byte_limit(value: Value, maximum: usize) -> Result<Self, InvalidToolInput> {
        let maximum = maximum.min(DEFAULT_TOOL_INPUT_BYTE_LIMIT);
        let encoded_json_bytes = serde_json::to_vec(&value)
            .map_err(|_| InvalidToolInput::Encoding)?
            .len();
        if encoded_json_bytes > maximum {
            return Err(InvalidToolInput::TooLarge {
                actual: encoded_json_bytes,
                maximum,
            });
        }
        Ok(Self {
            value,
            encoded_json_bytes,
        })
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn encoded_json_bytes(&self) -> usize {
        self.encoded_json_bytes
    }

    pub fn into_value(self) -> Value {
        self.value
    }
}

impl fmt::Debug for ToolInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolInput")
            .field("encoded_json_bytes", &self.encoded_json_bytes)
            .finish_non_exhaustive()
    }
}

impl<'de> Deserialize<'de> for ToolInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Why a model-emitted tool call is invalid.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InvalidToolCall {
    #[error(
        "tool call ID must be non-empty, at most {MAX_TOOL_CALL_ID_BYTES} bytes, and contain no control characters"
    )]
    InvalidId,
    #[error("tool call name must be 1..=128 ASCII letters, digits, '-' or '_'")]
    InvalidName,
    #[error("provider-owned tool activity is not a portable executable tool call")]
    ProviderOwned,
    #[error(transparent)]
    Input(#[from] InvalidToolInput),
}

/// Owned components returned when decomposing a [`ToolCall`].
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallParts {
    pub id: String,
    pub name: String,
    pub input: ToolInput,
    pub owner: ExecutionOwner,
}

/// Immutable identity of a local executable binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolBindingIdentity {
    pub name: String,
    pub fingerprint: String,
}

/// A validated model-emitted call with explicit execution ownership.
#[derive(Clone, PartialEq, Serialize)]
pub struct ToolCall {
    id: String,
    name: String,
    #[serde(rename = "arguments")]
    input: ToolInput,
    owner: ExecutionOwner,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: Value,
        owner: ExecutionOwner,
    ) -> Result<Self, InvalidToolCall> {
        Self::from_parts(ToolCallParts {
            id: id.into(),
            name: name.into(),
            input: ToolInput::new(arguments)?,
            owner,
        })
    }

    pub fn local(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: Value,
    ) -> Result<Self, InvalidToolCall> {
        Self::new(id, name, arguments, ExecutionOwner::Local)
    }

    pub fn from_parts(parts: ToolCallParts) -> Result<Self, InvalidToolCall> {
        let ToolCallParts {
            id,
            name,
            input,
            owner,
        } = parts;
        if !is_valid_tool_call_id(&id) {
            return Err(InvalidToolCall::InvalidId);
        }
        if !is_valid_tool_name(&name) {
            return Err(InvalidToolCall::InvalidName);
        }
        if !matches!(owner, ExecutionOwner::Local) {
            return Err(InvalidToolCall::ProviderOwned);
        }
        Ok(Self {
            id,
            name,
            input,
            owner,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn input(&self) -> &ToolInput {
        &self.input
    }

    pub fn arguments(&self) -> &Value {
        self.input.value()
    }

    pub fn owner(&self) -> &ExecutionOwner {
        &self.owner
    }

    pub fn into_parts(self) -> ToolCallParts {
        ToolCallParts {
            id: self.id,
            name: self.name,
            input: self.input,
            owner: self.owner,
        }
    }
}

impl fmt::Debug for ToolCall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolCall")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("input", &self.input)
            .field("owner", &self.owner)
            .finish()
    }
}

#[derive(Deserialize)]
struct ToolCallWire {
    id: String,
    name: String,
    arguments: Value,
    owner: ExecutionOwner,
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ToolCallWire::deserialize(deserializer)?;
        Self::new(wire.id, wire.name, wire.arguments, wire.owner).map_err(serde::de::Error::custom)
    }
}

fn is_valid_tool_call_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TOOL_CALL_ID_BYTES
        && !value.chars().any(char::is_control)
}

fn is_valid_tool_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
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

    #[test]
    fn local_tool_calls_keep_one_validated_json_value() {
        for arguments in [json!({"query": "siumai"}), json!([1, 2]), json!(7)] {
            let call = ToolCall::local("call_1", "lookup", arguments.clone()).unwrap();

            assert_eq!(call.id(), "call_1");
            assert_eq!(call.name(), "lookup");
            assert_eq!(call.arguments(), &arguments);
            assert_eq!(call.owner(), &ExecutionOwner::Local);

            let wire = serde_json::to_value(&call).unwrap();
            assert_eq!(wire["arguments"], arguments);
            assert_eq!(serde_json::from_value::<ToolCall>(wire).unwrap(), call);
        }
    }

    #[test]
    fn tool_call_ids_remain_opaque_across_serialization() {
        let call = ToolCall::local("call/+@?-\u{8c03}\u{7528}", "lookup", json!({})).unwrap();
        let wire = serde_json::to_value(&call).unwrap();
        let decoded = serde_json::from_value::<ToolCall>(wire).unwrap();

        assert_eq!(decoded.id(), "call/+@?-\u{8c03}\u{7528}");
        assert_eq!(decoded, call);
    }

    #[test]
    fn tool_call_deserialization_cannot_bypass_identity_or_input_limits() {
        assert!(
            serde_json::from_value::<ToolCall>(json!({
                "id": "",
                "name": "lookup",
                "arguments": {},
                "owner": "Local"
            }))
            .is_err()
        );
        assert!(ToolCall::local("call\n1", "lookup", json!({})).is_err());
        assert!(
            serde_json::from_value::<ToolCall>(json!({
                "id": "call_1",
                "name": "invalid name",
                "arguments": {},
                "owner": "Local"
            }))
            .is_err()
        );

        let oversized = Value::String("x".repeat(DEFAULT_TOOL_INPUT_BYTE_LIMIT));
        assert!(matches!(
            ToolCall::local("call_1", "lookup", oversized.clone()),
            Err(InvalidToolCall::Input(InvalidToolInput::TooLarge { .. }))
        ));
        assert!(matches!(
            ToolInput::with_byte_limit(oversized, usize::MAX),
            Err(InvalidToolInput::TooLarge {
                maximum: DEFAULT_TOOL_INPUT_BYTE_LIMIT,
                ..
            })
        ));
        assert!(matches!(
            ToolCall::new(
                "call_1",
                "lookup",
                json!({}),
                ExecutionOwner::Provider {
                    provider: ProviderId::new("openai").unwrap()
                }
            ),
            Err(InvalidToolCall::ProviderOwned)
        ));
        assert!(
            serde_json::from_value::<ToolCall>(json!({
                "id": "call_1",
                "name": "lookup",
                "arguments": {},
                "owner": {"Provider": {"provider": "openai"}}
            }))
            .is_err()
        );
    }

    #[test]
    fn tool_call_debug_redacts_semantic_input() {
        let call =
            ToolCall::local("call_1", "lookup", json!({"secret": "tool-input-sentinel"})).unwrap();

        let debug = format!("{call:?}");
        assert!(debug.contains("call_1"));
        assert!(debug.contains("lookup"));
        assert!(!debug.contains("tool-input-sentinel"));
    }
}
