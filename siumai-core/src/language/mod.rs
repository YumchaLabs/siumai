//! Canonical provider-neutral language request and response content.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::provider::{ModelId, ProviderId};
use crate::tool::{ToolCall, ToolResult, ToolSpec};
use crate::usage::Usage;

pub const DEFAULT_OPAQUE_ITEM_LIMIT: usize = 64 * 1024;
const MAX_OPAQUE_KIND_BYTES: usize = 256;
const MAX_OPAQUE_PROTOCOL_BYTES: usize = 128;
const MAX_OPAQUE_PLATFORM_BYTES: usize = 512;

/// Provenance required to replay or project provider-native state safely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderProvenance {
    pub provider: ProviderId,
    pub platform: Option<String>,
    pub protocol: String,
    pub model: ModelId,
}

/// A bounded provider-native item retained without pretending it is portable.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpaqueProviderItem {
    provenance: ProviderProvenance,
    kind: String,
    data: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum OpaqueProviderItemError {
    #[error("opaque provider item kind must not be empty")]
    EmptyKind,
    #[error("opaque provider item protocol must not be empty")]
    EmptyProtocol,
    #[error("opaque provider item {field} exceeds {maximum} bytes")]
    FieldTooLarge { field: &'static str, maximum: usize },
    #[error("opaque provider item envelope is {actual} bytes; maximum is {maximum}")]
    TooLarge { actual: usize, maximum: usize },
    #[error("failed to measure opaque provider item: {0}")]
    Serialization(String),
}

impl OpaqueProviderItem {
    pub fn new(
        provenance: ProviderProvenance,
        kind: impl Into<String>,
        data: Value,
    ) -> Result<Self, OpaqueProviderItemError> {
        Self::with_limit(provenance, kind, data, DEFAULT_OPAQUE_ITEM_LIMIT)
    }

    pub fn with_limit(
        provenance: ProviderProvenance,
        kind: impl Into<String>,
        data: Value,
        maximum: usize,
    ) -> Result<Self, OpaqueProviderItemError> {
        let kind = kind.into();
        if kind.trim().is_empty() {
            return Err(OpaqueProviderItemError::EmptyKind);
        }
        if kind.len() > MAX_OPAQUE_KIND_BYTES {
            return Err(OpaqueProviderItemError::FieldTooLarge {
                field: "kind",
                maximum: MAX_OPAQUE_KIND_BYTES,
            });
        }
        if provenance.protocol.trim().is_empty() {
            return Err(OpaqueProviderItemError::EmptyProtocol);
        }
        if provenance.protocol.len() > MAX_OPAQUE_PROTOCOL_BYTES {
            return Err(OpaqueProviderItemError::FieldTooLarge {
                field: "provenance.protocol",
                maximum: MAX_OPAQUE_PROTOCOL_BYTES,
            });
        }
        if provenance
            .platform
            .as_ref()
            .is_some_and(|platform| platform.len() > MAX_OPAQUE_PLATFORM_BYTES)
        {
            return Err(OpaqueProviderItemError::FieldTooLarge {
                field: "provenance.platform",
                maximum: MAX_OPAQUE_PLATFORM_BYTES,
            });
        }
        let wire = OpaqueProviderItemWire {
            provenance,
            kind,
            data,
        };
        let actual = serde_json::to_vec(&wire)
            .map_err(|error| OpaqueProviderItemError::Serialization(error.to_string()))?
            .len();
        if actual > maximum {
            return Err(OpaqueProviderItemError::TooLarge { actual, maximum });
        }
        Ok(Self {
            provenance: wire.provenance,
            kind: wire.kind,
            data: wire.data,
        })
    }

    pub fn provenance(&self) -> &ProviderProvenance {
        &self.provenance
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn data(&self) -> &Value {
        &self.data
    }
}

#[derive(Serialize, Deserialize)]
struct OpaqueProviderItemWire {
    provenance: ProviderProvenance,
    kind: String,
    data: Value,
}

impl<'de> Deserialize<'de> for OpaqueProviderItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = OpaqueProviderItemWire::deserialize(deserializer)?;
        Self::new(wire.provenance, wire.kind, wire.data).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MessageRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

/// Owned media input or output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MediaData {
    Bytes(Vec<u8>),
    Url(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaPart {
    pub media_type: String,
    pub data: MediaData,
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    pub source_id: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub start: Option<u64>,
    pub end: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ContentPart {
    Text { text: String },
    Reasoning { text: String },
    Media(MediaPart),
    Citation(Citation),
    Refusal { reason: Option<String> },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    ProviderOpaque(OpaqueProviderItem),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<ContentPart>,
}

impl Message {
    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text { text: text.into() }],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub generation: GenerationConfig,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<StructuredOutputSpec>,
}

impl LanguageRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            generation: GenerationConfig::default(),
            tools: Vec::new(),
            tool_choice: None,
            structured_output: None,
        }
    }

    pub fn with_generation(mut self, generation: GenerationConfig) -> Self {
        self.generation = generation;
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Validate cross-field invariants before provider policy and encoding.
    pub fn validate(&self) -> Result<(), LanguageRequestError> {
        self.generation.validate()?;
        let mut names = BTreeSet::new();
        for tool in &self.tools {
            if !names.insert(tool.name()) {
                return Err(LanguageRequestError::DuplicateTool {
                    name: tool.name().to_string(),
                });
            }
        }
        if let Some(ToolChoice::Named { name }) = &self.tool_choice
            && !names.contains(name.as_str())
        {
            return Err(LanguageRequestError::UnknownToolChoice { name: name.clone() });
        }
        if let Some(output) = &self.structured_output {
            if output.name.trim().is_empty() {
                return Err(LanguageRequestError::EmptyStructuredOutputName);
            }
            if !output.schema.is_object() && !output.schema.is_boolean() {
                return Err(LanguageRequestError::InvalidStructuredOutputSchema);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LanguageRequestError {
    #[error(transparent)]
    Generation(#[from] GenerationConfigError),
    #[error("tool `{name}` is defined more than once")]
    DuplicateTool { name: String },
    #[error("tool choice names undefined tool `{name}`")]
    UnknownToolChoice { name: String },
    #[error("structured output name must not be empty")]
    EmptyStructuredOutputName,
    #[error("structured output schema must be a JSON Schema object or boolean")]
    InvalidStructuredOutputSchema,
}

/// Common generation controls with omission preserved explicitly.
///
/// Providers reject controls they cannot encode without loss. Model-specific
/// controls remain in typed provider options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_output_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GenerationConfigError {
    #[error("temperature must be finite and non-negative")]
    InvalidTemperature,
    #[error("top_p must be finite and between 0 and 1 inclusive")]
    InvalidTopP,
    #[error("stop sequence {index} must not be empty")]
    EmptyStopSequence { index: usize },
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<(), GenerationConfigError> {
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(GenerationConfigError::InvalidTemperature);
        }
        if self
            .top_p
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(GenerationConfigError::InvalidTopP);
        }
        if let Some(index) = self
            .stop_sequences
            .iter()
            .position(|sequence| sequence.is_empty())
        {
            return Err(GenerationConfigError::EmptyStopSequence { index });
        }
        Ok(())
    }
}

/// Portable tool selection. Provider-only hosted tools remain typed extensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named { name: String },
}

/// Provider-neutral JSON Schema request shaping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputSpec {
    pub name: String,
    pub description: Option<String>,
    pub schema: Value,
    pub strict: bool,
}

/// A streamed structured value that has not passed final schema validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartialStructuredOutput {
    value: Value,
}

impl PartialStructuredOutput {
    pub fn unvalidated(value: Value) -> Self {
        Self { value }
    }

    pub fn value(&self) -> &Value {
        &self.value
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Refusal,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Warning {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageResponse {
    pub id: Option<String>,
    pub model: Option<ModelId>,
    pub content: Vec<ContentPart>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn provenance() -> ProviderProvenance {
        ProviderProvenance {
            provider: ProviderId::new("openai").unwrap(),
            platform: None,
            protocol: "responses".to_string(),
            model: ModelId::new("gpt-future:preview").unwrap(),
        }
    }

    #[test]
    fn opaque_items_round_trip_with_provenance() {
        let item = OpaqueProviderItem::new(
            provenance(),
            "reasoning.encrypted",
            json!({"encrypted_content": "opaque"}),
        )
        .unwrap();

        let encoded = serde_json::to_value(&item).unwrap();
        let decoded: OpaqueProviderItem = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, item);
        assert_eq!(decoded.provenance().protocol, "responses");
    }

    #[test]
    fn opaque_items_enforce_a_byte_limit() {
        let error = OpaqueProviderItem::with_limit(
            provenance(),
            "large",
            Value::String("too large".to_string()),
            2,
        )
        .unwrap_err();
        assert!(matches!(error, OpaqueProviderItemError::TooLarge { .. }));
    }

    #[test]
    fn common_generation_controls_preserve_omission_and_validate_ranges() {
        let request = LanguageRequest::new(vec![Message::text(MessageRole::Developer, "rules")]);
        assert_eq!(request.generation.temperature, None);
        assert_eq!(request.tool_choice, None);

        let invalid = GenerationConfig {
            top_p: Some(1.5),
            ..GenerationConfig::default()
        };
        assert_eq!(invalid.validate(), Err(GenerationConfigError::InvalidTopP));
    }

    #[test]
    fn language_request_validation_rejects_unknown_named_tool() {
        let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
            .with_tool_choice(ToolChoice::Named {
                name: "lookup".to_string(),
            });
        assert_eq!(
            request.validate(),
            Err(LanguageRequestError::UnknownToolChoice {
                name: "lookup".to_string()
            })
        );
    }
}
