//! Canonical provider-neutral language request and response content.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::error::PublicDiagnosticText;
use crate::provider::{ModelId, ProviderId};
use crate::tool::{ToolCall, ToolResult, ToolSpec};
use crate::usage::Usage;

/// Default per-item bound leaves room for encrypted reasoning and provider tool
/// payloads while remaining small enough to reject unbounded history growth.
pub const DEFAULT_OPAQUE_ITEM_LIMIT: usize = 1024 * 1024;
pub const DEFAULT_OPAQUE_ITEM_COUNT_LIMIT: usize = 128;
/// Default aggregate bound for one request or response's provider-native state.
pub const DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT: usize = 16 * 1024 * 1024;
const MAX_OPAQUE_KIND_BYTES: usize = 256;
const MAX_OPAQUE_PROTOCOL_BYTES: usize = 128;
const MAX_OPAQUE_PLATFORM_BYTES: usize = 512;
const MAX_OPAQUE_ITEM_ID_BYTES: usize = 512;
const MAX_OPAQUE_RELATION_KIND_BYTES: usize = 128;
const MAX_OPAQUE_RELATION_TARGET_BYTES: usize = 512;
const MAX_OPAQUE_RELATIONS: usize = 32;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    item_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    relations: Vec<ProviderItemRelation>,
    data: Value,
    #[serde(skip)]
    encoded_json_bytes: usize,
}

/// One provider-native identity relation retained outside the opaque payload.
///
/// Relations are directed from the containing [`OpaqueProviderItem`] to
/// `target_id`. For example, a `caller` relation points from a function-call
/// item to the program call that produced it.
///
/// The relation name remains provider-defined so future protocols can preserve
/// new identity edges without changing the stable core contract. The built-in
/// constructors cover the item, call, and caller relations used by current
/// Responses items.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderItemRelation {
    kind: String,
    target_id: String,
}

impl ProviderItemRelation {
    pub fn new(
        kind: impl Into<String>,
        target_id: impl Into<String>,
    ) -> Result<Self, OpaqueProviderItemError> {
        let kind = kind.into();
        validate_opaque_field(&kind, "relation.kind", MAX_OPAQUE_RELATION_KIND_BYTES)?;
        let target_id = target_id.into();
        validate_opaque_field(
            &target_id,
            "relation.target_id",
            MAX_OPAQUE_RELATION_TARGET_BYTES,
        )?;
        Ok(Self { kind, target_id })
    }

    pub fn related_item(item_id: impl Into<String>) -> Result<Self, OpaqueProviderItemError> {
        Self::new("related_item", item_id)
    }

    pub fn call(call_id: impl Into<String>) -> Result<Self, OpaqueProviderItemError> {
        Self::new("call", call_id)
    }

    pub fn caller(caller_id: impl Into<String>) -> Result<Self, OpaqueProviderItemError> {
        Self::new("caller", caller_id)
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn target_id(&self) -> &str {
        &self.target_id
    }
}

#[derive(Deserialize)]
struct ProviderItemRelationWire {
    kind: String,
    target_id: String,
}

impl<'de> Deserialize<'de> for ProviderItemRelation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderItemRelationWire::deserialize(deserializer)?;
        Self::new(wire.kind, wire.target_id).map_err(serde::de::Error::custom)
    }
}

/// Aggregate limits for provider-native state retained in one request or response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpaqueProviderBudget {
    max_items: usize,
    max_item_bytes: usize,
    max_total_bytes: usize,
}

impl OpaqueProviderBudget {
    pub const fn new(max_items: usize, max_item_bytes: usize, max_total_bytes: usize) -> Self {
        Self {
            max_items,
            max_item_bytes,
            max_total_bytes,
        }
    }

    pub const fn max_items(self) -> usize {
        self.max_items
    }

    pub const fn max_item_bytes(self) -> usize {
        self.max_item_bytes
    }

    pub const fn max_total_bytes(self) -> usize {
        self.max_total_bytes
    }

    pub fn validate<'a>(
        self,
        items: impl IntoIterator<Item = &'a OpaqueProviderItem>,
    ) -> Result<(), OpaqueProviderItemError> {
        let mut item_count = 0usize;
        let mut total_bytes = 0usize;
        for item in items {
            item_count =
                item_count
                    .checked_add(1)
                    .ok_or(OpaqueProviderItemError::TooManyItems {
                        actual: usize::MAX,
                        maximum: self.max_items,
                    })?;
            if item_count > self.max_items {
                return Err(OpaqueProviderItemError::TooManyItems {
                    actual: item_count,
                    maximum: self.max_items,
                });
            }
            if item.encoded_json_bytes > self.max_item_bytes {
                return Err(OpaqueProviderItemError::TooLarge {
                    actual: item.encoded_json_bytes,
                    maximum: self.max_item_bytes,
                });
            }
            total_bytes = total_bytes.checked_add(item.encoded_json_bytes).ok_or(
                OpaqueProviderItemError::CollectionTooLarge {
                    actual: usize::MAX,
                    maximum: self.max_total_bytes,
                },
            )?;
            if total_bytes > self.max_total_bytes {
                return Err(OpaqueProviderItemError::CollectionTooLarge {
                    actual: total_bytes,
                    maximum: self.max_total_bytes,
                });
            }
        }
        Ok(())
    }
}

impl Default for OpaqueProviderBudget {
    fn default() -> Self {
        Self::new(
            DEFAULT_OPAQUE_ITEM_COUNT_LIMIT,
            DEFAULT_OPAQUE_ITEM_LIMIT,
            DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT,
        )
    }
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
    #[error("opaque provider item {field} must not be empty or contain control characters")]
    InvalidField { field: &'static str },
    #[error("opaque provider item has {actual} relations; maximum is {maximum}")]
    TooManyRelations { actual: usize, maximum: usize },
    #[error("opaque provider item envelope is {actual} bytes; maximum is {maximum}")]
    TooLarge { actual: usize, maximum: usize },
    #[error("opaque provider item collection has {actual} items; maximum is {maximum}")]
    TooManyItems { actual: usize, maximum: usize },
    #[error("opaque provider item collection is {actual} bytes; maximum is {maximum}")]
    CollectionTooLarge { actual: usize, maximum: usize },
    #[error("failed to measure opaque provider item: {0}")]
    Serialization(String),
}

/// Builder for a bounded provider-native item envelope.
#[derive(Debug)]
pub struct OpaqueProviderItemBuilder {
    provenance: ProviderProvenance,
    kind: String,
    item_id: Option<String>,
    relations: Vec<ProviderItemRelation>,
    data: Value,
    maximum_bytes: usize,
}

impl OpaqueProviderItemBuilder {
    pub fn item_id(mut self, item_id: impl Into<String>) -> Self {
        self.item_id = Some(item_id.into());
        self
    }

    pub fn relation(mut self, relation: ProviderItemRelation) -> Self {
        self.relations.push(relation);
        self
    }

    pub fn relations(mut self, relations: impl IntoIterator<Item = ProviderItemRelation>) -> Self {
        self.relations.extend(relations);
        self
    }

    /// Tighten the default per-item bound for this item.
    pub fn maximum_bytes(mut self, maximum_bytes: usize) -> Self {
        self.maximum_bytes = maximum_bytes.min(DEFAULT_OPAQUE_ITEM_LIMIT);
        self
    }

    pub fn build(self) -> Result<OpaqueProviderItem, OpaqueProviderItemError> {
        build_opaque_provider_item(
            self.provenance,
            self.kind,
            self.item_id,
            self.relations,
            self.data,
            self.maximum_bytes,
        )
    }
}

impl OpaqueProviderItem {
    pub fn builder(
        provenance: ProviderProvenance,
        kind: impl Into<String>,
        data: Value,
    ) -> OpaqueProviderItemBuilder {
        OpaqueProviderItemBuilder {
            provenance,
            kind: kind.into(),
            item_id: None,
            relations: Vec::new(),
            data,
            maximum_bytes: DEFAULT_OPAQUE_ITEM_LIMIT,
        }
    }

    pub fn new(
        provenance: ProviderProvenance,
        kind: impl Into<String>,
        data: Value,
    ) -> Result<Self, OpaqueProviderItemError> {
        Self::builder(provenance, kind, data).build()
    }

    pub fn with_limit(
        provenance: ProviderProvenance,
        kind: impl Into<String>,
        data: Value,
        maximum: usize,
    ) -> Result<Self, OpaqueProviderItemError> {
        Self::builder(provenance, kind, data)
            .maximum_bytes(maximum)
            .build()
    }

    pub fn provenance(&self) -> &ProviderProvenance {
        &self.provenance
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn item_id(&self) -> Option<&str> {
        self.item_id.as_deref()
    }

    pub fn relations(&self) -> &[ProviderItemRelation] {
        &self.relations
    }

    pub fn data(&self) -> &Value {
        &self.data
    }

    pub fn encoded_json_bytes(&self) -> usize {
        self.encoded_json_bytes
    }
}

#[derive(Serialize, Deserialize)]
struct OpaqueProviderItemWire {
    provenance: ProviderProvenance,
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    item_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    relations: Vec<ProviderItemRelation>,
    data: Value,
}

impl<'de> Deserialize<'de> for OpaqueProviderItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = OpaqueProviderItemWire::deserialize(deserializer)?;
        build_opaque_provider_item(
            wire.provenance,
            wire.kind,
            wire.item_id,
            wire.relations,
            wire.data,
            DEFAULT_OPAQUE_ITEM_LIMIT,
        )
        .map_err(serde::de::Error::custom)
    }
}

fn build_opaque_provider_item(
    provenance: ProviderProvenance,
    kind: String,
    item_id: Option<String>,
    relations: Vec<ProviderItemRelation>,
    data: Value,
    maximum: usize,
) -> Result<OpaqueProviderItem, OpaqueProviderItemError> {
    if kind.trim().is_empty() {
        return Err(OpaqueProviderItemError::EmptyKind);
    }
    validate_opaque_field(&kind, "kind", MAX_OPAQUE_KIND_BYTES)?;
    if provenance.protocol.trim().is_empty() {
        return Err(OpaqueProviderItemError::EmptyProtocol);
    }
    validate_opaque_field(
        &provenance.protocol,
        "provenance.protocol",
        MAX_OPAQUE_PROTOCOL_BYTES,
    )?;
    if let Some(platform) = &provenance.platform {
        validate_opaque_field(platform, "provenance.platform", MAX_OPAQUE_PLATFORM_BYTES)?;
    }
    if let Some(item_id) = &item_id {
        validate_opaque_field(item_id, "item_id", MAX_OPAQUE_ITEM_ID_BYTES)?;
    }
    if relations.len() > MAX_OPAQUE_RELATIONS {
        return Err(OpaqueProviderItemError::TooManyRelations {
            actual: relations.len(),
            maximum: MAX_OPAQUE_RELATIONS,
        });
    }
    let wire = OpaqueProviderItemWire {
        provenance,
        kind,
        item_id,
        relations,
        data,
    };
    let actual = serde_json::to_vec(&wire)
        .map_err(|error| OpaqueProviderItemError::Serialization(error.to_string()))?
        .len();
    if actual > maximum {
        return Err(OpaqueProviderItemError::TooLarge { actual, maximum });
    }
    Ok(OpaqueProviderItem {
        provenance: wire.provenance,
        kind: wire.kind,
        item_id: wire.item_id,
        relations: wire.relations,
        data: wire.data,
        encoded_json_bytes: actual,
    })
}

fn validate_opaque_field(
    value: &str,
    field: &'static str,
    maximum: usize,
) -> Result<(), OpaqueProviderItemError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(OpaqueProviderItemError::InvalidField { field });
    }
    if value.len() > maximum {
        return Err(OpaqueProviderItemError::FieldTooLarge { field, maximum });
    }
    Ok(())
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
    Bytes(Bytes),
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
        self.validate_with_opaque_budget(OpaqueProviderBudget::default())
    }

    /// Validate with an explicit aggregate budget for retained provider state.
    pub fn validate_with_opaque_budget(
        &self,
        opaque_budget: OpaqueProviderBudget,
    ) -> Result<(), LanguageRequestError> {
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
        opaque_budget.validate(self.messages.iter().flat_map(|message| {
            message.content.iter().filter_map(|part| match part {
                ContentPart::ProviderOpaque(item) => Some(item),
                _ => None,
            })
        }))?;
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
    #[error(transparent)]
    OpaqueProviderItems(#[from] OpaqueProviderItemError),
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
    Error,
    Cancelled,
    Other(String),
}

/// Why a final language response ended before normal completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LanguageIncompleteReason {
    MaxOutputTokens,
    ContentFilter,
    Other(String),
}

/// Provider-neutral terminal state of one language response resource.
///
/// Queued and in-progress provider resources remain provider extensions. The
/// stable language family returns only terminal response states.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LanguageResponseStatus {
    #[default]
    Completed,
    Incomplete {
        reason: Option<LanguageIncompleteReason>,
    },
    Failed,
    Cancelled,
}

/// Stable warning categories emitted without changing call success semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WarningKind {
    UnknownModel,
    DeprecatedModel,
    RetiredModel,
    RollingModelAlias,
    UnsupportedOption,
    IgnoredOption,
    PartialResult,
    Provider { code: PublicDiagnosticText },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Warning {
    kind: WarningKind,
    message: PublicDiagnosticText,
}

impl Warning {
    pub fn new(kind: WarningKind, message: impl Into<PublicDiagnosticText>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn provider(
        code: impl Into<PublicDiagnosticText>,
        message: impl Into<PublicDiagnosticText>,
    ) -> Self {
        Self::new(WarningKind::Provider { code: code.into() }, message)
    }

    pub fn kind(&self) -> &WarningKind {
        &self.kind
    }

    pub fn message(&self) -> &str {
        self.message.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LanguageResponse {
    id: Option<String>,
    model: Option<ModelId>,
    status: LanguageResponseStatus,
    content: Vec<ContentPart>,
    finish_reason: FinishReason,
    usage: Usage,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<Warning>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    provider: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LanguageResponseError {
    #[error("language response status and finish reason are inconsistent")]
    StatusFinishReasonMismatch,
    #[error("custom incomplete reason must not be empty")]
    EmptyIncompleteReason,
    #[error(transparent)]
    OpaqueProviderItems(#[from] OpaqueProviderItemError),
}

impl LanguageResponse {
    pub fn new(
        status: LanguageResponseStatus,
        content: Vec<ContentPart>,
        finish_reason: FinishReason,
        usage: Usage,
    ) -> Result<Self, LanguageResponseError> {
        let response = Self {
            id: None,
            model: None,
            status,
            content,
            finish_reason,
            usage,
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };
        response.validate()?;
        Ok(response)
    }

    pub fn completed(
        content: Vec<ContentPart>,
        finish_reason: FinishReason,
        usage: Usage,
    ) -> Result<Self, LanguageResponseError> {
        Self::new(
            LanguageResponseStatus::Completed,
            content,
            finish_reason,
            usage,
        )
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    pub fn with_provider_metadata(mut self, provider: BTreeMap<String, Value>) -> Self {
        self.provider = provider;
        self
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn model(&self) -> Option<&ModelId> {
        self.model.as_ref()
    }

    pub fn status(&self) -> &LanguageResponseStatus {
        &self.status
    }

    pub fn content(&self) -> &[ContentPart] {
        &self.content
    }

    pub fn finish_reason(&self) -> &FinishReason {
        &self.finish_reason
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    pub fn provider_metadata(&self) -> &BTreeMap<String, Value> {
        &self.provider
    }

    pub fn validate(&self) -> Result<(), LanguageResponseError> {
        self.validate_with_opaque_budget(OpaqueProviderBudget::default())
    }

    pub fn validate_with_opaque_budget(
        &self,
        opaque_budget: OpaqueProviderBudget,
    ) -> Result<(), LanguageResponseError> {
        validate_response_status(&self.status, &self.finish_reason)?;
        opaque_budget.validate(self.content.iter().filter_map(|part| match part {
            ContentPart::ProviderOpaque(item) => Some(item),
            _ => None,
        }))?;
        Ok(())
    }
}

#[derive(Deserialize)]
struct LanguageResponseWire {
    id: Option<String>,
    model: Option<ModelId>,
    #[serde(default)]
    status: LanguageResponseStatus,
    content: Vec<ContentPart>,
    finish_reason: FinishReason,
    usage: Usage,
    #[serde(default)]
    warnings: Vec<Warning>,
    #[serde(default)]
    provider: BTreeMap<String, Value>,
}

impl<'de> Deserialize<'de> for LanguageResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LanguageResponseWire::deserialize(deserializer)?;
        let mut response = Self::new(wire.status, wire.content, wire.finish_reason, wire.usage)
            .map_err(serde::de::Error::custom)?;
        response.id = wire.id;
        response.model = wire.model;
        response.warnings = wire.warnings;
        response.provider = wire.provider;
        Ok(response)
    }
}

fn validate_response_status(
    status: &LanguageResponseStatus,
    finish_reason: &FinishReason,
) -> Result<(), LanguageResponseError> {
    let valid = match status {
        LanguageResponseStatus::Completed => !matches!(
            finish_reason,
            FinishReason::Length
                | FinishReason::ContentFilter
                | FinishReason::Error
                | FinishReason::Cancelled
        ),
        LanguageResponseStatus::Incomplete { reason } => match reason {
            Some(LanguageIncompleteReason::MaxOutputTokens) => {
                matches!(finish_reason, FinishReason::Length)
            }
            Some(LanguageIncompleteReason::ContentFilter) => {
                matches!(finish_reason, FinishReason::ContentFilter)
            }
            Some(LanguageIncompleteReason::Other(reason)) => {
                if reason.trim().is_empty() {
                    return Err(LanguageResponseError::EmptyIncompleteReason);
                }
                matches!(
                    finish_reason,
                    FinishReason::Length | FinishReason::ContentFilter | FinishReason::Other(_)
                )
            }
            None => matches!(
                finish_reason,
                FinishReason::Length | FinishReason::ContentFilter | FinishReason::Other(_)
            ),
        },
        LanguageResponseStatus::Failed => matches!(finish_reason, FinishReason::Error),
        LanguageResponseStatus::Cancelled => matches!(finish_reason, FinishReason::Cancelled),
    };
    if valid {
        Ok(())
    } else {
        Err(LanguageResponseError::StatusFinishReasonMismatch)
    }
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
        let item = OpaqueProviderItem::builder(
            provenance(),
            "function_call",
            json!({
                "id": "item_1",
                "call_id": "call_1",
                "caller": {"type": "program", "caller_id": "program_1"}
            }),
        )
        .item_id("item_1")
        .relation(ProviderItemRelation::call("call_1").unwrap())
        .relation(ProviderItemRelation::caller("program_1").unwrap())
        .build()
        .unwrap();

        let encoded = serde_json::to_value(&item).unwrap();
        let decoded: OpaqueProviderItem = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, item);
        assert_eq!(decoded.provenance().protocol, "responses");
        assert_eq!(decoded.item_id(), Some("item_1"));
        assert_eq!(decoded.relations()[0].kind(), "call");
        assert_eq!(decoded.relations()[0].target_id(), "call_1");
        assert_eq!(decoded.relations()[1].kind(), "caller");
        assert_eq!(decoded.relations()[1].target_id(), "program_1");
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
    fn opaque_collection_budget_bounds_count_and_total_bytes() {
        let item = OpaqueProviderItem::new(
            provenance(),
            "reasoning",
            json!({"encrypted_content": "opaque"}),
        )
        .unwrap();
        let item_bytes = item.encoded_json_bytes();

        OpaqueProviderBudget::new(1, item_bytes, item_bytes)
            .validate([&item])
            .unwrap();
        assert!(matches!(
            OpaqueProviderBudget::new(1, item_bytes, item_bytes * 2).validate([&item, &item]),
            Err(OpaqueProviderItemError::TooManyItems { .. })
        ));
        assert!(matches!(
            OpaqueProviderBudget::new(2, item_bytes, item_bytes.saturating_sub(1))
                .validate([&item]),
            Err(OpaqueProviderItemError::CollectionTooLarge { .. })
        ));
    }

    #[test]
    fn language_request_applies_opaque_budget_without_parsing_payloads() {
        let first = OpaqueProviderItem::new(provenance(), "program", json!({"wire": 1})).unwrap();
        let second =
            OpaqueProviderItem::new(provenance(), "program_output", json!({"wire": 2})).unwrap();
        let request = LanguageRequest::new(vec![Message {
            role: MessageRole::Assistant,
            content: vec![
                ContentPart::ProviderOpaque(first),
                ContentPart::ProviderOpaque(second),
            ],
        }]);

        assert!(matches!(
            request.validate_with_opaque_budget(OpaqueProviderBudget::new(
                1,
                DEFAULT_OPAQUE_ITEM_LIMIT,
                DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT,
            )),
            Err(LanguageRequestError::OpaqueProviderItems(
                OpaqueProviderItemError::TooManyItems { .. }
            ))
        ));
    }

    #[test]
    fn response_status_is_terminal_and_strictly_matches_finish_reason() {
        let incomplete = LanguageResponse::new(
            LanguageResponseStatus::Incomplete {
                reason: Some(LanguageIncompleteReason::MaxOutputTokens),
            },
            Vec::new(),
            FinishReason::Length,
            Usage::default(),
        )
        .unwrap();
        let decoded: LanguageResponse =
            serde_json::from_value(serde_json::to_value(&incomplete).unwrap()).unwrap();
        assert!(matches!(
            decoded.status(),
            LanguageResponseStatus::Incomplete {
                reason: Some(LanguageIncompleteReason::MaxOutputTokens)
            }
        ));

        assert_eq!(
            LanguageResponse::new(
                LanguageResponseStatus::Failed,
                Vec::new(),
                FinishReason::Stop,
                Usage::default(),
            ),
            Err(LanguageResponseError::StatusFinishReasonMismatch)
        );
        assert_eq!(
            LanguageResponse::completed(Vec::new(), FinishReason::Length, Usage::default()),
            Err(LanguageResponseError::StatusFinishReasonMismatch)
        );
        assert_eq!(
            LanguageResponse::new(
                LanguageResponseStatus::Cancelled,
                Vec::new(),
                FinishReason::Cancelled,
                Usage::default(),
            )
            .unwrap()
            .status(),
            &LanguageResponseStatus::Cancelled
        );
    }

    #[test]
    fn language_response_applies_an_explicit_opaque_collection_budget() {
        let response = LanguageResponse::completed(
            vec![
                ContentPart::ProviderOpaque(
                    OpaqueProviderItem::new(provenance(), "program", json!({"wire": 1})).unwrap(),
                ),
                ContentPart::ProviderOpaque(
                    OpaqueProviderItem::new(provenance(), "program_output", json!({"wire": 2}))
                        .unwrap(),
                ),
            ],
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap();

        assert!(matches!(
            response.validate_with_opaque_budget(OpaqueProviderBudget::new(
                1,
                DEFAULT_OPAQUE_ITEM_LIMIT,
                DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT,
            )),
            Err(LanguageResponseError::OpaqueProviderItems(
                OpaqueProviderItemError::TooManyItems { .. }
            ))
        ));
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
