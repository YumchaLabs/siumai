//! Canonical provider-neutral language request and response content.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt;

use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::annotations::{
    ContentAnnotationTarget, ContentAnnotations, MessageAnnotationTarget, MessageAnnotations,
    ProviderAnnotationBudget, ProviderAnnotationError, ProviderAnnotationUsage,
    TypedProviderAnnotation,
};
use crate::error::PublicDiagnosticText;
use crate::provider::{ModelId, PlatformId, ProtocolId, ProviderId, ProviderScope, ReplayDomain};
use crate::tool::{ToolCall, ToolResult, ToolSpec};
use crate::usage::Usage;

/// Default per-item bound leaves room for encrypted reasoning and provider tool
/// payloads while remaining small enough to reject unbounded history growth.
pub const DEFAULT_OPAQUE_ITEM_LIMIT: usize = 1024 * 1024;
pub const DEFAULT_OPAQUE_ITEM_COUNT_LIMIT: usize = 128;
/// Default aggregate bound for one request or response's provider-native state.
pub const DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT: usize = 16 * 1024 * 1024;
const MAX_OPAQUE_KIND_BYTES: usize = 256;
const MAX_OPAQUE_ITEM_ID_BYTES: usize = 512;
const MAX_OPAQUE_RELATION_KIND_BYTES: usize = 128;
const MAX_OPAQUE_RELATION_TARGET_BYTES: usize = 512;
const MAX_OPAQUE_RELATIONS: usize = 32;

/// Provenance required to replay or project provider-native state safely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderProvenance {
    #[serde(flatten)]
    scope: ProviderScope,
    model: ModelId,
}

impl ProviderProvenance {
    pub fn from_scope(
        scope: &ProviderScope,
        model: ModelId,
    ) -> Result<Self, ProviderProvenanceError> {
        if scope.protocol().is_none() {
            return Err(ProviderProvenanceError::MissingProtocol);
        }
        if scope.replay_domain().is_none() {
            return Err(ProviderProvenanceError::MissingReplayDomain);
        }
        Ok(Self {
            scope: scope.clone(),
            model,
        })
    }

    pub fn scope(&self) -> &ProviderScope {
        &self.scope
    }

    pub fn provider(&self) -> &ProviderId {
        self.scope.provider_id()
    }

    pub fn platform(&self) -> Option<&PlatformId> {
        self.scope.platform()
    }

    pub fn protocol(&self) -> &ProtocolId {
        self.scope
            .protocol()
            .expect("validated provider provenance always has a protocol")
    }

    pub fn replay_domain(&self) -> &ReplayDomain {
        self.scope
            .replay_domain()
            .expect("validated provider provenance always has a replay domain")
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn matches_replay_target(&self, target: &ProviderScope) -> bool {
        self.scope.shares_replay_domain(target)
    }
}

#[derive(Deserialize)]
struct ProviderProvenanceWire {
    #[serde(flatten)]
    scope: ProviderScope,
    model: ModelId,
}

impl<'de> Deserialize<'de> for ProviderProvenance {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderProvenanceWire::deserialize(deserializer)?;
        Self::from_scope(&wire.scope, wire.model).map_err(serde::de::Error::custom)
    }
}

/// A bounded provider-native item retained without pretending it is portable.
#[derive(Clone, PartialEq, Serialize)]
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

impl fmt::Debug for OpaqueProviderItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpaqueProviderItem")
            .field("provenance", &self.provenance)
            .field("kind", &self.kind)
            .field("item_id", &self.item_id)
            .field("relations", &self.relations)
            .field("data", &"<redacted>")
            .field("encoded_json_bytes", &self.encoded_json_bytes)
            .finish()
    }
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

/// Aggregate limits applied while validating one language request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageRequestBudget {
    opaque_provider_items: OpaqueProviderBudget,
    provider_annotations: ProviderAnnotationBudget,
}

impl LanguageRequestBudget {
    pub const fn new(
        opaque_provider_items: OpaqueProviderBudget,
        provider_annotations: ProviderAnnotationBudget,
    ) -> Self {
        Self {
            opaque_provider_items,
            provider_annotations,
        }
    }

    pub const fn opaque_provider_items(self) -> OpaqueProviderBudget {
        self.opaque_provider_items
    }

    pub const fn provider_annotations(self) -> ProviderAnnotationBudget {
        self.provider_annotations
    }
}

impl Default for LanguageRequestBudget {
    fn default() -> Self {
        Self::new(
            OpaqueProviderBudget::default(),
            ProviderAnnotationBudget::default(),
        )
    }
}

/// Invalid provenance for replay-critical provider-native state.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ProviderProvenanceError {
    #[error("provider provenance requires an exact protocol")]
    MissingProtocol,
    #[error("provider provenance requires an explicit non-secret replay domain")]
    MissingReplayDomain,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum OpaqueProviderItemError {
    #[error("opaque provider item kind must not be empty")]
    EmptyKind,
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
pub struct OpaqueProviderItemBuilder {
    provenance: ProviderProvenance,
    kind: String,
    item_id: Option<String>,
    relations: Vec<ProviderItemRelation>,
    data: Value,
    maximum_bytes: usize,
}

impl fmt::Debug for OpaqueProviderItemBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpaqueProviderItemBuilder")
            .field("provenance", &self.provenance)
            .field("kind", &self.kind)
            .field("item_id", &self.item_id)
            .field("relations", &self.relations)
            .field("data", &"<redacted>")
            .field("maximum_bytes", &self.maximum_bytes)
            .finish()
    }
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
    role: MessageRole,
    content: Vec<MessagePart>,
    #[serde(
        default,
        rename = "providerAnnotations",
        skip_serializing_if = "MessageAnnotations::is_empty"
    )]
    provider_annotations: MessageAnnotations,
}

/// One request content part plus provider-owned durable annotations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessagePart {
    content: ContentPart,
    #[serde(
        default,
        rename = "providerAnnotations",
        skip_serializing_if = "ContentAnnotations::is_empty"
    )]
    provider_annotations: ContentAnnotations,
}

impl MessagePart {
    pub fn new(content: ContentPart) -> Self {
        Self::from_parts(content, ContentAnnotations::default())
    }

    /// Rebuild a content node while preserving previously validated annotations.
    pub fn from_parts(content: ContentPart, provider_annotations: ContentAnnotations) -> Self {
        Self {
            content,
            provider_annotations,
        }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self::new(ContentPart::Text { text: text.into() })
    }

    pub fn content(&self) -> &ContentPart {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut ContentPart {
        &mut self.content
    }

    pub fn annotations(&self) -> &ContentAnnotations {
        &self.provider_annotations
    }

    pub fn with_provider_annotation<T>(
        mut self,
        annotation: &T,
    ) -> Result<Self, ProviderAnnotationError>
    where
        T: TypedProviderAnnotation<Target = ContentAnnotationTarget>,
    {
        self.provider_annotations.insert(annotation)?;
        Ok(self)
    }

    pub fn into_parts(self) -> (ContentPart, ContentAnnotations) {
        (self.content, self.provider_annotations)
    }
}

impl From<ContentPart> for MessagePart {
    fn from(content: ContentPart) -> Self {
        Self::new(content)
    }
}

impl Message {
    pub fn new<I, Part>(role: MessageRole, content: I) -> Self
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        Self::from_parts(role, content, MessageAnnotations::default())
    }

    /// Rebuild a message while preserving previously validated annotations.
    pub fn from_parts<I, Part>(
        role: MessageRole,
        content: I,
        provider_annotations: MessageAnnotations,
    ) -> Self
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        Self {
            role,
            content: content.into_iter().map(Into::into).collect(),
            provider_annotations,
        }
    }

    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self::new(role, [MessagePart::text(text)])
    }

    /// Construct a system instruction containing portable text.
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(MessageRole::System, text)
    }

    /// Construct a developer instruction containing portable text.
    pub fn developer(text: impl Into<String>) -> Self {
        Self::text(MessageRole::Developer, text)
    }

    /// Construct a user message containing portable text.
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(MessageRole::User, text)
    }

    /// Construct an assistant message containing portable text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(MessageRole::Assistant, text)
    }

    /// Construct and validate a user message containing text and/or input media.
    pub fn user_parts<I, Part>(content: I) -> Result<Self, MessageValidationError>
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        Self::validated(MessageRole::User, content)
    }

    /// Construct and validate assistant history or replay content.
    pub fn assistant_parts<I, Part>(content: I) -> Result<Self, MessageValidationError>
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        Self::validated(MessageRole::Assistant, content)
    }

    /// Construct a canonical message for one tool result.
    pub fn tool_result(result: ToolResult) -> Self {
        Self::new(MessageRole::Tool, [ContentPart::ToolResult(result)])
    }

    /// Construct and validate a canonical message for one or more tool results.
    pub fn tool_results<I>(results: I) -> Result<Self, MessageValidationError>
    where
        I: IntoIterator<Item = ToolResult>,
    {
        Self::validated(
            MessageRole::Tool,
            results.into_iter().map(ContentPart::ToolResult),
        )
    }

    fn validated<I, Part>(role: MessageRole, content: I) -> Result<Self, MessageValidationError>
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        let message = Self::new(role, content);
        message.validate()?;
        Ok(message)
    }

    /// Validate the provider-neutral role/content contract for this message.
    pub fn validate(&self) -> Result<(), MessageValidationError> {
        for (content_index, part) in self.content.iter().enumerate() {
            if !content_allowed_for_role(self.role, part.content()) {
                return Err(MessageValidationError::ContentNotAllowed {
                    role: self.role,
                    content_index,
                    content_kind: content_kind(part.content()),
                });
            }
        }
        Ok(())
    }

    pub fn role(&self) -> MessageRole {
        self.role
    }

    pub fn content(&self) -> &[MessagePart] {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut Vec<MessagePart> {
        &mut self.content
    }

    pub fn annotations(&self) -> &MessageAnnotations {
        &self.provider_annotations
    }

    pub fn with_provider_annotation<T>(
        mut self,
        annotation: &T,
    ) -> Result<Self, ProviderAnnotationError>
    where
        T: TypedProviderAnnotation<Target = MessageAnnotationTarget>,
    {
        self.provider_annotations.insert(annotation)?;
        Ok(self)
    }

    pub fn into_parts(self) -> (MessageRole, Vec<MessagePart>, MessageAnnotations) {
        (self.role, self.content, self.provider_annotations)
    }
}

/// Invalid provider-neutral message role/content combination.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum MessageValidationError {
    #[error("{content_kind} content at index {content_index} is not allowed for role {role:?}")]
    ContentNotAllowed {
        role: MessageRole,
        content_index: usize,
        content_kind: &'static str,
    },
}

/// Response-only content omitted while projecting an assistant response into history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AssistantHistoryOmissionKind {
    Citation,
    Refusal,
    ToolResult,
}

/// One observable omission made by assistant-history projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantHistoryOmission {
    content_index: usize,
    kind: AssistantHistoryOmissionKind,
}

impl AssistantHistoryOmission {
    pub const fn content_index(self) -> usize {
        self.content_index
    }

    pub const fn kind(self) -> AssistantHistoryOmissionKind {
        self.kind
    }
}

/// A role-safe assistant history message and its explicit projection diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantHistoryProjection {
    message: Option<Message>,
    omissions: Vec<AssistantHistoryOmission>,
}

impl AssistantHistoryProjection {
    pub fn message(&self) -> Option<&Message> {
        self.message.as_ref()
    }

    pub fn omissions(&self) -> &[AssistantHistoryOmission] {
        &self.omissions
    }

    pub fn into_parts(self) -> (Option<Message>, Vec<AssistantHistoryOmission>) {
        (self.message, self.omissions)
    }

    pub fn into_message(self) -> Option<Message> {
        self.message
    }
}

fn content_allowed_for_role(role: MessageRole, content: &ContentPart) -> bool {
    match role {
        MessageRole::System | MessageRole::Developer => {
            matches!(content, ContentPart::Text { .. })
        }
        MessageRole::User => matches!(content, ContentPart::Text { .. } | ContentPart::Media(_)),
        MessageRole::Assistant => matches!(
            content,
            ContentPart::Text { .. }
                | ContentPart::Reasoning { .. }
                | ContentPart::Media(_)
                | ContentPart::ToolCall(_)
                | ContentPart::ProviderOpaque(_)
        ),
        MessageRole::Tool => matches!(content, ContentPart::ToolResult(_)),
    }
}

fn content_kind(content: &ContentPart) -> &'static str {
    match content {
        ContentPart::Text { .. } => "text",
        ContentPart::Reasoning { .. } => "reasoning",
        ContentPart::Media(_) => "media",
        ContentPart::Citation(_) => "citation",
        ContentPart::Refusal { .. } => "refusal",
        ContentPart::ToolCall(_) => "tool call",
        ContentPart::ToolResult(_) => "tool result",
        ContentPart::ProviderOpaque(_) => "provider-opaque",
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
        self.validate_with_budget(LanguageRequestBudget::default())
    }

    /// Validate with explicit aggregate budgets for retained provider state.
    pub fn validate_with_budget(
        &self,
        budget: LanguageRequestBudget,
    ) -> Result<(), LanguageRequestError> {
        self.generation.validate()?;
        for (message_index, message) in self.messages.iter().enumerate() {
            message
                .validate()
                .map_err(|source| LanguageRequestError::InvalidMessage {
                    message_index,
                    source,
                })?;
        }
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
        budget
            .opaque_provider_items
            .validate(self.messages.iter().flat_map(|message| {
                message
                    .content
                    .iter()
                    .filter_map(|part| match part.content() {
                        ContentPart::ProviderOpaque(item) => Some(item),
                        _ => None,
                    })
            }))?;

        let mut annotation_usage = ProviderAnnotationUsage::default();
        for message in &self.messages {
            budget
                .provider_annotations
                .validate_node(message.annotations(), &mut annotation_usage)?;
            for part in message.content() {
                budget
                    .provider_annotations
                    .validate_node(part.annotations(), &mut annotation_usage)?;
            }
        }
        for tool in &self.tools {
            budget
                .provider_annotations
                .validate_node(tool.annotations(), &mut annotation_usage)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LanguageRequestError {
    #[error(transparent)]
    Generation(#[from] GenerationConfigError),
    #[error("message {message_index} violates the canonical role/content contract: {source}")]
    InvalidMessage {
        message_index: usize,
        source: MessageValidationError,
    },
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
    #[error(transparent)]
    ProviderAnnotations(#[from] ProviderAnnotationError),
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

    /// Project generated output into canonical assistant history.
    ///
    /// Citations and refusals remain available on this response but are not
    /// request content. The returned diagnostics make those omissions
    /// observable, while replay-critical provider state remains in the
    /// projected message as `ProviderOpaque` content.
    pub fn project_assistant_history(&self) -> AssistantHistoryProjection {
        let mut content = Vec::with_capacity(self.content.len());
        let mut omissions = Vec::new();

        for (content_index, part) in self.content.iter().enumerate() {
            let kind = match part {
                ContentPart::Citation(_) => Some(AssistantHistoryOmissionKind::Citation),
                ContentPart::Refusal { .. } => Some(AssistantHistoryOmissionKind::Refusal),
                ContentPart::ToolResult(_) => Some(AssistantHistoryOmissionKind::ToolResult),
                ContentPart::Text { .. }
                | ContentPart::Reasoning { .. }
                | ContentPart::Media(_)
                | ContentPart::ToolCall(_)
                | ContentPart::ProviderOpaque(_) => {
                    content.push(MessagePart::new(part.clone()));
                    None
                }
            };
            if let Some(kind) = kind {
                omissions.push(AssistantHistoryOmission {
                    content_index,
                    kind,
                });
            }
        }

        let message = (!content.is_empty()).then(|| Message::new(MessageRole::Assistant, content));
        AssistantHistoryProjection { message, omissions }
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
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use crate::annotations::{
        DEFAULT_PROVIDER_ANNOTATION_COLLECTION_BYTE_LIMIT,
        DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT, DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT,
    };
    use crate::tool::ToolOutcome;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct CacheAnnotation {
        cache_control: String,
    }

    impl TypedProviderAnnotation for CacheAnnotation {
        type Target = ContentAnnotationTarget;

        const NAMESPACE: &'static str = "anthropic";
        const API_MODE: Option<&'static str> = Some("messages");
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MessageLabelAnnotation {
        label: String,
    }

    impl TypedProviderAnnotation for MessageLabelAnnotation {
        type Target = MessageAnnotationTarget;

        const NAMESPACE: &'static str = "anthropic";
        const API_MODE: Option<&'static str> = Some("messages");
    }

    fn provenance() -> ProviderProvenance {
        let scope = ProviderScope::new(ProviderId::new("openai").unwrap())
            .with_protocol(ProtocolId::new("responses").unwrap())
            .with_replay_domain(ReplayDomain::official(
                crate::provider::ReplayDomainId::new("openai-public-api").unwrap(),
            ));
        ProviderProvenance::from_scope(&scope, ModelId::new("gpt-future:preview").unwrap()).unwrap()
    }

    #[test]
    fn provider_provenance_deserialization_cannot_bypass_replay_requirements() {
        let missing_protocol = json!({
            "provider": "openai",
            "platform": "public-api",
            "api_mode": "responses",
            "replay_domain": {
                "audience": {"Official": "public-api"}
            },
            "model": "gpt-5.6"
        });
        let missing_domain = json!({
            "provider": "openai",
            "platform": "public-api",
            "protocol": "openai-responses",
            "api_mode": "responses",
            "model": "gpt-5.6"
        });

        assert!(serde_json::from_value::<ProviderProvenance>(missing_protocol).is_err());
        assert!(serde_json::from_value::<ProviderProvenance>(missing_domain).is_err());
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
        assert_eq!(decoded.provenance().protocol().as_str(), "responses");
        assert_eq!(decoded.item_id(), Some("item_1"));
        assert_eq!(decoded.relations()[0].kind(), "call");
        assert_eq!(decoded.relations()[0].target_id(), "call_1");
        assert_eq!(decoded.relations()[1].kind(), "caller");
        assert_eq!(decoded.relations()[1].target_id(), "program_1");
    }

    #[test]
    fn opaque_item_debug_redacts_provider_payload() {
        let builder = OpaqueProviderItem::builder(
            provenance(),
            "encrypted_reasoning",
            json!({"encrypted_content": "provider-secret-canary"}),
        )
        .item_id("reasoning-1");
        let builder_debug = format!("{builder:?}");
        assert!(!builder_debug.contains("provider-secret-canary"));
        assert!(builder_debug.contains("<redacted>"));

        let item = builder.build().unwrap();
        let item_debug = format!("{item:?}");
        assert!(!item_debug.contains("provider-secret-canary"));
        assert!(item_debug.contains("<redacted>"));
        assert!(item_debug.contains("reasoning-1"));
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
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::Assistant,
            [
                ContentPart::ProviderOpaque(first),
                ContentPart::ProviderOpaque(second),
            ],
        )]);

        assert!(matches!(
            request.validate_with_budget(LanguageRequestBudget::new(
                OpaqueProviderBudget::new(
                    1,
                    DEFAULT_OPAQUE_ITEM_LIMIT,
                    DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT,
                ),
                ProviderAnnotationBudget::default(),
            )),
            Err(LanguageRequestError::OpaqueProviderItems(
                OpaqueProviderItemError::TooManyItems { .. }
            ))
        ));
    }

    #[test]
    fn message_parts_are_ergonomic_and_preserve_annotations_through_serde() {
        let expected = CacheAnnotation {
            cache_control: "ephemeral".to_string(),
        };
        let part = MessagePart::text("hello")
            .with_provider_annotation(&expected)
            .unwrap();
        let message = Message::new(MessageRole::User, [part]);

        let decoded: Message =
            serde_json::from_value(serde_json::to_value(&message).unwrap()).unwrap();
        assert_eq!(decoded.role(), MessageRole::User);
        assert!(matches!(
            decoded.content()[0].content(),
            ContentPart::Text { text } if text == "hello"
        ));
        assert_eq!(
            decoded.content()[0]
                .annotations()
                .decode::<CacheAnnotation>()
                .unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn message_rebuild_paths_preserve_node_annotations() {
        let content_annotation = CacheAnnotation {
            cache_control: "ephemeral".to_string(),
        };
        let message_annotation = MessageLabelAnnotation {
            label: "history-entry".to_string(),
        };
        let mut part = MessagePart::text("before")
            .with_provider_annotation(&content_annotation)
            .unwrap();
        *part.content_mut() = ContentPart::Text {
            text: "after".to_string(),
        };
        let message = Message::new(MessageRole::User, [part])
            .with_provider_annotation(&message_annotation)
            .unwrap();

        let (role, content, annotations) = message.into_parts();
        let rebuilt = Message::from_parts(role, content, annotations);

        assert!(matches!(
            rebuilt.content()[0].content(),
            ContentPart::Text { text } if text == "after"
        ));
        assert_eq!(
            rebuilt.content()[0]
                .annotations()
                .decode::<CacheAnnotation>()
                .unwrap(),
            Some(content_annotation)
        );
        assert_eq!(
            rebuilt
                .annotations()
                .decode::<MessageLabelAnnotation>()
                .unwrap(),
            Some(message_annotation)
        );
    }

    #[test]
    fn language_request_budget_counts_annotations_across_nodes() {
        let annotation = CacheAnnotation {
            cache_control: "ephemeral".to_string(),
        };
        let first = MessagePart::text("first")
            .with_provider_annotation(&annotation)
            .unwrap();
        let second = MessagePart::text("second")
            .with_provider_annotation(&annotation)
            .unwrap();
        let entry_bytes = first.annotations().encoded_bytes();
        let request = LanguageRequest::new(vec![Message::new(MessageRole::User, [first, second])]);
        let budget = LanguageRequestBudget::new(
            OpaqueProviderBudget::default(),
            ProviderAnnotationBudget::new(
                DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT,
                1,
                DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT,
                DEFAULT_PROVIDER_ANNOTATION_COLLECTION_BYTE_LIMIT,
            ),
        );

        assert!(matches!(
            request.validate_with_budget(budget),
            Err(LanguageRequestError::ProviderAnnotations(
                ProviderAnnotationError::TooManyEntries { .. }
            ))
        ));

        let total_budget = LanguageRequestBudget::new(
            OpaqueProviderBudget::default(),
            ProviderAnnotationBudget::new(
                DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT,
                2,
                DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT,
                entry_bytes.saturating_mul(2).saturating_sub(1),
            ),
        );
        assert!(matches!(
            request.validate_with_budget(total_budget),
            Err(LanguageRequestError::ProviderAnnotations(
                ProviderAnnotationError::CollectionTooLarge { .. }
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

    #[test]
    fn role_safe_message_constructors_follow_the_canonical_matrix() {
        assert_eq!(Message::system("rules").role(), MessageRole::System);
        assert_eq!(Message::developer("policy").role(), MessageRole::Developer);
        assert_eq!(Message::user("question").role(), MessageRole::User);
        assert_eq!(Message::assistant("answer").role(), MessageRole::Assistant);

        let media = MediaPart {
            media_type: "image/png".to_string(),
            data: MediaData::Bytes(Bytes::from_static(b"image")),
            name: None,
        };
        Message::user_parts([ContentPart::Media(media)])
            .expect("input media is valid user content");
        Message::assistant_parts([ContentPart::Reasoning {
            text: "bounded reasoning".to_string(),
        }])
        .expect("reasoning is valid assistant content");
        Message::tool_result(ToolResult {
            call_id: "call-1".to_string(),
            name: "lookup".to_string(),
            outcome: ToolOutcome::Success { value: json!(1) },
        })
        .validate()
        .expect("tool results are valid tool content");
    }

    #[test]
    fn request_validation_rejects_invalid_role_content_before_encoding() {
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::User,
            [ContentPart::ToolResult(ToolResult {
                call_id: "call-1".to_string(),
                name: "lookup".to_string(),
                outcome: ToolOutcome::Success { value: json!(1) },
            })],
        )]);

        assert!(matches!(
            request.validate(),
            Err(LanguageRequestError::InvalidMessage {
                message_index: 0,
                source: MessageValidationError::ContentNotAllowed {
                    role: MessageRole::User,
                    content_index: 0,
                    content_kind: "tool result",
                },
            })
        ));

        let response_only = LanguageRequest::new(vec![Message::new(
            MessageRole::Assistant,
            [ContentPart::Refusal {
                reason: Some("declined".to_string()),
            }],
        )]);
        assert!(matches!(
            response_only.validate(),
            Err(LanguageRequestError::InvalidMessage { .. })
        ));
    }

    #[test]
    fn assistant_history_projection_is_explicit_and_role_safe() {
        let response = LanguageResponse::completed(
            vec![
                ContentPart::Text {
                    text: "answer".to_string(),
                },
                ContentPart::Citation(Citation {
                    source_id: "source-1".to_string(),
                    title: None,
                    url: None,
                    start: None,
                    end: None,
                    provider: BTreeMap::new(),
                }),
                ContentPart::Refusal {
                    reason: Some("detail omitted".to_string()),
                },
                ContentPart::ProviderOpaque(
                    OpaqueProviderItem::new(
                        provenance(),
                        "response.output",
                        json!({"id": "item-1"}),
                    )
                    .unwrap(),
                ),
            ],
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap();

        let projection = response.project_assistant_history();
        assert_eq!(
            projection
                .omissions()
                .iter()
                .map(|omission| omission.kind())
                .collect::<Vec<_>>(),
            vec![
                AssistantHistoryOmissionKind::Citation,
                AssistantHistoryOmissionKind::Refusal,
            ]
        );
        let message = projection.message().expect("portable history remains");
        assert_eq!(message.content().len(), 2);
        message.validate().expect("projected history is role safe");
    }
}
