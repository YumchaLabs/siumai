use crate::provider::{ModelId, PlatformId, ProtocolId, ProviderId, ProviderScope, ReplayDomain};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::fmt;
use thiserror::Error;
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
            .field("kind", &self.kind)
            .field("has_item_id", &self.item_id.is_some())
            .field("relation_count", &self.relations.len())
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
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ProviderItemRelation {
    kind: String,
    target_id: String,
}

impl fmt::Debug for ProviderItemRelation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderItemRelation")
            .field("kind", &self.kind)
            .field("target_id_bytes", &self.target_id.len())
            .finish()
    }
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
            .field("kind", &self.kind)
            .field("has_item_id", &self.item_id.is_some())
            .field("relation_count", &self.relations.len())
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
