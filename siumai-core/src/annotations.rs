//! Typed provider-owned annotations attached to durable request nodes.
//!
//! Annotations are serialized with messages, content parts, and tool specs.
//! They are deliberately separate from request-scoped provider call options.

use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

use serde::de::{DeserializeOwned, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::options::{ProviderOptionError, reject_protected_fields, validate_option_structure};
use crate::provider::{ApiModeId, ProviderId};

/// Default maximum number of provider namespaces attached to one request node.
pub const DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT: usize = 8;
/// Default maximum number of provider annotation entries in one request.
pub const DEFAULT_PROVIDER_ANNOTATION_ENTRY_COUNT_LIMIT: usize = 512;
/// Default maximum encoded JSON size of one complete namespace entry.
pub const DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT: usize = 64 * 1024;
/// Default maximum aggregate encoded JSON size of provider annotations in one request.
pub const DEFAULT_PROVIDER_ANNOTATION_COLLECTION_BYTE_LIMIT: usize = 4 * 1024 * 1024;

mod private {
    pub trait Sealed {}
}

/// The durable request node kind that owns a provider annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProviderAnnotationKind {
    Message,
    Content,
    Tool,
}

/// A sealed marker for one supported provider-annotation target.
#[allow(private_bounds)]
pub trait ProviderAnnotationTarget: private::Sealed + 'static {
    const KIND: ProviderAnnotationKind;
}

/// Marker for annotations attached to a complete message.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MessageAnnotationTarget;

/// Marker for annotations attached to one message content part.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContentAnnotationTarget;

/// Marker for annotations attached to a model-visible tool definition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolAnnotationTarget;

impl private::Sealed for MessageAnnotationTarget {}
impl private::Sealed for ContentAnnotationTarget {}
impl private::Sealed for ToolAnnotationTarget {}

impl ProviderAnnotationTarget for MessageAnnotationTarget {
    const KIND: ProviderAnnotationKind = ProviderAnnotationKind::Message;
}

impl ProviderAnnotationTarget for ContentAnnotationTarget {
    const KIND: ProviderAnnotationKind = ProviderAnnotationKind::Content;
}

impl ProviderAnnotationTarget for ToolAnnotationTarget {
    const KIND: ProviderAnnotationKind = ProviderAnnotationKind::Tool;
}

/// A typed, provider-owned annotation for one durable request-node kind.
///
/// Provider annotation types that are decoded from durable data should use a
/// strict deserializer, such as Serde's `deny_unknown_fields`, and validate all
/// provider-specific relationships in [`Self::validate`]. Validation errors
/// must be safe to expose and must not include annotation field values.
pub trait TypedProviderAnnotation: Serialize {
    /// The single semantic node kind on which this annotation is valid.
    type Target: ProviderAnnotationTarget;

    const NAMESPACE: &'static str;
    /// Exact provider API mode when the annotation is mode-specific.
    const API_MODE: Option<&'static str> = None;

    fn validate(&self) -> Result<(), ProviderAnnotationError> {
        Ok(())
    }
}

/// A validated erased provider annotation.
#[derive(Clone, PartialEq)]
pub struct ProviderAnnotation {
    namespace: ProviderId,
    api_mode: Option<ApiModeId>,
    value: Map<String, Value>,
    encoded_json_bytes: usize,
}

impl fmt::Debug for ProviderAnnotation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderAnnotation")
            .field("namespace", &self.namespace)
            .field("api_mode", &self.api_mode)
            .field("encoded_json_bytes", &self.encoded_json_bytes)
            .finish_non_exhaustive()
    }
}

impl ProviderAnnotation {
    fn typed<T>(value: &T) -> Result<Self, ProviderAnnotationError>
    where
        T: TypedProviderAnnotation,
    {
        value.validate()?;
        let namespace = parse_namespace(T::NAMESPACE)?;
        let api_mode = parse_api_mode(T::API_MODE)?;
        let value =
            serde_json::to_value(value).map_err(|_| ProviderAnnotationError::Serialization {
                namespace: namespace.to_string(),
            })?;
        Self::from_value(namespace, api_mode, value)
    }

    fn from_value(
        namespace: ProviderId,
        api_mode: Option<ApiModeId>,
        value: Value,
    ) -> Result<Self, ProviderAnnotationError> {
        let Value::Object(value) = value else {
            return Err(ProviderAnnotationError::ExpectedObject {
                namespace: namespace.to_string(),
            });
        };
        validate_annotation_object(&namespace, &value)?;
        let encoded_json_bytes = serde_json::to_vec(&ProviderAnnotationEntryWireRef {
            namespace: &namespace,
            api_mode: api_mode.as_ref(),
            value: &value,
        })
        .map_err(|_| ProviderAnnotationError::Serialization {
            namespace: namespace.to_string(),
        })?
        .len();
        if encoded_json_bytes > DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT {
            return Err(ProviderAnnotationError::TooLarge {
                actual: encoded_json_bytes,
                maximum: DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT,
            });
        }
        Ok(Self {
            namespace,
            api_mode,
            value,
            encoded_json_bytes,
        })
    }

    pub fn namespace(&self) -> &ProviderId {
        &self.namespace
    }

    pub fn api_mode(&self) -> Option<&ApiModeId> {
        self.api_mode.as_ref()
    }

    /// Exact encoded size of this annotation as a single-entry namespace map.
    pub fn encoded_json_bytes(&self) -> usize {
        self.encoded_json_bytes
    }
}

/// Provider annotations attached to one durable request node.
///
/// The collection supports Serde so requests can cross process and durable
/// storage boundaries. Deserialization validates the bounded erased envelope;
/// it does not prove that a value matches a provider-owned schema. Providers
/// must call [`Self::decode`] before interpreting their namespace.
#[derive(Clone, PartialEq)]
pub struct ProviderAnnotations<Target: ProviderAnnotationTarget> {
    entries: BTreeMap<ProviderId, ProviderAnnotation>,
    target: PhantomData<fn() -> Target>,
}

pub type MessageAnnotations = ProviderAnnotations<MessageAnnotationTarget>;
pub type ContentAnnotations = ProviderAnnotations<ContentAnnotationTarget>;
pub type ToolAnnotations = ProviderAnnotations<ToolAnnotationTarget>;

impl<Target: ProviderAnnotationTarget> Default for ProviderAnnotations<Target> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            target: PhantomData,
        }
    }
}

impl<Target: ProviderAnnotationTarget> fmt::Debug for ProviderAnnotations<Target> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderAnnotations")
            .field("target", &Target::KIND)
            .field("namespaces", &self.entries.keys().collect::<Vec<_>>())
            .field("encoded_json_bytes", &self.encoded_bytes())
            .finish()
    }
}

impl<Target: ProviderAnnotationTarget> ProviderAnnotations<Target> {
    pub fn insert<T>(&mut self, value: &T) -> Result<(), ProviderAnnotationError>
    where
        T: TypedProviderAnnotation<Target = Target>,
    {
        let annotation = ProviderAnnotation::typed(value)?;
        self.insert_erased(annotation)
    }

    pub fn with<T>(mut self, value: &T) -> Result<Self, ProviderAnnotationError>
    where
        T: TypedProviderAnnotation<Target = Target>,
    {
        self.insert(value)?;
        Ok(self)
    }

    pub fn get(&self, namespace: &ProviderId) -> Option<&ProviderAnnotation> {
        self.entries.get(namespace)
    }

    pub fn namespaces(&self) -> impl ExactSizeIterator<Item = &ProviderId> {
        self.entries.keys()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return a conservative encoded size for all namespace entries on this node.
    ///
    /// Each entry is measured as an exact single-entry namespace map. Summing
    /// those maps includes every namespace and envelope byte and may only
    /// over-count separators when a node carries several namespaces.
    pub fn encoded_bytes(&self) -> usize {
        self.entries.values().fold(0usize, |total, annotation| {
            total.saturating_add(annotation.encoded_json_bytes)
        })
    }

    /// Strictly decode this provider's namespace into its target-specific type.
    ///
    /// Other provider namespaces are intentionally ignored. A matching
    /// namespace with the wrong API mode or runtime shape is an error.
    pub fn decode<T>(&self) -> Result<Option<T>, ProviderAnnotationError>
    where
        T: DeserializeOwned + TypedProviderAnnotation<Target = Target>,
    {
        let namespace = parse_namespace(T::NAMESPACE)?;
        let Some(annotation) = self.entries.get(&namespace) else {
            return Ok(None);
        };
        let expected_api_mode = parse_api_mode(T::API_MODE)?;
        if annotation.api_mode != expected_api_mode {
            return Err(ProviderAnnotationError::ApiModeMismatch {
                namespace: namespace.to_string(),
                expected_api_mode: expected_api_mode.as_ref().map(ToString::to_string),
                actual_api_mode: annotation.api_mode.as_ref().map(ToString::to_string),
            });
        }
        let decoded =
            serde_json::from_value(Value::Object(annotation.value.clone())).map_err(|_| {
                ProviderAnnotationError::InvalidTypedPayload {
                    namespace: namespace.to_string(),
                }
            })?;
        T::validate(&decoded)?;
        Ok(Some(decoded))
    }

    fn insert_erased(
        &mut self,
        annotation: ProviderAnnotation,
    ) -> Result<(), ProviderAnnotationError> {
        if self.entries.contains_key(annotation.namespace()) {
            return Err(ProviderAnnotationError::DuplicateNamespace {
                namespace: annotation.namespace().to_string(),
            });
        }
        if self.entries.len() >= DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT {
            return Err(ProviderAnnotationError::TooManyNamespaces {
                actual: self.entries.len().saturating_add(1),
                maximum: DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT,
            });
        }
        self.entries
            .insert(annotation.namespace.clone(), annotation);
        Ok(())
    }
}

struct ProviderAnnotationWireRef<'a> {
    api_mode: Option<&'a ApiModeId>,
    value: &'a Map<String, Value>,
}

struct ProviderAnnotationEntryWireRef<'a> {
    namespace: &'a ProviderId,
    api_mode: Option<&'a ApiModeId>,
    value: &'a Map<String, Value>,
}

impl Serialize for ProviderAnnotationEntryWireRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(
            self.namespace.as_str(),
            &ProviderAnnotationWireRef {
                api_mode: self.api_mode,
                value: self.value,
            },
        )?;
        map.end()
    }
}

impl Serialize for ProviderAnnotationWireRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let field_count = usize::from(self.api_mode.is_some()) + 1;
        let mut state = serializer.serialize_struct("ProviderAnnotation", field_count)?;
        if let Some(api_mode) = self.api_mode {
            state.serialize_field("apiMode", api_mode)?;
        }
        state.serialize_field("value", self.value)?;
        state.end()
    }
}

impl<Target: ProviderAnnotationTarget> Serialize for ProviderAnnotations<Target> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for (namespace, annotation) in &self.entries {
            map.serialize_entry(
                namespace.as_str(),
                &ProviderAnnotationWireRef {
                    api_mode: annotation.api_mode.as_ref(),
                    value: &annotation.value,
                },
            )?;
        }
        map.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderAnnotationWire {
    #[serde(default, rename = "apiMode")]
    api_mode: Option<String>,
    value: Value,
}

struct ProviderAnnotationsVisitor<Target>(PhantomData<fn() -> Target>);

impl<'de, Target: ProviderAnnotationTarget> Visitor<'de> for ProviderAnnotationsVisitor<Target> {
    type Value = ProviderAnnotations<Target>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a map of provider annotation namespaces")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut annotations = ProviderAnnotations::default();
        while let Some(raw_namespace) = map.next_key::<String>()? {
            let namespace = ProviderId::new(&raw_namespace)
                .map_err(|error| serde::de::Error::custom(error.to_string()))?;
            if annotations.entries.contains_key(&namespace) {
                return Err(serde::de::Error::custom(
                    ProviderAnnotationError::DuplicateNamespace {
                        namespace: namespace.to_string(),
                    },
                ));
            }
            if annotations.entries.len() >= DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT {
                return Err(serde::de::Error::custom(
                    ProviderAnnotationError::TooManyNamespaces {
                        actual: annotations.entries.len().saturating_add(1),
                        maximum: DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT,
                    },
                ));
            }
            let wire = map.next_value::<ProviderAnnotationWire>()?;
            let api_mode = wire
                .api_mode
                .map(ApiModeId::new)
                .transpose()
                .map_err(|error| serde::de::Error::custom(error.to_string()))?;
            let annotation = ProviderAnnotation::from_value(namespace, api_mode, wire.value)
                .map_err(serde::de::Error::custom)?;
            annotations
                .insert_erased(annotation)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(annotations)
    }
}

impl<'de, Target: ProviderAnnotationTarget> Deserialize<'de> for ProviderAnnotations<Target> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ProviderAnnotationsVisitor(PhantomData))
    }
}

/// Aggregate limits for durable provider annotations retained in one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderAnnotationBudget {
    max_namespaces_per_node: usize,
    max_entries: usize,
    max_entry_bytes: usize,
    max_total_bytes: usize,
}

impl ProviderAnnotationBudget {
    pub const fn new(
        max_namespaces_per_node: usize,
        max_entries: usize,
        max_entry_bytes: usize,
        max_total_bytes: usize,
    ) -> Self {
        Self {
            max_namespaces_per_node,
            max_entries,
            max_entry_bytes,
            max_total_bytes,
        }
    }

    pub const fn max_namespaces_per_node(self) -> usize {
        self.max_namespaces_per_node
    }

    pub const fn max_entries(self) -> usize {
        self.max_entries
    }

    pub const fn max_entry_bytes(self) -> usize {
        self.max_entry_bytes
    }

    pub const fn max_total_bytes(self) -> usize {
        self.max_total_bytes
    }

    pub(crate) fn validate_node<Target: ProviderAnnotationTarget>(
        self,
        annotations: &ProviderAnnotations<Target>,
        usage: &mut ProviderAnnotationUsage,
    ) -> Result<(), ProviderAnnotationError> {
        if annotations.entries.len() > self.max_namespaces_per_node {
            return Err(ProviderAnnotationError::TooManyNamespaces {
                actual: annotations.entries.len(),
                maximum: self.max_namespaces_per_node,
            });
        }
        for annotation in annotations.entries.values() {
            if annotation.encoded_json_bytes > self.max_entry_bytes {
                return Err(ProviderAnnotationError::TooLarge {
                    actual: annotation.encoded_json_bytes,
                    maximum: self.max_entry_bytes,
                });
            }
            usage.entries = usage.entries.saturating_add(1);
            if usage.entries > self.max_entries {
                return Err(ProviderAnnotationError::TooManyEntries {
                    actual: usage.entries,
                    maximum: self.max_entries,
                });
            }
            usage.total_bytes = usage
                .total_bytes
                .saturating_add(annotation.encoded_json_bytes);
            if usage.total_bytes > self.max_total_bytes {
                return Err(ProviderAnnotationError::CollectionTooLarge {
                    actual: usage.total_bytes,
                    maximum: self.max_total_bytes,
                });
            }
        }
        Ok(())
    }
}

impl Default for ProviderAnnotationBudget {
    fn default() -> Self {
        Self::new(
            DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT,
            DEFAULT_PROVIDER_ANNOTATION_ENTRY_COUNT_LIMIT,
            DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT,
            DEFAULT_PROVIDER_ANNOTATION_COLLECTION_BYTE_LIMIT,
        )
    }
}

#[derive(Debug, Default)]
pub(crate) struct ProviderAnnotationUsage {
    entries: usize,
    total_bytes: usize,
}

/// Provider annotation validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ProviderAnnotationError {
    #[error("provider annotation namespace is invalid: {0}")]
    InvalidNamespace(String),
    #[error("provider annotation API mode is invalid: {0}")]
    InvalidApiMode(String),
    #[error("provider annotation for `{namespace}` must serialize to an object")]
    ExpectedObject { namespace: String },
    #[error("provider annotation namespace `{namespace}` occurs more than once on one node")]
    DuplicateNamespace { namespace: String },
    #[error("provider annotations contain {actual} namespaces; the per-node maximum is {maximum}")]
    TooManyNamespaces { actual: usize, maximum: usize },
    #[error("provider annotation contains protected field `{path}`")]
    ProtectedField { path: String },
    #[error("provider annotation is {actual} bytes; the maximum is {maximum}")]
    TooLarge { actual: usize, maximum: usize },
    #[error("provider annotation exceeds the maximum JSON nesting depth of {maximum}")]
    TooDeep { maximum: usize },
    #[error("provider annotation exceeds the {maximum}-field limit")]
    TooManyFields { maximum: usize },
    #[error(
        "provider annotation for `{namespace}` has API mode {actual_api_mode:?}, expected {expected_api_mode:?}"
    )]
    ApiModeMismatch {
        namespace: String,
        expected_api_mode: Option<String>,
        actual_api_mode: Option<String>,
    },
    #[error("provider annotation for `{namespace}` does not match its typed schema")]
    InvalidTypedPayload { namespace: String },
    #[error("provider annotations contain {actual} entries; the request maximum is {maximum}")]
    TooManyEntries { actual: usize, maximum: usize },
    #[error("provider annotations total {actual} bytes; the request maximum is {maximum}")]
    CollectionTooLarge { actual: usize, maximum: usize },
    /// Provider-owned validation failure with a sanitized public reason.
    #[error("provider rejected annotation `{path}`: {reason}")]
    Rejected { path: String, reason: String },
    #[error("failed to serialize provider annotation for `{namespace}`")]
    Serialization { namespace: String },
}

fn parse_namespace(value: &str) -> Result<ProviderId, ProviderAnnotationError> {
    ProviderId::new(value)
        .map_err(|error| ProviderAnnotationError::InvalidNamespace(error.to_string()))
}

fn parse_api_mode(value: Option<&str>) -> Result<Option<ApiModeId>, ProviderAnnotationError> {
    value
        .map(ApiModeId::new)
        .transpose()
        .map_err(|error| ProviderAnnotationError::InvalidApiMode(error.to_string()))
}

fn validate_annotation_object(
    namespace: &ProviderId,
    object: &Map<String, Value>,
) -> Result<(), ProviderAnnotationError> {
    validate_option_structure(object).map_err(|error| map_option_error(namespace, error))?;
    reject_protected_fields(object, "").map_err(|error| map_option_error(namespace, error))?;
    Ok(())
}

fn map_option_error(namespace: &ProviderId, error: ProviderOptionError) -> ProviderAnnotationError {
    match error {
        ProviderOptionError::ProtectedField { path } => {
            ProviderAnnotationError::ProtectedField { path }
        }
        ProviderOptionError::TooDeep { maximum } => ProviderAnnotationError::TooDeep { maximum },
        ProviderOptionError::TooManyFields { maximum } => {
            ProviderAnnotationError::TooManyFields { maximum }
        }
        ProviderOptionError::TooLarge { maximum } => ProviderAnnotationError::TooLarge {
            actual: maximum.saturating_add(1),
            maximum,
        },
        _ => ProviderAnnotationError::Serialization {
            namespace: namespace.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value, json};

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ResponsesCacheAnnotation {
        cache_control: String,
    }

    impl TypedProviderAnnotation for ResponsesCacheAnnotation {
        type Target = ContentAnnotationTarget;

        const NAMESPACE: &'static str = "openai";
        const API_MODE: Option<&'static str> = Some("responses");
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ChatCompletionsCacheAnnotation {
        cache_control: String,
    }

    impl TypedProviderAnnotation for ChatCompletionsCacheAnnotation {
        type Target = ContentAnnotationTarget;

        const NAMESPACE: &'static str = "openai";
        const API_MODE: Option<&'static str> = Some("chat-completions");
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MessageAnnotation {
        message_label: String,
    }

    impl TypedProviderAnnotation for MessageAnnotation {
        type Target = MessageAnnotationTarget;

        const NAMESPACE: &'static str = "openai";
        const API_MODE: Option<&'static str> = Some("responses");
    }

    #[derive(Serialize)]
    struct ProtectedAnnotation {
        endpoint: String,
    }

    impl TypedProviderAnnotation for ProtectedAnnotation {
        type Target = ContentAnnotationTarget;

        const NAMESPACE: &'static str = "openai";
    }

    #[derive(Serialize)]
    struct LargeAnnotation {
        payload: String,
    }

    impl TypedProviderAnnotation for LargeAnnotation {
        type Target = ContentAnnotationTarget;

        const NAMESPACE: &'static str = "openai";
    }

    #[test]
    fn typed_annotations_round_trip_and_decode_strictly() {
        let expected = ResponsesCacheAnnotation {
            cache_control: "ephemeral".to_string(),
        };
        let annotations = ContentAnnotations::default().with(&expected).unwrap();

        let encoded = serde_json::to_value(&annotations).unwrap();
        assert_eq!(encoded["openai"]["apiMode"], "responses");
        assert_eq!(encoded["openai"]["value"]["cache_control"], "ephemeral");

        let decoded: ContentAnnotations = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.decode().unwrap(), Some(expected));
    }

    #[test]
    fn foreign_namespaces_are_inert_for_typed_decode() {
        let annotations: ContentAnnotations = serde_json::from_value(json!({
            "anthropic": {
                "value": {"cache_control": "ephemeral"}
            }
        }))
        .unwrap();

        assert_eq!(
            annotations.decode::<ResponsesCacheAnnotation>().unwrap(),
            None
        );
    }

    #[test]
    fn normalized_duplicate_namespaces_are_rejected() {
        let error = serde_json::from_str::<ContentAnnotations>(
            r#"{
                "OpenAI": {"value": {"cache_control": "first"}},
                "openai": {"unexpected": true}
            }"#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("occurs more than once"));
    }

    #[test]
    fn api_mode_and_runtime_shape_mismatches_are_typed_errors() {
        let annotations = ContentAnnotations::default()
            .with(&ResponsesCacheAnnotation {
                cache_control: "ephemeral".to_string(),
            })
            .unwrap();
        assert!(matches!(
            annotations.decode::<ChatCompletionsCacheAnnotation>(),
            Err(ProviderAnnotationError::ApiModeMismatch { .. })
        ));

        let wire = serde_json::to_value(&annotations).unwrap();
        let message_annotations: MessageAnnotations = serde_json::from_value(wire).unwrap();
        assert!(matches!(
            message_annotations.decode::<MessageAnnotation>(),
            Err(ProviderAnnotationError::InvalidTypedPayload { .. })
        ));
    }

    #[test]
    fn protected_fields_and_oversized_entries_are_rejected() {
        let protected = ContentAnnotations::default().with(&ProtectedAnnotation {
            endpoint: "https://private.example".to_string(),
        });
        assert!(matches!(
            protected,
            Err(ProviderAnnotationError::ProtectedField { .. })
        ));

        let oversized = ContentAnnotations::default().with(&LargeAnnotation {
            payload: "x".repeat(DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT),
        });
        assert!(matches!(
            oversized,
            Err(ProviderAnnotationError::TooLarge { .. })
        ));
    }

    #[test]
    fn encoded_byte_accounting_includes_namespaces_and_is_conservative() {
        let single = ContentAnnotations::default()
            .with(&LargeAnnotation {
                payload: String::new(),
            })
            .unwrap();
        assert_eq!(
            single.encoded_bytes(),
            serde_json::to_vec(&single).unwrap().len()
        );

        let payload_bytes = DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT
            .saturating_sub(single.encoded_bytes())
            .saturating_add(1);
        assert!(matches!(
            ContentAnnotations::default().with(&LargeAnnotation {
                payload: "x".repeat(payload_bytes),
            }),
            Err(ProviderAnnotationError::TooLarge {
                actual,
                maximum: DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT,
            }) if actual == DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT.saturating_add(1)
        ));

        let multiple: ContentAnnotations = serde_json::from_value(json!({
            "anthropic": {"value": {"enabled": true}},
            "openai": {"value": {"enabled": true}}
        }))
        .unwrap();
        assert_eq!(
            multiple.encoded_bytes(),
            serde_json::to_vec(&multiple).unwrap().len() + 1
        );
    }

    #[test]
    fn deserialization_revalidates_limits_and_protected_fields() {
        let protected = serde_json::from_value::<ContentAnnotations>(json!({
            "openai": {"value": {"authorization": "secret"}}
        }));
        assert!(protected.is_err());

        let mut namespaces = Map::new();
        for index in 0..=DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT {
            namespaces.insert(
                format!("provider-{index}"),
                json!({"value": {"enabled": true}}),
            );
        }
        assert!(serde_json::from_value::<ContentAnnotations>(Value::Object(namespaces)).is_err());
    }

    #[test]
    fn deep_oversized_input_is_rejected_before_entry_serialization() {
        let sensitive_marker = "do-not-log-this-annotation-value";
        let mut nested = Value::String(sensitive_marker.repeat(4096));
        for _ in 0..64 {
            nested = Value::Array(vec![nested]);
        }

        let error = serde_json::from_value::<ContentAnnotations>(json!({
            "openai": {"value": {"nested": nested}}
        }))
        .unwrap_err();
        let diagnostic = error.to_string();

        assert!(diagnostic.contains("maximum JSON nesting depth"));
        assert!(!diagnostic.contains(sensitive_marker));
    }

    #[test]
    fn namespace_limit_is_rejected_before_deserializing_the_overflow_value() {
        let valid_entries = (0..DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT)
            .map(|index| format!(r#""provider-{index}":{{"value":{{"enabled":true}}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let wire = format!(r#"{{{valid_entries},"provider-overflow":{{"unexpected":true}}}}"#);

        let error = serde_json::from_str::<ContentAnnotations>(&wire).unwrap_err();

        assert!(error.to_string().contains("per-node maximum"));
    }

    #[test]
    fn debug_output_never_contains_annotation_fields_or_values() {
        let annotations = ContentAnnotations::default()
            .with(&ResponsesCacheAnnotation {
                cache_control: "do-not-log-this-value".to_string(),
            })
            .unwrap();
        let namespace = ProviderId::new("openai").unwrap();

        let collection_debug = format!("{annotations:?}");
        let entry_debug = format!("{:?}", annotations.get(&namespace).unwrap());
        for debug in [&collection_debug, &entry_debug] {
            assert!(!debug.contains("cache_control"));
            assert!(!debug.contains("do-not-log-this-value"));
        }
    }
}
