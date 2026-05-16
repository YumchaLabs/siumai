use crate::types::{
    DataContent, FilePartSource, MediaSource, ProviderOptionsMap, ProviderReference,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::super::ProviderMetadata;

macro_rules! fixed_language_model_v4_type_marker {
    ($name:ident, $value:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(super) enum $name {
            Marker,
        }

        impl Default for $name {
            fn default() -> Self {
                Self::Marker
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str($value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                if value == $value {
                    Ok(Self::Marker)
                } else {
                    Err(serde::de::Error::custom(format!(
                        "expected AI SDK V4 type marker `{}`, got `{value}`",
                        $value
                    )))
                }
            }
        }
    };
}

fixed_language_model_v4_type_marker!(LanguageModelV4FileMarker, "file");
fixed_language_model_v4_type_marker!(LanguageModelV4CustomMarker, "custom");
fixed_language_model_v4_type_marker!(LanguageModelV4TextMarker, "text");
fixed_language_model_v4_type_marker!(LanguageModelV4ReasoningMarker, "reasoning");
fixed_language_model_v4_type_marker!(LanguageModelV4ReasoningFileMarker, "reasoning-file");
fixed_language_model_v4_type_marker!(LanguageModelV4ToolCallMarker, "tool-call");
fixed_language_model_v4_type_marker!(LanguageModelV4ToolResultMarker, "tool-result");
fixed_language_model_v4_type_marker!(
    LanguageModelV4ToolApprovalResponseMarker,
    "tool-approval-response"
);
fixed_language_model_v4_type_marker!(
    LanguageModelV4ToolApprovalRequestMarker,
    "tool-approval-request"
);
fixed_language_model_v4_type_marker!(LanguageModelV4SystemRoleMarker, "system");
fixed_language_model_v4_type_marker!(LanguageModelV4UserRoleMarker, "user");
fixed_language_model_v4_type_marker!(LanguageModelV4AssistantRoleMarker, "assistant");
fixed_language_model_v4_type_marker!(LanguageModelV4ToolRoleMarker, "tool");

/// AI SDK V4 data content.
///
/// JavaScript's `Uint8Array | string | URL` contract is represented as bytes or a string payload.
/// URL values should be passed as their string form.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum LanguageModelV4DataContent {
    /// Base64 data, URL strings, or other provider-accepted string payloads.
    String(String),
    /// Binary data.
    Bytes(Vec<u8>),
}

impl LanguageModelV4DataContent {
    /// Create string-backed data content.
    pub fn string(value: impl Into<String>) -> Self {
        Self::String(value.into())
    }

    /// Create URL-backed data content.
    pub fn url(value: impl Into<String>) -> Self {
        Self::String(value.into())
    }

    /// Create binary data content.
    pub fn bytes(value: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(value.into())
    }
}

impl From<DataContent> for LanguageModelV4DataContent {
    fn from(value: DataContent) -> Self {
        match value {
            DataContent::Base64(value) => Self::String(value),
            DataContent::Binary(value) => Self::Bytes(value),
        }
    }
}

impl From<&DataContent> for LanguageModelV4DataContent {
    fn from(value: &DataContent) -> Self {
        match value {
            DataContent::Base64(value) => Self::String(value.clone()),
            DataContent::Binary(value) => Self::Bytes(value.clone()),
        }
    }
}

impl From<String> for LanguageModelV4DataContent {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for LanguageModelV4DataContent {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<Vec<u8>> for LanguageModelV4DataContent {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for LanguageModelV4DataContent {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

/// AI SDK V4 generated file data.
///
/// Generated `file` and `reasoning-file` content intentionally accepts only
/// base64/string payloads or raw bytes. Prompt data can also carry URL values;
/// generated content cannot in the upstream provider contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum LanguageModelV4GeneratedFileData {
    /// Base64 data or other provider-returned string payloads.
    String(String),
    /// Binary data.
    Bytes(Vec<u8>),
}

impl LanguageModelV4GeneratedFileData {
    /// Create string-backed generated file data.
    pub fn string(value: impl Into<String>) -> Self {
        Self::String(value.into())
    }

    /// Create binary generated file data.
    pub fn bytes(value: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(value.into())
    }
}

impl From<String> for LanguageModelV4GeneratedFileData {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for LanguageModelV4GeneratedFileData {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<Vec<u8>> for LanguageModelV4GeneratedFileData {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

impl From<&[u8]> for LanguageModelV4GeneratedFileData {
    fn from(value: &[u8]) -> Self {
        Self::Bytes(value.to_vec())
    }
}

/// AI SDK V4 file-part data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum LanguageModelV4FilePartData {
    /// Inline data or URL string.
    Data(LanguageModelV4DataContent),
    /// Provider-managed file reference map.
    ProviderReference(ProviderReference),
}

impl From<LanguageModelV4DataContent> for LanguageModelV4FilePartData {
    fn from(value: LanguageModelV4DataContent) -> Self {
        Self::Data(value)
    }
}

impl From<ProviderReference> for LanguageModelV4FilePartData {
    fn from(value: ProviderReference) -> Self {
        Self::ProviderReference(value)
    }
}

impl From<&FilePartSource> for LanguageModelV4FilePartData {
    fn from(value: &FilePartSource) -> Self {
        match value {
            FilePartSource::Media(source) => Self::Data(language_model_v4_data_from_media(source)),
            FilePartSource::ProviderReference { provider_reference } => {
                Self::ProviderReference(provider_reference.clone())
            }
        }
    }
}

impl From<FilePartSource> for LanguageModelV4FilePartData {
    fn from(value: FilePartSource) -> Self {
        Self::from(&value)
    }
}

pub(super) fn language_model_v4_data_from_media(
    source: &MediaSource,
) -> LanguageModelV4DataContent {
    match source {
        MediaSource::Url { url } => LanguageModelV4DataContent::url(url.clone()),
        MediaSource::Base64 { data } => LanguageModelV4DataContent::string(data.clone()),
        MediaSource::Binary { data } => LanguageModelV4DataContent::bytes(data.clone()),
    }
}

pub(super) fn language_model_v4_provider_options_from_stable(
    provider_options: &ProviderOptionsMap,
) -> ProviderOptionsMap {
    let mut projected = ProviderOptionsMap::default();
    for (provider_id, value) in &provider_options.0 {
        if value.is_object() {
            projected.insert(provider_id, value.clone());
        }
    }
    projected
}

fn language_model_v4_provider_options_are_object_shaped(
    provider_options: &ProviderOptionsMap,
) -> bool {
    provider_options
        .0
        .values()
        .all(serde_json::Value::is_object)
}

pub(super) fn serialize_language_model_v4_provider_options_map<S>(
    provider_options: &ProviderOptionsMap,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if !language_model_v4_provider_options_are_object_shaped(provider_options) {
        return Err(serde::ser::Error::custom(
            "expected AI SDK V4 providerOptions values to be JSON objects",
        ));
    }

    provider_options.serialize(serializer)
}

pub(super) fn deserialize_language_model_v4_provider_options_map<'de, D>(
    deserializer: D,
) -> Result<ProviderOptionsMap, D::Error>
where
    D: Deserializer<'de>,
{
    let provider_options = ProviderOptionsMap::deserialize(deserializer)?;
    if !language_model_v4_provider_options_are_object_shaped(&provider_options) {
        return Err(serde::de::Error::custom(
            "expected AI SDK V4 providerOptions values to be JSON objects",
        ));
    }

    Ok(provider_options)
}

pub(super) fn language_model_v4_provider_metadata_from_stable(
    provider_metadata: &Option<ProviderMetadata>,
) -> Option<ProviderMetadata> {
    let projected: ProviderMetadata = provider_metadata
        .as_ref()?
        .iter()
        .filter(|(_, value)| value.is_object())
        .map(|(provider_id, value)| (provider_id.clone(), value.clone()))
        .collect();

    (!projected.is_empty()).then_some(projected)
}

fn language_model_v4_provider_metadata_are_object_shaped(
    provider_metadata: &ProviderMetadata,
) -> bool {
    provider_metadata.values().all(serde_json::Value::is_object)
}

pub(crate) fn serialize_optional_language_model_v4_provider_metadata<S>(
    provider_metadata: &Option<ProviderMetadata>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if provider_metadata.as_ref().is_some_and(|provider_metadata| {
        !language_model_v4_provider_metadata_are_object_shaped(provider_metadata)
    }) {
        return Err(serde::ser::Error::custom(
            "expected AI SDK V4 providerMetadata values to be JSON objects",
        ));
    }

    provider_metadata.serialize(serializer)
}

pub(crate) fn deserialize_optional_language_model_v4_provider_metadata<'de, D>(
    deserializer: D,
) -> Result<Option<ProviderMetadata>, D::Error>
where
    D: Deserializer<'de>,
{
    let provider_metadata = Option::<ProviderMetadata>::deserialize(deserializer)?;
    if provider_metadata.as_ref().is_some_and(|provider_metadata| {
        !language_model_v4_provider_metadata_are_object_shaped(provider_metadata)
    }) {
        return Err(serde::de::Error::custom(
            "expected AI SDK V4 providerMetadata values to be JSON objects",
        ));
    }

    Ok(provider_metadata)
}

pub(super) fn is_language_model_v4_custom_kind(kind: &str) -> bool {
    kind.split_once('.')
        .is_some_and(|(provider, custom_type)| !provider.is_empty() && !custom_type.is_empty())
}

pub(super) fn serialize_language_model_v4_custom_kind<S>(
    kind: &str,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if !is_language_model_v4_custom_kind(kind) {
        return Err(serde::ser::Error::custom(
            "expected AI SDK V4 custom kind in `{provider}.{provider-type}` format",
        ));
    }

    serializer.serialize_str(kind)
}

pub(super) fn deserialize_language_model_v4_custom_kind<'de, D>(
    deserializer: D,
) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let kind = String::deserialize(deserializer)?;
    if !is_language_model_v4_custom_kind(&kind) {
        return Err(serde::de::Error::custom(
            "expected AI SDK V4 custom kind in `{provider}.{provider-type}` format",
        ));
    }

    Ok(kind)
}
