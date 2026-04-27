//! Vercel AI SDK aligned stream parts (typed).
//!
//! Historical `LanguageModelV3*` names are kept for compatibility, but the
//! shapes now track a V4-capable superset of the AI SDK stream-part contract.
//! In particular, this layer now includes V4-only `custom` and
//! `reasoning-file` parts in addition to the older V3-compatible surface.
//!
//! It is intentionally kept as a lightweight, typed representation so users can:
//! - parse provider-specific `ChatStreamEvent::Custom` payloads into a stable schema
//! - implement custom bridging rules between providers without hand-writing JSON maps
//! - format stream parts into SSE `data: ...\n\n` frames when needed
//!
//! English-only comments in code as requested.

use crate::error::LlmError;
use crate::types::{
    ChatStreamCustomContent, ChatStreamEvent, ChatStreamFileData, ChatStreamFilePart,
    ChatStreamFinishInfo, ChatStreamPart, ChatStreamToolApprovalRequest, ChatStreamToolCall,
    ChatStreamToolResult,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Provider metadata object keyed by provider name.
///
/// Vercel AI SDK uses `Record<string, JSONObject>`. We keep this type permissive
/// to preserve metadata for forward compatibility.
pub type SharedV3ProviderMetadata = crate::types::ProviderMetadataMap;

/// Provider-facing V4 metadata root.
///
/// The Rust carrier stays compatible with the shared stable map, but V4 serde/projection helpers
/// enforce AI SDK `SharedV4ProviderMetadata = Record<string, JSONObject>` at provider boundaries.
pub type SharedV4ProviderMetadata = SharedV3ProviderMetadata;

fn serialize_stream_part_non_null_json_value<S>(
    value: &serde_json::Value,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_null() {
        return Err(serde::ser::Error::custom("expected non-null JSON value"));
    }

    value.serialize(serializer)
}

fn deserialize_stream_part_non_null_json_value<'de, D>(
    deserializer: D,
) -> Result<serde_json::Value, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    if value.is_null() {
        return Err(serde::de::Error::custom("expected non-null JSON value"));
    }

    Ok(value)
}

fn stream_v4_provider_metadata_from_stable(
    metadata: Option<crate::types::ProviderMetadataMap>,
) -> Option<SharedV4ProviderMetadata> {
    let projected: SharedV4ProviderMetadata = metadata?
        .into_iter()
        .filter(|(_, value)| value.is_object())
        .collect();

    (!projected.is_empty()).then_some(projected)
}

fn stream_v4_provider_metadata_are_object_shaped(metadata: &SharedV4ProviderMetadata) -> bool {
    metadata.values().all(serde_json::Value::is_object)
}

fn serialize_optional_stream_v4_provider_metadata<S>(
    metadata: &Option<SharedV4ProviderMetadata>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if let Some(metadata) = metadata {
        if !stream_v4_provider_metadata_are_object_shaped(metadata) {
            return Err(serde::ser::Error::custom(
                "expected AI SDK V4 providerMetadata values to be JSON objects",
            ));
        }
    }

    metadata.serialize(serializer)
}

fn deserialize_optional_stream_v4_provider_metadata<'de, D>(
    deserializer: D,
) -> Result<Option<SharedV4ProviderMetadata>, D::Error>
where
    D: Deserializer<'de>,
{
    let metadata = Option::<SharedV4ProviderMetadata>::deserialize(deserializer)?;
    if let Some(metadata) = &metadata {
        if !stream_v4_provider_metadata_are_object_shaped(metadata) {
            return Err(serde::de::Error::custom(
                "expected AI SDK V4 providerMetadata values to be JSON objects",
            ));
        }
    }

    Ok(metadata)
}

fn is_language_model_v4_stream_custom_kind(kind: &str) -> bool {
    kind.split_once('.')
        .is_some_and(|(provider, custom_type)| !provider.is_empty() && !custom_type.is_empty())
}

fn serialize_language_model_v4_stream_custom_kind<S>(
    kind: &str,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if !is_language_model_v4_stream_custom_kind(kind) {
        return Err(serde::ser::Error::custom(
            "expected AI SDK V4 custom kind in `{provider}.{provider-type}` format",
        ));
    }

    serializer.serialize_str(kind)
}

fn deserialize_language_model_v4_stream_custom_kind<'de, D>(
    deserializer: D,
) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let kind = String::deserialize(deserializer)?;
    if !is_language_model_v4_stream_custom_kind(&kind) {
        return Err(serde::de::Error::custom(
            "expected AI SDK V4 custom kind in `{provider}.{provider-type}` format",
        ));
    }

    Ok(kind)
}

/// Shared warning type (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum SharedV3Warning {
    Unsupported {
        feature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    Compatibility {
        feature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    Deprecated {
        setting: String,
        message: String,
    },
    Other {
        message: String,
    },
}

/// V4-capable warning alias kept alongside the historical V3 compatibility name.
pub type SharedV4Warning = SharedV3Warning;

/// Finish reason (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV3FinishReason {
    pub unified: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

/// V4 stream finish-reason payload kept alongside the historical V3 compatibility name.
pub type LanguageModelV4StreamFinishReason = LanguageModelV3FinishReason;

/// Response metadata (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3ResponseMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "modelId")]
    pub model_id: Option<String>,
}

/// V4 stream response-metadata payload kept alongside the historical V3 compatibility name.
pub type LanguageModelV4StreamResponseMetadata = LanguageModelV3ResponseMetadata;

/// Usage (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3Usage {
    #[serde(rename = "inputTokens")]
    pub input_tokens: LanguageModelV3InputTokens,
    #[serde(rename = "outputTokens")]
    pub output_tokens: LanguageModelV3OutputTokens,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Map<String, serde_json::Value>>,
}

/// V4 stream usage payload kept alongside the historical V3 compatibility name.
pub type LanguageModelV4StreamUsage = LanguageModelV3Usage;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3InputTokens {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "noCache")]
    pub no_cache: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "cacheRead")]
    pub cache_read: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "cacheWrite"
    )]
    pub cache_write: Option<u64>,
}

/// V4 stream input-token payload kept alongside the historical V3 compatibility name.
pub type LanguageModelV4StreamInputTokens = LanguageModelV3InputTokens;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3OutputTokens {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
}

/// V4 stream output-token payload kept alongside the historical V3 compatibility name.
pub type LanguageModelV4StreamOutputTokens = LanguageModelV3OutputTokens;

/// Tool call (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3ToolCall {
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    /// Stringified JSON tool arguments (Vercel expects a JSON string).
    pub input: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerExecuted"
    )]
    pub provider_executed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

/// V4 stream tool-call payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4StreamToolCall {
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    /// Stringified JSON tool arguments (Vercel expects a JSON string).
    pub input: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerExecuted"
    )]
    pub provider_executed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata",
        deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
        serialize_with = "serialize_optional_stream_v4_provider_metadata"
    )]
    pub provider_metadata: Option<SharedV4ProviderMetadata>,
}

/// Tool result (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3ToolResult {
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    #[serde(
        deserialize_with = "deserialize_stream_part_non_null_json_value",
        serialize_with = "serialize_stream_part_non_null_json_value"
    )]
    pub result: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "isError")]
    pub is_error: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

/// V4 stream tool-result payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4StreamToolResult {
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    #[serde(
        deserialize_with = "deserialize_stream_part_non_null_json_value",
        serialize_with = "serialize_stream_part_non_null_json_value"
    )]
    pub result: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "isError")]
    pub is_error: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata",
        deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
        serialize_with = "serialize_optional_stream_v4_provider_metadata"
    )]
    pub provider_metadata: Option<SharedV4ProviderMetadata>,
}

/// Tool approval request (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV3ToolApprovalRequest {
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

/// V4 stream tool-approval payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelV4StreamToolApprovalRequest {
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata",
        deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
        serialize_with = "serialize_optional_stream_v4_provider_metadata"
    )]
    pub provider_metadata: Option<SharedV4ProviderMetadata>,
}

/// File part (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3File {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub data: LanguageModelV3FileData,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

/// V4 stream file payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4StreamFile {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub data: LanguageModelV4StreamFileData,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata",
        deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
        serialize_with = "serialize_optional_stream_v4_provider_metadata"
    )]
    pub provider_metadata: Option<SharedV4ProviderMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LanguageModelV3FileData {
    Base64(String),
    Bytes(Vec<u8>),
}

/// V4 stream file-data payload kept alongside the historical V3 compatibility name.
pub type LanguageModelV4StreamFileData = LanguageModelV3FileData;

/// Reasoning file part (Vercel AI SDK V4 aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3ReasoningFile {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub data: LanguageModelV3FileData,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

/// V4 stream reasoning-file payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4StreamReasoningFile {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub data: LanguageModelV4StreamFileData,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata",
        deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
        serialize_with = "serialize_optional_stream_v4_provider_metadata"
    )]
    pub provider_metadata: Option<SharedV4ProviderMetadata>,
}

/// Custom content part (Vercel AI SDK V4 aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV3CustomContent {
    pub kind: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<SharedV3ProviderMetadata>,
}

/// V4 stream custom-content payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4StreamCustomContent {
    #[serde(
        deserialize_with = "deserialize_language_model_v4_stream_custom_kind",
        serialize_with = "serialize_language_model_v4_stream_custom_kind"
    )]
    pub kind: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata",
        deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
        serialize_with = "serialize_optional_stream_v4_provider_metadata"
    )]
    pub provider_metadata: Option<SharedV4ProviderMetadata>,
}

/// Source part (Vercel-aligned).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "sourceType", rename_all = "lowercase")]
pub enum LanguageModelV3Source {
    Url {
        id: String,
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    Document {
        id: String,
        #[serde(rename = "mediaType")]
        media_type: String,
        title: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
}

/// V4 stream source payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "sourceType", rename_all = "lowercase")]
pub enum LanguageModelV4StreamSource {
    Url {
        id: String,
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    Document {
        id: String,
        #[serde(rename = "mediaType")]
        media_type: String,
        title: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
}

/// Typed Vercel stream part union.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum LanguageModelV3StreamPart {
    TextStart {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    TextDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    TextEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },

    ReasoningStart {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    ReasoningDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    ReasoningEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },

    ToolInputStart {
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerExecuted"
        )]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    ToolInputDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },
    ToolInputEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },

    ToolApprovalRequest(LanguageModelV3ToolApprovalRequest),
    ToolCall(LanguageModelV3ToolCall),
    ToolResult(LanguageModelV3ToolResult),
    Custom(LanguageModelV3CustomContent),
    File(LanguageModelV3File),
    ReasoningFile(LanguageModelV3ReasoningFile),
    Source(LanguageModelV3Source),

    StreamStart {
        warnings: Vec<SharedV3Warning>,
    },

    ResponseMetadata(LanguageModelV3ResponseMetadata),

    Finish {
        usage: LanguageModelV3Usage,
        #[serde(rename = "finishReason")]
        finish_reason: LanguageModelV3FinishReason,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata"
        )]
        provider_metadata: Option<SharedV3ProviderMetadata>,
    },

    Raw {
        #[serde(rename = "rawValue")]
        raw_value: serde_json::Value,
    },

    Error {
        error: serde_json::Value,
    },
}

/// Provider-facing AI SDK V4 stream part union.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum LanguageModelV4StreamPart {
    TextStart {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    TextDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    TextEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },

    ReasoningStart {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    ReasoningDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    ReasoningEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },

    ToolInputStart {
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerExecuted"
        )]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    ToolInputDelta {
        id: String,
        delta: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    ToolInputEnd {
        id: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },

    ToolApprovalRequest(LanguageModelV4StreamToolApprovalRequest),
    ToolCall(LanguageModelV4StreamToolCall),
    ToolResult(LanguageModelV4StreamToolResult),
    Custom(LanguageModelV4StreamCustomContent),
    File(LanguageModelV4StreamFile),
    ReasoningFile(LanguageModelV4StreamReasoningFile),
    Source(LanguageModelV4StreamSource),

    StreamStart {
        warnings: Vec<SharedV4Warning>,
    },

    ResponseMetadata(LanguageModelV4StreamResponseMetadata),

    Finish {
        usage: LanguageModelV4StreamUsage,
        #[serde(rename = "finishReason")]
        finish_reason: LanguageModelV4StreamFinishReason,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            deserialize_with = "deserialize_optional_stream_v4_provider_metadata",
            serialize_with = "serialize_optional_stream_v4_provider_metadata"
        )]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },

    Raw {
        #[serde(rename = "rawValue")]
        raw_value: serde_json::Value,
    },

    Error {
        error: serde_json::Value,
    },
}

/// Target namespace used when formatting a stream part into provider-specific custom events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPartNamespace {
    OpenAi,
    Anthropic,
    Gemini,
}

/// Controls how v3 stream parts that cannot be represented in `ChatStreamEvent`
/// should be handled during protocol re-serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum V3UnsupportedPartBehavior {
    /// Drop the part (strictest behavior).
    #[default]
    Drop,
    /// Convert the part into a text delta (lossy, but preserves information).
    AsText,
}

fn stream_metadata_from_hashmap(
    metadata: Option<crate::types::ProviderMetadataMap>,
) -> Option<SharedV3ProviderMetadata> {
    metadata
}

fn stream_metadata_to_hashmap(
    metadata: Option<SharedV3ProviderMetadata>,
) -> Option<crate::types::ProviderMetadataMap> {
    metadata
}

fn finish_reason_to_stream_payload(
    finish_reason: ChatStreamFinishInfo,
) -> LanguageModelV3FinishReason {
    match finish_reason.unified {
        crate::types::FinishReason::Stop => LanguageModelV3FinishReason {
            unified: "stop".to_string(),
            raw: finish_reason.raw,
        },
        crate::types::FinishReason::Length => LanguageModelV3FinishReason {
            unified: "length".to_string(),
            raw: finish_reason.raw,
        },
        crate::types::FinishReason::ToolCalls => LanguageModelV3FinishReason {
            unified: "tool-calls".to_string(),
            raw: finish_reason.raw,
        },
        crate::types::FinishReason::ContentFilter => LanguageModelV3FinishReason {
            unified: "content-filter".to_string(),
            raw: finish_reason.raw,
        },
        crate::types::FinishReason::StopSequence => LanguageModelV3FinishReason {
            unified: "stop".to_string(),
            raw: finish_reason
                .raw
                .or_else(|| Some("stop_sequence".to_string())),
        },
        crate::types::FinishReason::Error => LanguageModelV3FinishReason {
            unified: "error".to_string(),
            raw: finish_reason.raw,
        },
        crate::types::FinishReason::Other(value) => LanguageModelV3FinishReason {
            unified: "other".to_string(),
            raw: finish_reason.raw.or(Some(value)),
        },
        crate::types::FinishReason::Unknown => LanguageModelV3FinishReason {
            unified: "other".to_string(),
            raw: finish_reason.raw,
        },
    }
}

fn finish_reason_from_stream_payload(
    finish_reason: &LanguageModelV3FinishReason,
) -> crate::types::FinishReason {
    match finish_reason.unified.as_str() {
        "stop" if finish_reason.raw.as_deref() == Some("stop_sequence") => {
            crate::types::FinishReason::StopSequence
        }
        "stop" => crate::types::FinishReason::Stop,
        "length" => crate::types::FinishReason::Length,
        "tool-calls" | "tool_calls" => crate::types::FinishReason::ToolCalls,
        "content-filter" | "content_filter" => crate::types::FinishReason::ContentFilter,
        "stop_sequence" => crate::types::FinishReason::StopSequence,
        "error" => crate::types::FinishReason::Error,
        "unknown" => crate::types::FinishReason::Unknown,
        "other" => crate::types::FinishReason::Other(
            finish_reason
                .raw
                .clone()
                .unwrap_or_else(|| "other".to_string()),
        ),
        other => crate::types::FinishReason::Other(other.to_string()),
    }
}

impl From<ChatStreamFileData> for LanguageModelV3FileData {
    fn from(value: ChatStreamFileData) -> Self {
        match value {
            ChatStreamFileData::Base64(data) => Self::Base64(data),
            ChatStreamFileData::Bytes(data) => Self::Bytes(data),
        }
    }
}

impl From<LanguageModelV3FileData> for ChatStreamFileData {
    fn from(value: LanguageModelV3FileData) -> Self {
        match value {
            LanguageModelV3FileData::Base64(data) => Self::Base64(data),
            LanguageModelV3FileData::Bytes(data) => Self::Bytes(data),
        }
    }
}

impl From<ChatStreamToolApprovalRequest> for LanguageModelV3ToolApprovalRequest {
    fn from(value: ChatStreamToolApprovalRequest) -> Self {
        Self {
            approval_id: value.approval_id,
            tool_call_id: value.tool_call_id,
            provider_metadata: stream_metadata_from_hashmap(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV3ToolApprovalRequest> for ChatStreamToolApprovalRequest {
    fn from(value: LanguageModelV3ToolApprovalRequest) -> Self {
        Self {
            approval_id: value.approval_id,
            tool_call_id: value.tool_call_id,
            provider_metadata: stream_metadata_to_hashmap(value.provider_metadata),
        }
    }
}

impl From<ChatStreamToolCall> for LanguageModelV3ToolCall {
    fn from(value: ChatStreamToolCall) -> Self {
        Self {
            tool_call_id: value.tool_call_id,
            tool_name: value.tool_name,
            input: value.input,
            provider_executed: value.provider_executed,
            dynamic: value.dynamic,
            provider_metadata: stream_metadata_from_hashmap(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV3ToolCall> for ChatStreamToolCall {
    fn from(value: LanguageModelV3ToolCall) -> Self {
        Self {
            tool_call_id: value.tool_call_id,
            tool_name: value.tool_name,
            input: value.input,
            provider_executed: value.provider_executed,
            dynamic: value.dynamic,
            provider_metadata: stream_metadata_to_hashmap(value.provider_metadata),
        }
    }
}

impl From<ChatStreamToolResult> for LanguageModelV3ToolResult {
    fn from(value: ChatStreamToolResult) -> Self {
        Self {
            tool_call_id: value.tool_call_id,
            tool_name: value.tool_name,
            result: value.result,
            is_error: value.is_error,
            preliminary: value.preliminary,
            dynamic: value.dynamic,
            provider_metadata: stream_metadata_from_hashmap(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV3ToolResult> for ChatStreamToolResult {
    fn from(value: LanguageModelV3ToolResult) -> Self {
        Self {
            tool_call_id: value.tool_call_id,
            tool_name: value.tool_name,
            result: value.result,
            is_error: value.is_error,
            preliminary: value.preliminary,
            dynamic: value.dynamic,
            provider_metadata: stream_metadata_to_hashmap(value.provider_metadata),
        }
    }
}

impl From<ChatStreamCustomContent> for LanguageModelV3CustomContent {
    fn from(value: ChatStreamCustomContent) -> Self {
        Self {
            kind: value.kind,
            provider_metadata: stream_metadata_from_hashmap(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV3CustomContent> for ChatStreamCustomContent {
    fn from(value: LanguageModelV3CustomContent) -> Self {
        Self {
            kind: value.kind,
            provider_metadata: stream_metadata_to_hashmap(value.provider_metadata),
        }
    }
}

impl From<ChatStreamFilePart> for LanguageModelV3File {
    fn from(value: ChatStreamFilePart) -> Self {
        Self {
            media_type: value.media_type,
            data: value.data.into(),
            provider_metadata: stream_metadata_from_hashmap(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV3File> for ChatStreamFilePart {
    fn from(value: LanguageModelV3File) -> Self {
        Self {
            media_type: value.media_type,
            data: value.data.into(),
            provider_metadata: stream_metadata_to_hashmap(value.provider_metadata),
        }
    }
}

impl From<ChatStreamFilePart> for LanguageModelV3ReasoningFile {
    fn from(value: ChatStreamFilePart) -> Self {
        Self {
            media_type: value.media_type,
            data: value.data.into(),
            provider_metadata: stream_metadata_from_hashmap(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV3ReasoningFile> for ChatStreamFilePart {
    fn from(value: LanguageModelV3ReasoningFile) -> Self {
        Self {
            media_type: value.media_type,
            data: value.data.into(),
            provider_metadata: stream_metadata_to_hashmap(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV3ToolApprovalRequest> for LanguageModelV4StreamToolApprovalRequest {
    fn from(value: LanguageModelV3ToolApprovalRequest) -> Self {
        Self {
            approval_id: value.approval_id,
            tool_call_id: value.tool_call_id,
            provider_metadata: stream_v4_provider_metadata_from_stable(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV4StreamToolApprovalRequest> for LanguageModelV3ToolApprovalRequest {
    fn from(value: LanguageModelV4StreamToolApprovalRequest) -> Self {
        Self {
            approval_id: value.approval_id,
            tool_call_id: value.tool_call_id,
            provider_metadata: value.provider_metadata,
        }
    }
}

impl From<LanguageModelV3ToolCall> for LanguageModelV4StreamToolCall {
    fn from(value: LanguageModelV3ToolCall) -> Self {
        Self {
            tool_call_id: value.tool_call_id,
            tool_name: value.tool_name,
            input: value.input,
            provider_executed: value.provider_executed,
            dynamic: value.dynamic,
            provider_metadata: stream_v4_provider_metadata_from_stable(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV4StreamToolCall> for LanguageModelV3ToolCall {
    fn from(value: LanguageModelV4StreamToolCall) -> Self {
        Self {
            tool_call_id: value.tool_call_id,
            tool_name: value.tool_name,
            input: value.input,
            provider_executed: value.provider_executed,
            dynamic: value.dynamic,
            provider_metadata: value.provider_metadata,
        }
    }
}

impl From<LanguageModelV3ToolResult> for LanguageModelV4StreamToolResult {
    fn from(value: LanguageModelV3ToolResult) -> Self {
        Self {
            tool_call_id: value.tool_call_id,
            tool_name: value.tool_name,
            result: value.result,
            is_error: value.is_error,
            preliminary: value.preliminary,
            dynamic: value.dynamic,
            provider_metadata: stream_v4_provider_metadata_from_stable(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV4StreamToolResult> for LanguageModelV3ToolResult {
    fn from(value: LanguageModelV4StreamToolResult) -> Self {
        Self {
            tool_call_id: value.tool_call_id,
            tool_name: value.tool_name,
            result: value.result,
            is_error: value.is_error,
            preliminary: value.preliminary,
            dynamic: value.dynamic,
            provider_metadata: value.provider_metadata,
        }
    }
}

impl From<LanguageModelV3CustomContent> for LanguageModelV4StreamCustomContent {
    fn from(value: LanguageModelV3CustomContent) -> Self {
        Self {
            kind: value.kind,
            provider_metadata: stream_v4_provider_metadata_from_stable(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV4StreamCustomContent> for LanguageModelV3CustomContent {
    fn from(value: LanguageModelV4StreamCustomContent) -> Self {
        Self {
            kind: value.kind,
            provider_metadata: value.provider_metadata,
        }
    }
}

impl From<LanguageModelV3File> for LanguageModelV4StreamFile {
    fn from(value: LanguageModelV3File) -> Self {
        Self {
            media_type: value.media_type,
            data: value.data,
            provider_metadata: stream_v4_provider_metadata_from_stable(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV4StreamFile> for LanguageModelV3File {
    fn from(value: LanguageModelV4StreamFile) -> Self {
        Self {
            media_type: value.media_type,
            data: value.data,
            provider_metadata: value.provider_metadata,
        }
    }
}

impl From<LanguageModelV3ReasoningFile> for LanguageModelV4StreamReasoningFile {
    fn from(value: LanguageModelV3ReasoningFile) -> Self {
        Self {
            media_type: value.media_type,
            data: value.data,
            provider_metadata: stream_v4_provider_metadata_from_stable(value.provider_metadata),
        }
    }
}

impl From<LanguageModelV4StreamReasoningFile> for LanguageModelV3ReasoningFile {
    fn from(value: LanguageModelV4StreamReasoningFile) -> Self {
        Self {
            media_type: value.media_type,
            data: value.data,
            provider_metadata: value.provider_metadata,
        }
    }
}

impl From<LanguageModelV3Source> for LanguageModelV4StreamSource {
    fn from(value: LanguageModelV3Source) -> Self {
        match value {
            LanguageModelV3Source::Url {
                id,
                url,
                title,
                provider_metadata,
            } => Self::Url {
                id,
                url,
                title,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3Source::Document {
                id,
                media_type,
                title,
                filename,
                provider_metadata,
            } => Self::Document {
                id,
                media_type,
                title,
                filename,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
        }
    }
}

impl From<LanguageModelV4StreamSource> for LanguageModelV3Source {
    fn from(value: LanguageModelV4StreamSource) -> Self {
        match value {
            LanguageModelV4StreamSource::Url {
                id,
                url,
                title,
                provider_metadata,
            } => Self::Url {
                id,
                url,
                title,
                provider_metadata,
            },
            LanguageModelV4StreamSource::Document {
                id,
                media_type,
                title,
                filename,
                provider_metadata,
            } => Self::Document {
                id,
                media_type,
                title,
                filename,
                provider_metadata,
            },
        }
    }
}

impl From<LanguageModelV3StreamPart> for LanguageModelV4StreamPart {
    fn from(value: LanguageModelV3StreamPart) -> Self {
        match value {
            LanguageModelV3StreamPart::TextStart {
                id,
                provider_metadata,
            } => Self::TextStart {
                id,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3StreamPart::TextDelta {
                id,
                delta,
                provider_metadata,
            } => Self::TextDelta {
                id,
                delta,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3StreamPart::TextEnd {
                id,
                provider_metadata,
            } => Self::TextEnd {
                id,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3StreamPart::ReasoningStart {
                id,
                provider_metadata,
            } => Self::ReasoningStart {
                id,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3StreamPart::ReasoningDelta {
                id,
                delta,
                provider_metadata,
            } => Self::ReasoningDelta {
                id,
                delta,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3StreamPart::ReasoningEnd {
                id,
                provider_metadata,
            } => Self::ReasoningEnd {
                id,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3StreamPart::ToolInputStart {
                id,
                tool_name,
                provider_metadata,
                provider_executed,
                dynamic,
                title,
            } => Self::ToolInputStart {
                id,
                tool_name,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
                provider_executed,
                dynamic,
                title,
            },
            LanguageModelV3StreamPart::ToolInputDelta {
                id,
                delta,
                provider_metadata,
            } => Self::ToolInputDelta {
                id,
                delta,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3StreamPart::ToolInputEnd {
                id,
                provider_metadata,
            } => Self::ToolInputEnd {
                id,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3StreamPart::ToolApprovalRequest(request) => {
                Self::ToolApprovalRequest(request.into())
            }
            LanguageModelV3StreamPart::ToolCall(call) => Self::ToolCall(call.into()),
            LanguageModelV3StreamPart::ToolResult(result) => Self::ToolResult(result.into()),
            LanguageModelV3StreamPart::Custom(custom) => Self::Custom(custom.into()),
            LanguageModelV3StreamPart::File(file) => Self::File(file.into()),
            LanguageModelV3StreamPart::ReasoningFile(file) => Self::ReasoningFile(file.into()),
            LanguageModelV3StreamPart::Source(source) => Self::Source(source.into()),
            LanguageModelV3StreamPart::StreamStart { warnings } => Self::StreamStart { warnings },
            LanguageModelV3StreamPart::ResponseMetadata(metadata) => {
                Self::ResponseMetadata(metadata)
            }
            LanguageModelV3StreamPart::Finish {
                usage,
                finish_reason,
                provider_metadata,
            } => Self::Finish {
                usage,
                finish_reason,
                provider_metadata: stream_v4_provider_metadata_from_stable(provider_metadata),
            },
            LanguageModelV3StreamPart::Raw { raw_value } => Self::Raw { raw_value },
            LanguageModelV3StreamPart::Error { error } => Self::Error { error },
        }
    }
}

impl From<LanguageModelV4StreamPart> for LanguageModelV3StreamPart {
    fn from(value: LanguageModelV4StreamPart) -> Self {
        match value {
            LanguageModelV4StreamPart::TextStart {
                id,
                provider_metadata,
            } => Self::TextStart {
                id,
                provider_metadata,
            },
            LanguageModelV4StreamPart::TextDelta {
                id,
                delta,
                provider_metadata,
            } => Self::TextDelta {
                id,
                delta,
                provider_metadata,
            },
            LanguageModelV4StreamPart::TextEnd {
                id,
                provider_metadata,
            } => Self::TextEnd {
                id,
                provider_metadata,
            },
            LanguageModelV4StreamPart::ReasoningStart {
                id,
                provider_metadata,
            } => Self::ReasoningStart {
                id,
                provider_metadata,
            },
            LanguageModelV4StreamPart::ReasoningDelta {
                id,
                delta,
                provider_metadata,
            } => Self::ReasoningDelta {
                id,
                delta,
                provider_metadata,
            },
            LanguageModelV4StreamPart::ReasoningEnd {
                id,
                provider_metadata,
            } => Self::ReasoningEnd {
                id,
                provider_metadata,
            },
            LanguageModelV4StreamPart::ToolInputStart {
                id,
                tool_name,
                provider_metadata,
                provider_executed,
                dynamic,
                title,
            } => Self::ToolInputStart {
                id,
                tool_name,
                provider_metadata,
                provider_executed,
                dynamic,
                title,
            },
            LanguageModelV4StreamPart::ToolInputDelta {
                id,
                delta,
                provider_metadata,
            } => Self::ToolInputDelta {
                id,
                delta,
                provider_metadata,
            },
            LanguageModelV4StreamPart::ToolInputEnd {
                id,
                provider_metadata,
            } => Self::ToolInputEnd {
                id,
                provider_metadata,
            },
            LanguageModelV4StreamPart::ToolApprovalRequest(request) => {
                Self::ToolApprovalRequest(request.into())
            }
            LanguageModelV4StreamPart::ToolCall(call) => Self::ToolCall(call.into()),
            LanguageModelV4StreamPart::ToolResult(result) => Self::ToolResult(result.into()),
            LanguageModelV4StreamPart::Custom(custom) => Self::Custom(custom.into()),
            LanguageModelV4StreamPart::File(file) => Self::File(file.into()),
            LanguageModelV4StreamPart::ReasoningFile(file) => Self::ReasoningFile(file.into()),
            LanguageModelV4StreamPart::Source(source) => Self::Source(source.into()),
            LanguageModelV4StreamPart::StreamStart { warnings } => Self::StreamStart { warnings },
            LanguageModelV4StreamPart::ResponseMetadata(metadata) => {
                Self::ResponseMetadata(metadata)
            }
            LanguageModelV4StreamPart::Finish {
                usage,
                finish_reason,
                provider_metadata,
            } => Self::Finish {
                usage,
                finish_reason,
                provider_metadata,
            },
            LanguageModelV4StreamPart::Raw { raw_value } => Self::Raw { raw_value },
            LanguageModelV4StreamPart::Error { error } => Self::Error { error },
        }
    }
}

impl LanguageModelV3StreamPart {
    /// Convert a stable runtime semantic part into the typed V4-capable overlay.
    pub fn from_runtime_part(part: ChatStreamPart) -> Self {
        match part {
            ChatStreamPart::TextStart {
                id,
                provider_metadata,
            } => Self::TextStart {
                id,
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
            },
            ChatStreamPart::TextDelta {
                id,
                delta,
                provider_metadata,
            } => Self::TextDelta {
                id,
                delta,
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
            },
            ChatStreamPart::TextEnd {
                id,
                provider_metadata,
            } => Self::TextEnd {
                id,
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
            },
            ChatStreamPart::ReasoningStart {
                id,
                provider_metadata,
            } => Self::ReasoningStart {
                id,
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
            },
            ChatStreamPart::ReasoningDelta {
                id,
                delta,
                provider_metadata,
            } => Self::ReasoningDelta {
                id,
                delta,
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
            },
            ChatStreamPart::ReasoningEnd {
                id,
                provider_metadata,
            } => Self::ReasoningEnd {
                id,
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
            },
            ChatStreamPart::ToolInputStart {
                id,
                tool_name,
                provider_metadata,
                provider_executed,
                dynamic,
                title,
            } => Self::ToolInputStart {
                id,
                tool_name,
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
                provider_executed,
                dynamic,
                title,
            },
            ChatStreamPart::ToolInputDelta {
                id,
                delta,
                provider_metadata,
            } => Self::ToolInputDelta {
                id,
                delta,
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
            },
            ChatStreamPart::ToolInputEnd {
                id,
                provider_metadata,
            } => Self::ToolInputEnd {
                id,
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
            },
            ChatStreamPart::ToolApprovalRequest(request) => {
                Self::ToolApprovalRequest(request.into())
            }
            ChatStreamPart::ToolCall(call) => Self::ToolCall(call.into()),
            ChatStreamPart::ToolResult(result) => Self::ToolResult(result.into()),
            ChatStreamPart::Custom(custom) => Self::Custom(custom.into()),
            ChatStreamPart::File(file) => Self::File(file.into()),
            ChatStreamPart::ReasoningFile(file) => Self::ReasoningFile(file.into()),
            ChatStreamPart::Source {
                id,
                source,
                provider_metadata,
            } => Self::Source(match source {
                crate::types::SourcePart::Url { url, title } => LanguageModelV3Source::Url {
                    id,
                    url,
                    title,
                    provider_metadata: stream_metadata_from_hashmap(provider_metadata),
                },
                crate::types::SourcePart::Document {
                    media_type,
                    title,
                    filename,
                } => LanguageModelV3Source::Document {
                    id,
                    media_type,
                    title,
                    filename,
                    provider_metadata: stream_metadata_from_hashmap(provider_metadata),
                },
            }),
            ChatStreamPart::StreamStart { warnings } => Self::StreamStart {
                warnings: warnings
                    .into_iter()
                    .map(|warning| match warning {
                        crate::types::Warning::Unsupported { feature, details } => {
                            SharedV3Warning::Unsupported { feature, details }
                        }
                        crate::types::Warning::UnsupportedSetting { setting, details } => {
                            SharedV3Warning::Unsupported {
                                feature: setting,
                                details,
                            }
                        }
                        crate::types::Warning::UnsupportedTool { tool_name, details } => {
                            SharedV3Warning::Unsupported {
                                feature: tool_name,
                                details,
                            }
                        }
                        crate::types::Warning::Compatibility { feature, details } => {
                            SharedV3Warning::Compatibility { feature, details }
                        }
                        crate::types::Warning::Deprecated { setting, message } => {
                            SharedV3Warning::Deprecated { setting, message }
                        }
                        crate::types::Warning::Other { message } => {
                            SharedV3Warning::Other { message }
                        }
                    })
                    .collect(),
            },
            ChatStreamPart::ResponseMetadata(metadata) => {
                Self::ResponseMetadata(LanguageModelV3ResponseMetadata {
                    id: metadata.id,
                    timestamp: metadata.created,
                    model_id: metadata.model,
                })
            }
            ChatStreamPart::Finish {
                usage,
                finish_reason,
                provider_metadata,
            } => Self::Finish {
                usage: {
                    let input_tokens = usage.normalized_input_tokens();
                    let output_tokens = usage.normalized_output_tokens();

                    LanguageModelV3Usage {
                        input_tokens: LanguageModelV3InputTokens {
                            total: input_tokens.total.map(u64::from),
                            no_cache: input_tokens.no_cache.map(u64::from),
                            cache_read: input_tokens.cache_read.map(u64::from),
                            cache_write: input_tokens.cache_write.map(u64::from),
                        },
                        output_tokens: LanguageModelV3OutputTokens {
                            total: output_tokens.total.map(u64::from),
                            text: output_tokens.text.map(u64::from),
                            reasoning: output_tokens.reasoning.map(u64::from),
                        },
                        raw: usage.raw.clone(),
                    }
                },
                finish_reason: finish_reason_to_stream_payload(finish_reason),
                provider_metadata: stream_metadata_from_hashmap(provider_metadata),
            },
            ChatStreamPart::Raw { raw_value } => Self::Raw { raw_value },
            ChatStreamPart::Error { error } => Self::Error { error },
        }
    }

    /// Convert this typed stream part into the stable runtime semantic part.
    pub fn to_runtime_part(&self) -> ChatStreamPart {
        match self {
            Self::TextStart {
                id,
                provider_metadata,
            } => ChatStreamPart::TextStart {
                id: id.clone(),
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
            },
            Self::TextDelta {
                id,
                delta,
                provider_metadata,
            } => ChatStreamPart::TextDelta {
                id: id.clone(),
                delta: delta.clone(),
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
            },
            Self::TextEnd {
                id,
                provider_metadata,
            } => ChatStreamPart::TextEnd {
                id: id.clone(),
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
            },
            Self::ReasoningStart {
                id,
                provider_metadata,
            } => ChatStreamPart::ReasoningStart {
                id: id.clone(),
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
            },
            Self::ReasoningDelta {
                id,
                delta,
                provider_metadata,
            } => ChatStreamPart::ReasoningDelta {
                id: id.clone(),
                delta: delta.clone(),
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
            },
            Self::ReasoningEnd {
                id,
                provider_metadata,
            } => ChatStreamPart::ReasoningEnd {
                id: id.clone(),
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
            },
            Self::ToolInputStart {
                id,
                tool_name,
                provider_metadata,
                provider_executed,
                dynamic,
                title,
            } => ChatStreamPart::ToolInputStart {
                id: id.clone(),
                tool_name: tool_name.clone(),
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
                provider_executed: *provider_executed,
                dynamic: *dynamic,
                title: title.clone(),
            },
            Self::ToolInputDelta {
                id,
                delta,
                provider_metadata,
            } => ChatStreamPart::ToolInputDelta {
                id: id.clone(),
                delta: delta.clone(),
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
            },
            Self::ToolInputEnd {
                id,
                provider_metadata,
            } => ChatStreamPart::ToolInputEnd {
                id: id.clone(),
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
            },
            Self::ToolApprovalRequest(request) => {
                ChatStreamPart::ToolApprovalRequest(request.clone().into())
            }
            Self::ToolCall(call) => ChatStreamPart::ToolCall(call.clone().into()),
            Self::ToolResult(result) => ChatStreamPart::ToolResult(result.clone().into()),
            Self::Custom(custom) => ChatStreamPart::Custom(custom.clone().into()),
            Self::File(file) => ChatStreamPart::File(file.clone().into()),
            Self::ReasoningFile(file) => ChatStreamPart::ReasoningFile(file.clone().into()),
            Self::Source(source) => match source.clone() {
                LanguageModelV3Source::Url {
                    id,
                    url,
                    title,
                    provider_metadata,
                } => ChatStreamPart::Source {
                    id,
                    source: crate::types::SourcePart::Url { url, title },
                    provider_metadata: stream_metadata_to_hashmap(provider_metadata),
                },
                LanguageModelV3Source::Document {
                    id,
                    media_type,
                    title,
                    filename,
                    provider_metadata,
                } => ChatStreamPart::Source {
                    id,
                    source: crate::types::SourcePart::Document {
                        media_type,
                        title,
                        filename,
                    },
                    provider_metadata: stream_metadata_to_hashmap(provider_metadata),
                },
            },
            Self::StreamStart { warnings } => ChatStreamPart::StreamStart {
                warnings: warnings
                    .iter()
                    .cloned()
                    .map(|warning| match warning {
                        SharedV3Warning::Unsupported { feature, details } => {
                            crate::types::Warning::Unsupported { feature, details }
                        }
                        SharedV3Warning::Compatibility { feature, details } => {
                            crate::types::Warning::Compatibility { feature, details }
                        }
                        SharedV3Warning::Deprecated { setting, message } => {
                            crate::types::Warning::Deprecated { setting, message }
                        }
                        SharedV3Warning::Other { message } => {
                            crate::types::Warning::Other { message }
                        }
                    })
                    .collect(),
            },
            Self::ResponseMetadata(metadata) => {
                ChatStreamPart::ResponseMetadata(crate::types::ResponseMetadata {
                    id: metadata.id.clone(),
                    model: metadata.model_id.clone(),
                    created: metadata.timestamp,
                    provider: String::new(),
                    request_id: None,
                    headers: None,
                })
            }
            Self::Finish {
                usage,
                finish_reason,
                provider_metadata,
            } => ChatStreamPart::Finish {
                usage: {
                    let mut builder = crate::types::Usage::builder().with_input_tokens(
                        crate::types::UsageInputTokens {
                            total: usage.input_tokens.total.map(|v| v as u32),
                            no_cache: usage.input_tokens.no_cache.map(|v| v as u32),
                            cache_read: usage.input_tokens.cache_read.map(|v| v as u32),
                            cache_write: usage.input_tokens.cache_write.map(|v| v as u32),
                        },
                    );
                    builder = builder.with_output_tokens(crate::types::UsageOutputTokens {
                        total: usage.output_tokens.total.map(|v| v as u32),
                        text: usage.output_tokens.text.map(|v| v as u32),
                        reasoning: usage.output_tokens.reasoning.map(|v| v as u32),
                    });
                    if let Some(cached_tokens) = usage.input_tokens.cache_read.map(|v| v as u32) {
                        builder = builder.with_cached_tokens(cached_tokens);
                    }
                    if let Some(reasoning_tokens) = usage.output_tokens.reasoning.map(|v| v as u32)
                    {
                        builder = builder.with_reasoning_tokens(reasoning_tokens);
                    }
                    if let Some(raw) = usage.raw.clone() {
                        builder = builder.with_raw_usage(raw);
                    }
                    builder.build()
                },
                finish_reason: ChatStreamFinishInfo {
                    unified: finish_reason_from_stream_payload(finish_reason),
                    raw: finish_reason.raw.clone(),
                },
                provider_metadata: stream_metadata_to_hashmap(provider_metadata.clone()),
            },
            Self::Raw { raw_value } => ChatStreamPart::Raw {
                raw_value: raw_value.clone(),
            },
            Self::Error { error } => ChatStreamPart::Error {
                error: error.clone(),
            },
        }
    }

    /// Best-effort parse a v3 stream part from a JSON payload that is close to the
    /// Vercel schema but may contain minor shape differences.
    ///
    /// This is mainly used for cross-provider stream transcoding, where some
    /// providers emit `input` as a JSON object instead of a stringified JSON.
    pub fn parse_loose_json(value: &serde_json::Value) -> Option<Self> {
        let mut v = value.clone();

        fn normalize_in_place(v: &mut serde_json::Value) {
            let Some(obj) = v.as_object_mut() else {
                return;
            };
            let Some(tpe) = obj.get("type").and_then(|v| v.as_str()) else {
                return;
            };

            match tpe {
                "tool-call" => {
                    if let Some(input) = obj.get_mut("input")
                        && !matches!(input, serde_json::Value::String(_))
                        && let Ok(s) = serde_json::to_string(input)
                    {
                        *input = serde_json::Value::String(s);
                    }
                }
                "finish" => {
                    if let Some(fr) = obj.get_mut("finishReason")
                        && let serde_json::Value::String(s) = fr
                    {
                        *fr = serde_json::json!({ "unified": s, "raw": serde_json::Value::Null });
                    }
                }
                "source" => {
                    if obj.get("sourceType").is_none()
                        && let Some(st) = obj.remove("source_type")
                    {
                        obj.insert("sourceType".to_string(), st);
                    }
                }
                _ => {}
            }
        }

        normalize_in_place(&mut v);
        serde_json::from_value::<LanguageModelV3StreamPart>(v).ok()
    }

    /// Best-effort parse from a `ChatStreamEvent`.
    ///
    /// This is primarily intended for `ChatStreamEvent::Custom` where `data` follows
    /// the Vercel stream part JSON shape.
    pub fn try_from_chat_event(ev: &ChatStreamEvent) -> Option<Self> {
        match ev {
            ChatStreamEvent::Part { part } => Some(Self::from_runtime_part(part.clone())),
            ChatStreamEvent::PartWithReplay { part, .. } => {
                Some(Self::from_runtime_part(part.clone()))
            }
            ChatStreamEvent::Custom { data, .. } => Self::parse_loose_json(data),
            ChatStreamEvent::Error { error } => Some(LanguageModelV3StreamPart::Error {
                error: serde_json::json!(error),
            }),
            _ => None,
        }
    }

    /// Format as a provider-specific protocol-side `ChatStreamEvent::Custom` (best-effort).
    ///
    /// This is serializer compatibility glue for provider wire formats that still
    /// encode some typed parts as `Custom` events. Stable runtime consumers should
    /// prefer `to_part_event()` / `to_runtime_part()` instead.
    pub fn to_protocol_custom_event(&self, ns: StreamPartNamespace) -> Option<ChatStreamEvent> {
        let (event_type, data) = match ns {
            StreamPartNamespace::OpenAi => self.to_openai_custom_event_payload()?,
            StreamPartNamespace::Anthropic => self.to_anthropic_custom_event_payload()?,
            StreamPartNamespace::Gemini => self.to_gemini_custom_event_payload()?,
        };

        Some(ChatStreamEvent::Custom { event_type, data })
    }

    /// Legacy compatibility alias for provider protocol serializers.
    #[inline]
    pub fn to_custom_event(&self, ns: StreamPartNamespace) -> Option<ChatStreamEvent> {
        self.to_protocol_custom_event(ns)
    }

    /// Convert this typed stream part into a first-class runtime stream part event.
    pub fn to_part_event(&self) -> ChatStreamEvent {
        ChatStreamEvent::Part {
            part: self.to_runtime_part(),
        }
    }

    /// Encode this stream part as an SSE `data: ...\n\n` frame.
    pub fn to_data_sse_bytes(&self) -> Result<Vec<u8>, LlmError> {
        let json = serde_json::to_string(self).map_err(|e| {
            LlmError::ParseError(format!("Failed to serialize stream part JSON: {e}"))
        })?;
        Ok(format!("data: {json}\n\n").into_bytes())
    }

    /// Convert this typed stream part into best-effort `ChatStreamEvent`s.
    ///
    /// This is primarily intended for protocol re-serialization, where a target
    /// provider's encoder might not understand the source provider's custom events.
    ///
    /// Notes:
    /// - Not all Vercel stream parts have a lossless representation in `ChatStreamEvent`.
    /// - Unsupported parts are dropped (empty vec).
    pub fn to_best_effort_chat_events(&self) -> Vec<ChatStreamEvent> {
        match self {
            LanguageModelV3StreamPart::TextDelta { delta, .. } => {
                vec![ChatStreamEvent::ContentDelta {
                    delta: delta.clone(),
                    index: None,
                }]
            }
            LanguageModelV3StreamPart::ReasoningDelta { delta, .. } => {
                vec![ChatStreamEvent::ThinkingDelta {
                    delta: delta.clone(),
                }]
            }
            LanguageModelV3StreamPart::ToolInputStart { id, tool_name, .. } => {
                vec![ChatStreamEvent::ToolCallDelta {
                    id: id.clone(),
                    function_name: Some(tool_name.clone()),
                    arguments_delta: None,
                    index: None,
                }]
            }
            LanguageModelV3StreamPart::ToolInputDelta { id, delta, .. } => {
                vec![ChatStreamEvent::ToolCallDelta {
                    id: id.clone(),
                    function_name: None,
                    arguments_delta: Some(delta.clone()),
                    index: None,
                }]
            }
            LanguageModelV3StreamPart::ToolCall(call) => vec![ChatStreamEvent::ToolCallDelta {
                id: call.tool_call_id.clone(),
                function_name: Some(call.tool_name.clone()),
                arguments_delta: Some(call.input.clone()),
                index: None,
            }],
            _ => vec![self.to_part_event()],
        }
    }

    /// Convert a v3 stream part into a lossy text representation.
    pub fn to_lossy_text(&self) -> Option<String> {
        match self {
            LanguageModelV3StreamPart::Source(src) => match src {
                LanguageModelV3Source::Url { url, title, .. } => Some(format!(
                    "[source] {} {}",
                    title.clone().unwrap_or_else(|| "url".to_string()),
                    url
                )),
                LanguageModelV3Source::Document {
                    title,
                    filename,
                    media_type,
                    ..
                } => Some(format!(
                    "[source] {} ({}){}",
                    title,
                    media_type,
                    filename
                        .as_ref()
                        .map(|f| format!(", {f}"))
                        .unwrap_or_default()
                )),
            },
            LanguageModelV3StreamPart::ToolResult(tr) => Some(format!(
                "[tool-result] {}: {}",
                tr.tool_name,
                serde_json::to_string(&tr.result).unwrap_or_else(|_| "{}".to_string())
            )),
            LanguageModelV3StreamPart::ToolApprovalRequest(req) => Some(format!(
                "[tool-approval-request] approvalId={} toolCallId={}",
                req.approval_id, req.tool_call_id
            )),
            LanguageModelV3StreamPart::Finish { finish_reason, .. } => {
                Some(format!("[finish] {}", finish_reason.unified))
            }
            LanguageModelV3StreamPart::Error { error } => Some(format!(
                "[error] {}",
                serde_json::to_string(error).unwrap_or_else(|_| "\"error\"".to_string())
            )),
            LanguageModelV3StreamPart::Raw { raw_value } => Some(format!(
                "[raw] {}",
                serde_json::to_string(raw_value).unwrap_or_else(|_| "\"raw\"".to_string())
            )),
            LanguageModelV3StreamPart::Custom(custom) => Some(format!("[custom] {}", custom.kind)),
            LanguageModelV3StreamPart::File(file) => {
                let len_hint = match &file.data {
                    LanguageModelV3FileData::Base64(s) => format!("base64_len={}", s.len()),
                    LanguageModelV3FileData::Bytes(b) => format!("bytes_len={}", b.len()),
                };

                Some(format!("[file] mediaType={} {}", file.media_type, len_hint))
            }
            LanguageModelV3StreamPart::ReasoningFile(file) => {
                let len_hint = match &file.data {
                    LanguageModelV3FileData::Base64(s) => format!("base64_len={}", s.len()),
                    LanguageModelV3FileData::Bytes(b) => format!("bytes_len={}", b.len()),
                };

                Some(format!(
                    "[reasoning-file] mediaType={} {}",
                    file.media_type, len_hint
                ))
            }
            _ => None,
        }
    }

    fn to_openai_custom_event_payload(&self) -> Option<(String, serde_json::Value)> {
        let data = serde_json::to_value(self).ok()?;
        let event_type = match self {
            LanguageModelV3StreamPart::TextStart { .. } => "openai:text-start",
            LanguageModelV3StreamPart::TextDelta { .. } => "openai:text-delta",
            LanguageModelV3StreamPart::TextEnd { .. } => "openai:text-end",
            LanguageModelV3StreamPart::ReasoningStart { .. } => "openai:reasoning-start",
            LanguageModelV3StreamPart::ReasoningDelta { .. } => "openai:reasoning-delta",
            LanguageModelV3StreamPart::ReasoningEnd { .. } => "openai:reasoning-end",
            LanguageModelV3StreamPart::ToolInputStart { .. } => "openai:tool-input-start",
            LanguageModelV3StreamPart::ToolInputDelta { .. } => "openai:tool-input-delta",
            LanguageModelV3StreamPart::ToolInputEnd { .. } => "openai:tool-input-end",
            LanguageModelV3StreamPart::ToolApprovalRequest(_) => "openai:tool-approval-request",
            LanguageModelV3StreamPart::ToolCall(_) => "openai:tool-call",
            LanguageModelV3StreamPart::ToolResult(_) => "openai:tool-result",
            LanguageModelV3StreamPart::Custom(_) => "openai:custom",
            LanguageModelV3StreamPart::Source(_) => "openai:source",
            LanguageModelV3StreamPart::StreamStart { .. } => "openai:stream-start",
            LanguageModelV3StreamPart::ResponseMetadata(_) => "openai:response-metadata",
            LanguageModelV3StreamPart::Finish { .. } => "openai:finish",
            LanguageModelV3StreamPart::Error { .. } => "openai:error",
            LanguageModelV3StreamPart::ReasoningFile(_) => "openai:reasoning-file",
            LanguageModelV3StreamPart::Raw { .. } | LanguageModelV3StreamPart::File(_) => {
                return None;
            }
        };
        Some((event_type.to_string(), data))
    }

    fn to_anthropic_custom_event_payload(&self) -> Option<(String, serde_json::Value)> {
        let data = serde_json::to_value(self).ok()?;
        let event_type = match self {
            LanguageModelV3StreamPart::TextStart { .. } => "anthropic:text-start",
            LanguageModelV3StreamPart::TextDelta { .. } => "anthropic:text-delta",
            LanguageModelV3StreamPart::TextEnd { .. } => "anthropic:text-end",
            LanguageModelV3StreamPart::ReasoningStart { .. } => "anthropic:reasoning-start",
            LanguageModelV3StreamPart::ReasoningDelta { .. } => "anthropic:reasoning-delta",
            LanguageModelV3StreamPart::ReasoningEnd { .. } => "anthropic:reasoning-end",
            LanguageModelV3StreamPart::ToolInputStart { .. } => "anthropic:tool-input-start",
            LanguageModelV3StreamPart::ToolInputDelta { .. } => "anthropic:tool-input-delta",
            LanguageModelV3StreamPart::ToolInputEnd { .. } => "anthropic:tool-input-end",
            LanguageModelV3StreamPart::ToolCall(_) => "anthropic:tool-call",
            LanguageModelV3StreamPart::ToolResult(_) => "anthropic:tool-result",
            LanguageModelV3StreamPart::Custom(_) => "anthropic:custom",
            LanguageModelV3StreamPart::Source(_) => "anthropic:source",
            LanguageModelV3StreamPart::StreamStart { .. } => "anthropic:stream-start",
            LanguageModelV3StreamPart::ResponseMetadata(_) => "anthropic:response-metadata",
            LanguageModelV3StreamPart::Finish { .. } => "anthropic:finish",
            LanguageModelV3StreamPart::Error { .. } => "anthropic:error",

            LanguageModelV3StreamPart::ToolApprovalRequest(_)
            | LanguageModelV3StreamPart::Raw { .. }
            | LanguageModelV3StreamPart::File(_)
            | LanguageModelV3StreamPart::ReasoningFile(_) => return None,
        };
        Some((event_type.to_string(), data))
    }

    fn to_gemini_custom_event_payload(&self) -> Option<(String, serde_json::Value)> {
        let data = serde_json::to_value(self).ok()?;
        let event_type = match self {
            LanguageModelV3StreamPart::ToolCall(_) | LanguageModelV3StreamPart::ToolResult(_) => {
                "gemini:tool"
            }
            LanguageModelV3StreamPart::Custom(_) => "gemini:custom",
            LanguageModelV3StreamPart::Source(_) => "gemini:source",
            LanguageModelV3StreamPart::ReasoningStart { .. }
            | LanguageModelV3StreamPart::ReasoningDelta { .. }
            | LanguageModelV3StreamPart::ReasoningEnd { .. } => "gemini:reasoning",
            LanguageModelV3StreamPart::ReasoningFile(_) => "gemini:reasoning-file",
            _ => return None,
        };
        Some((event_type.to_string(), data))
    }
}

impl LanguageModelV4StreamPart {
    /// Convert a stable runtime semantic part into the provider-facing V4 overlay.
    pub fn from_runtime_part(part: ChatStreamPart) -> Self {
        LanguageModelV3StreamPart::from_runtime_part(part).into()
    }

    /// Convert this V4 stream part into the stable runtime semantic part.
    pub fn to_runtime_part(&self) -> ChatStreamPart {
        let v3: LanguageModelV3StreamPart = self.clone().into();
        v3.to_runtime_part()
    }

    /// Best-effort parse a V4 stream part from JSON while preserving V4 serde validation.
    pub fn parse_loose_json(value: &serde_json::Value) -> Option<Self> {
        let mut v = value.clone();

        fn normalize_in_place(v: &mut serde_json::Value) {
            let Some(obj) = v.as_object_mut() else {
                return;
            };
            let Some(tpe) = obj.get("type").and_then(|v| v.as_str()) else {
                return;
            };

            match tpe {
                "tool-call" => {
                    if let Some(input) = obj.get_mut("input")
                        && !matches!(input, serde_json::Value::String(_))
                        && let Ok(s) = serde_json::to_string(input)
                    {
                        *input = serde_json::Value::String(s);
                    }
                }
                "finish" => {
                    if let Some(fr) = obj.get_mut("finishReason")
                        && let serde_json::Value::String(s) = fr
                    {
                        *fr = serde_json::json!({ "unified": s, "raw": serde_json::Value::Null });
                    }
                }
                "source" => {
                    if obj.get("sourceType").is_none()
                        && let Some(st) = obj.remove("source_type")
                    {
                        obj.insert("sourceType".to_string(), st);
                    }
                }
                _ => {}
            }
        }

        normalize_in_place(&mut v);
        serde_json::from_value::<LanguageModelV4StreamPart>(v).ok()
    }

    /// Best-effort parse from a `ChatStreamEvent`.
    pub fn try_from_chat_event(ev: &ChatStreamEvent) -> Option<Self> {
        match ev {
            ChatStreamEvent::Part { part } => Some(Self::from_runtime_part(part.clone())),
            ChatStreamEvent::PartWithReplay { part, .. } => {
                Some(Self::from_runtime_part(part.clone()))
            }
            ChatStreamEvent::Custom { data, .. } => Self::parse_loose_json(data),
            ChatStreamEvent::Error { error } => Some(Self::Error {
                error: serde_json::json!(error),
            }),
            _ => None,
        }
    }

    /// Format as a provider-specific protocol-side `ChatStreamEvent::Custom` (best-effort).
    pub fn to_protocol_custom_event(&self, ns: StreamPartNamespace) -> Option<ChatStreamEvent> {
        let data = serde_json::to_value(self).ok()?;
        let v3: LanguageModelV3StreamPart = self.clone().into();
        let ChatStreamEvent::Custom { event_type, .. } = v3.to_protocol_custom_event(ns)? else {
            return None;
        };

        Some(ChatStreamEvent::Custom { event_type, data })
    }

    /// Legacy compatibility alias for provider protocol serializers.
    #[inline]
    pub fn to_custom_event(&self, ns: StreamPartNamespace) -> Option<ChatStreamEvent> {
        self.to_protocol_custom_event(ns)
    }

    /// Convert this typed stream part into a first-class runtime stream part event.
    pub fn to_part_event(&self) -> ChatStreamEvent {
        ChatStreamEvent::Part {
            part: self.to_runtime_part(),
        }
    }

    /// Encode this stream part as an SSE `data: ...\n\n` frame.
    pub fn to_data_sse_bytes(&self) -> Result<Vec<u8>, LlmError> {
        let json = serde_json::to_string(self).map_err(|e| {
            LlmError::ParseError(format!("Failed to serialize stream part JSON: {e}"))
        })?;
        Ok(format!("data: {json}\n\n").into_bytes())
    }

    /// Convert this typed stream part into best-effort `ChatStreamEvent`s.
    pub fn to_best_effort_chat_events(&self) -> Vec<ChatStreamEvent> {
        let v3: LanguageModelV3StreamPart = self.clone().into();
        v3.to_best_effort_chat_events()
    }

    /// Convert a V4 stream part into a lossy text representation.
    pub fn to_lossy_text(&self) -> Option<String> {
        let v3: LanguageModelV3StreamPart = self.clone().into();
        v3.to_lossy_text()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_usage_omits_unknown_token_fields() {
        let usage = LanguageModelV3Usage {
            input_tokens: LanguageModelV3InputTokens {
                total: None,
                no_cache: None,
                cache_read: None,
                cache_write: None,
            },
            output_tokens: LanguageModelV3OutputTokens {
                total: None,
                text: None,
                reasoning: None,
            },
            raw: None,
        };

        let value = serde_json::to_value(&usage).expect("serialize stream usage");
        assert_eq!(value["inputTokens"], serde_json::json!({}));
        assert_eq!(value["outputTokens"], serde_json::json!({}));
    }

    #[test]
    fn stream_part_parses_from_custom_event_payload() {
        let ev = ChatStreamEvent::Custom {
            event_type: "openai:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": "0",
                "delta": "hello"
            }),
        };

        let part = LanguageModelV3StreamPart::try_from_chat_event(&ev).expect("parsed");
        match part {
            LanguageModelV3StreamPart::TextDelta { id, delta, .. } => {
                assert_eq!(id, "0");
                assert_eq!(delta, "hello");
            }
            other => panic!("unexpected part: {other:?}"),
        }
    }

    #[test]
    fn stream_part_formats_to_openai_protocol_custom_event() {
        let part = LanguageModelV3StreamPart::ToolCall(LanguageModelV3ToolCall {
            tool_call_id: "call_1".to_string(),
            tool_name: "web_search".to_string(),
            input: "{}".to_string(),
            provider_executed: Some(true),
            dynamic: None,
            provider_metadata: None,
        });

        let ev = part
            .to_protocol_custom_event(StreamPartNamespace::OpenAi)
            .expect("custom event");

        match ev {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-call");
                assert_eq!(data.get("type").and_then(|v| v.as_str()), Some("tool-call"));
            }
            _ => panic!("expected custom event"),
        }
    }

    #[test]
    fn stream_part_roundtrips_runtime_tool_call_part() {
        let runtime_part = ChatStreamPart::ToolCall(ChatStreamToolCall {
            tool_call_id: "call_1".to_string(),
            tool_name: "weather".to_string(),
            input: "{\"city\":\"Tokyo\"}".to_string(),
            provider_executed: Some(true),
            dynamic: Some(false),
            provider_metadata: Some(std::collections::HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "itemId": "fc_1" }),
            )])),
        });

        let part = LanguageModelV3StreamPart::from_runtime_part(runtime_part);
        match &part {
            LanguageModelV3StreamPart::ToolCall(call) => {
                assert_eq!(call.tool_call_id, "call_1");
                assert_eq!(call.tool_name, "weather");
                assert_eq!(call.input, "{\"city\":\"Tokyo\"}");
                assert_eq!(call.provider_executed, Some(true));
                assert_eq!(call.dynamic, Some(false));
                assert_eq!(
                    call.provider_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("openai"))
                        .and_then(|metadata| metadata.get("itemId"))
                        .and_then(|value| value.as_str()),
                    Some("fc_1")
                );
            }
            other => panic!("unexpected part: {other:?}"),
        }

        match part.to_runtime_part() {
            ChatStreamPart::ToolCall(call) => {
                assert_eq!(call.tool_call_id, "call_1");
                assert_eq!(call.tool_name, "weather");
                assert_eq!(call.input, "{\"city\":\"Tokyo\"}");
                assert_eq!(call.provider_executed, Some(true));
                assert_eq!(call.dynamic, Some(false));
                assert_eq!(
                    call.provider_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("openai"))
                        .and_then(|metadata| metadata.get("itemId"))
                        .and_then(|value| value.as_str()),
                    Some("fc_1")
                );
            }
            other => panic!("unexpected runtime part: {other:?}"),
        }
    }

    #[test]
    fn stream_part_formats_reasoning_to_anthropic_protocol_custom_event_with_provider_metadata() {
        let part = LanguageModelV3StreamPart::ReasoningStart {
            id: "0".to_string(),
            provider_metadata: Some(std::collections::HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "contentBlockIndex": 0,
                    "redactedData": "abc123"
                }),
            )])),
        };

        let ev = part
            .to_protocol_custom_event(StreamPartNamespace::Anthropic)
            .expect("custom event");

        match ev {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "anthropic:reasoning-start");
                assert_eq!(
                    data.get("type").and_then(|v| v.as_str()),
                    Some("reasoning-start")
                );
                assert_eq!(data.get("id").and_then(|v| v.as_str()), Some("0"));
                assert_eq!(
                    data.pointer("/providerMetadata/anthropic/redactedData"),
                    Some(&serde_json::json!("abc123"))
                );
            }
            _ => panic!("expected custom event"),
        }
    }

    #[test]
    fn stream_part_formats_as_data_sse_frame() {
        let part = LanguageModelV3StreamPart::TextStart {
            id: "0".to_string(),
            provider_metadata: None,
        };
        let bytes = part.to_data_sse_bytes().expect("bytes");
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.starts_with("data: "));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn stream_part_parse_loose_accepts_tool_call_input_object() {
        let v = serde_json::json!({
            "type": "tool-call",
            "toolCallId": "call_1",
            "toolName": "web_search",
            "input": { "query": "rust" }
        });

        let part = LanguageModelV3StreamPart::parse_loose_json(&v).expect("parsed");
        match part {
            LanguageModelV3StreamPart::ToolCall(call) => {
                assert_eq!(call.tool_call_id, "call_1");
                assert_eq!(call.tool_name, "web_search");
                assert!(call.input.contains("\"query\""));
            }
            other => panic!("unexpected part: {other:?}"),
        }
    }

    #[test]
    fn stream_part_parse_loose_accepts_finish_reason_string() {
        let v = serde_json::json!({
            "type": "finish",
            "usage": {
                "inputTokens": { "total": 1 },
                "outputTokens": { "total": 2 }
            },
            "finishReason": "stop"
        });

        let part = LanguageModelV3StreamPart::parse_loose_json(&v).expect("parsed");
        match part {
            LanguageModelV3StreamPart::Finish { finish_reason, .. } => {
                assert_eq!(finish_reason.unified, "stop");
            }
            other => panic!("unexpected part: {other:?}"),
        }
    }

    #[test]
    fn stream_part_finish_reason_uses_ai_sdk_v4_unified_values() {
        let usage = crate::types::Usage::unknown();
        let part = LanguageModelV3StreamPart::from_runtime_part(ChatStreamPart::Finish {
            usage,
            finish_reason: ChatStreamFinishInfo {
                unified: crate::types::FinishReason::ToolCalls,
                raw: None,
            },
            provider_metadata: None,
        });

        let value = serde_json::to_value(&part).expect("serialize stream part");
        assert_eq!(
            value.pointer("/finishReason/unified"),
            Some(&serde_json::json!("tool-calls"))
        );
    }

    #[test]
    fn stream_part_finish_reason_preserves_stop_sequence_as_raw() {
        let usage = crate::types::Usage::unknown();
        let part = LanguageModelV3StreamPart::from_runtime_part(ChatStreamPart::Finish {
            usage,
            finish_reason: ChatStreamFinishInfo {
                unified: crate::types::FinishReason::StopSequence,
                raw: None,
            },
            provider_metadata: None,
        });

        let value = serde_json::to_value(&part).expect("serialize stream part");
        assert_eq!(
            value.pointer("/finishReason/unified"),
            Some(&serde_json::json!("stop"))
        );
        assert_eq!(
            value.pointer("/finishReason/raw"),
            Some(&serde_json::json!("stop_sequence"))
        );

        let runtime = part.to_runtime_part();
        match runtime {
            ChatStreamPart::Finish { finish_reason, .. } => {
                assert_eq!(
                    finish_reason.unified,
                    crate::types::FinishReason::StopSequence
                );
                assert_eq!(finish_reason.raw.as_deref(), Some("stop_sequence"));
            }
            other => panic!("unexpected runtime part: {other:?}"),
        }
    }

    #[test]
    fn stream_part_finish_reason_accepts_legacy_underscore_values() {
        let part = LanguageModelV3StreamPart::Finish {
            usage: LanguageModelV3Usage {
                input_tokens: LanguageModelV3InputTokens {
                    total: None,
                    no_cache: None,
                    cache_read: None,
                    cache_write: None,
                },
                output_tokens: LanguageModelV3OutputTokens {
                    total: None,
                    text: None,
                    reasoning: None,
                },
                raw: None,
            },
            finish_reason: LanguageModelV3FinishReason {
                unified: "content_filter".to_string(),
                raw: Some("safety".to_string()),
            },
            provider_metadata: None,
        };

        let runtime = part.to_runtime_part();
        match runtime {
            ChatStreamPart::Finish { finish_reason, .. } => {
                assert_eq!(
                    finish_reason.unified,
                    crate::types::FinishReason::ContentFilter
                );
                assert_eq!(finish_reason.raw.as_deref(), Some("safety"));
            }
            other => panic!("unexpected runtime part: {other:?}"),
        }
    }

    #[test]
    fn stream_part_parse_loose_accepts_custom_content() {
        let v = serde_json::json!({
            "type": "custom",
            "kind": "openai.compaction",
            "providerMetadata": {
                "openai": {
                    "itemId": "cmp_123"
                }
            }
        });

        let part = LanguageModelV3StreamPart::parse_loose_json(&v).expect("parsed");
        match part {
            LanguageModelV3StreamPart::Custom(custom) => {
                assert_eq!(custom.kind, "openai.compaction");
                assert_eq!(
                    custom
                        .provider_metadata
                        .as_ref()
                        .and_then(|m| m.get("openai"))
                        .and_then(|m| m.get("itemId"))
                        .and_then(|v| v.as_str()),
                    Some("cmp_123")
                );
            }
            other => panic!("unexpected part: {other:?}"),
        }
    }

    #[test]
    fn language_model_v4_stream_provider_metadata_requires_object_values() {
        let invalid = serde_json::json!({
            "type": "text-start",
            "id": "0",
            "providerMetadata": {
                "openai": "not-an-object"
            }
        });
        assert!(serde_json::from_value::<LanguageModelV4StreamPart>(invalid.clone()).is_err());
        assert!(serde_json::from_value::<LanguageModelV3StreamPart>(invalid).is_ok());

        let invalid_part = LanguageModelV4StreamPart::ToolCall(LanguageModelV4StreamToolCall {
            tool_call_id: "call_1".to_string(),
            tool_name: "weather".to_string(),
            input: "{}".to_string(),
            provider_executed: None,
            dynamic: None,
            provider_metadata: Some(std::collections::HashMap::from([(
                "openai".to_string(),
                serde_json::json!(false),
            )])),
        });
        assert!(serde_json::to_value(&invalid_part).is_err());
        assert!(
            invalid_part
                .to_protocol_custom_event(StreamPartNamespace::OpenAi)
                .is_none()
        );

        let invalid_custom =
            LanguageModelV4StreamPart::Custom(LanguageModelV4StreamCustomContent {
                kind: "compaction".to_string(),
                provider_metadata: None,
            });
        assert!(serde_json::to_value(&invalid_custom).is_err());
        assert!(
            invalid_custom
                .to_protocol_custom_event(StreamPartNamespace::OpenAi)
                .is_none()
        );

        let valid = serde_json::json!({
            "type": "tool-call",
            "toolCallId": "call_1",
            "toolName": "weather",
            "input": "{}",
            "providerMetadata": {
                "openai": {
                    "itemId": "fc_1"
                }
            }
        });
        let part: LanguageModelV4StreamPart =
            serde_json::from_value(valid.clone()).expect("deserialize V4 stream part");
        assert_eq!(
            serde_json::to_value(&part).expect("serialize V4 stream part"),
            valid
        );
    }

    #[test]
    fn language_model_v4_stream_projection_filters_non_object_provider_metadata() {
        let metadata = std::collections::HashMap::from([
            (
                "openai".to_string(),
                serde_json::json!({ "itemId": "msg_1" }),
            ),
            ("legacy".to_string(), serde_json::json!("drop")),
        ]);
        let part = LanguageModelV4StreamPart::from_runtime_part(ChatStreamPart::TextStart {
            id: "0".to_string(),
            provider_metadata: Some(metadata),
        });
        let value = serde_json::to_value(&part).expect("serialize projected V4 stream part");

        assert_eq!(
            value["providerMetadata"],
            serde_json::json!({
                "openai": {
                    "itemId": "msg_1"
                }
            })
        );
        assert!(value["providerMetadata"].get("legacy").is_none());
    }

    #[test]
    fn stream_part_tool_result_rejects_null_result_payload() {
        let invalid = serde_json::json!({
            "type": "tool-result",
            "toolCallId": "call_1",
            "toolName": "weather",
            "result": null
        });

        assert!(serde_json::from_value::<LanguageModelV3StreamPart>(invalid).is_err());

        let part = LanguageModelV3StreamPart::ToolResult(LanguageModelV3ToolResult {
            tool_call_id: "call_1".to_string(),
            tool_name: "weather".to_string(),
            result: serde_json::Value::Null,
            is_error: None,
            preliminary: None,
            dynamic: None,
            provider_metadata: None,
        });
        assert!(serde_json::to_value(&part).is_err());
    }

    #[test]
    fn stream_part_formats_reasoning_file_to_openai_protocol_custom_event() {
        let part = LanguageModelV3StreamPart::ReasoningFile(LanguageModelV3ReasoningFile {
            media_type: "image/png".to_string(),
            data: LanguageModelV3FileData::Base64("ZmFrZQ==".to_string()),
            provider_metadata: Some(std::collections::HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "itemId": "rs_1" }),
            )])),
        });

        let ev = part
            .to_protocol_custom_event(StreamPartNamespace::OpenAi)
            .expect("custom event");

        match ev {
            ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:reasoning-file");
                assert_eq!(
                    data.get("type").and_then(|v| v.as_str()),
                    Some("reasoning-file")
                );
                assert_eq!(
                    data.get("mediaType").and_then(|v| v.as_str()),
                    Some("image/png")
                );
            }
            _ => panic!("expected custom event"),
        }
    }

    #[test]
    fn stream_part_lossy_text_includes_custom_kind_and_reasoning_file_hint() {
        let custom = LanguageModelV3StreamPart::Custom(LanguageModelV3CustomContent {
            kind: "openai.compaction".to_string(),
            provider_metadata: None,
        });
        assert_eq!(
            custom.to_lossy_text().as_deref(),
            Some("[custom] openai.compaction")
        );

        let reasoning_file =
            LanguageModelV3StreamPart::ReasoningFile(LanguageModelV3ReasoningFile {
                media_type: "image/png".to_string(),
                data: LanguageModelV3FileData::Base64("ZmFrZQ==".to_string()),
                provider_metadata: None,
            });
        assert_eq!(
            reasoning_file.to_lossy_text().as_deref(),
            Some("[reasoning-file] mediaType=image/png base64_len=8")
        );
    }
}
