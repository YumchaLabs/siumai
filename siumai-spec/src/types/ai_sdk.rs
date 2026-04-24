//! AI SDK-aligned shared surface aliases and metadata helpers.
//!
//! These names intentionally mirror the shared `packages/ai/src/types/*` contract where
//! Siumai already has a stable equivalent or can expose a passive data structure honestly
//! without pretending the runtime wiring is more complete than it is today.

use super::chat::{ContentPart, SourcePart};
use super::{
    AssistantModelMessage, DataContent, EmbeddingUsage, FinishReason, HttpRequestInfo,
    HttpResponseInfo, ModelMessage, ProviderMetadataMap, ProviderOptionsMap, ResponseMetadata,
    StandardizedPrompt, Tool, ToolChoice, ToolModelMessage, ToolResultOutput, Usage, Warning,
};
use base64::{Engine, engine::general_purpose::STANDARD};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::time::Duration;
use tokio_util::sync::{CancellationToken, WaitForCancellationFuture};

/// AI SDK-style JSON value alias.
pub type JSONValue = serde_json::Value;

/// AI SDK-style JSON Schema draft-07 value alias.
pub type JSONSchema7 = serde_json::Value;

/// AI SDK-style shared warning alias.
pub type CallWarning = Warning;

/// AI SDK-style shared provider-metadata root.
pub type ProviderMetadata = ProviderMetadataMap;

/// AI SDK-style shared provider-options root.
pub type ProviderOptions = ProviderOptionsMap;

/// AI SDK-style shared execution context object.
pub type Context = HashMap<String, JSONValue>;

/// AI SDK-style single embedding vector.
pub type Embedding = Vec<f32>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceMarker {
    Source,
}

impl Default for SourceMarker {
    fn default() -> Self {
        Self::Source
    }
}

impl Serialize for SourceMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("source")
    }
}

impl<'de> Deserialize<'de> for SourceMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "source" {
            Ok(Self::Source)
        } else {
            Err(serde::de::Error::custom(format!(
                "expected source type marker `source`, got `{value}`"
            )))
        }
    }
}

/// AI SDK-style source citation used by language-model responses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Source {
    /// Fixed AI SDK type marker. Serialized as `type: "source"`.
    #[serde(rename = "type", default)]
    kind: SourceMarker,
    /// Source id.
    pub id: String,
    /// Strict URL/document source union.
    #[serde(flatten)]
    pub source: SourcePart,
    /// Additional provider metadata for the source.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl Source {
    /// Create a URL-backed source without a title.
    pub fn url(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            kind: SourceMarker::Source,
            id: id.into(),
            source: SourcePart::Url {
                url: url.into(),
                title: None,
            },
            provider_metadata: None,
        }
    }

    /// Create a URL-backed source with a title.
    pub fn url_with_title(
        id: impl Into<String>,
        url: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        Self {
            kind: SourceMarker::Source,
            id: id.into(),
            source: SourcePart::Url {
                url: url.into(),
                title: Some(title.into()),
            },
            provider_metadata: None,
        }
    }

    /// Create a document-backed source.
    pub fn document(
        id: impl Into<String>,
        media_type: impl Into<String>,
        title: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self {
            kind: SourceMarker::Source,
            id: id.into(),
            source: SourcePart::Document {
                media_type: media_type.into(),
                title: title.into(),
                filename,
            },
            provider_metadata: None,
        }
    }

    /// Return the fixed AI SDK source marker.
    pub const fn r#type(&self) -> &'static str {
        "source"
    }

    /// Return the source-type discriminator.
    pub fn source_type(&self) -> &'static str {
        self.source.source_type()
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

impl From<Source> for ContentPart {
    fn from(value: Source) -> Self {
        Self::Source {
            id: value.id,
            source: value.source,
            provider_metadata: value.provider_metadata,
        }
    }
}

impl TryFrom<ContentPart> for Source {
    type Error = ContentPart;

    fn try_from(value: ContentPart) -> Result<Self, Self::Error> {
        match value {
            ContentPart::Source {
                id,
                source,
                provider_metadata,
            } => Ok(Self {
                kind: SourceMarker::Source,
                id,
                source,
                provider_metadata,
            }),
            other => Err(other),
        }
    }
}

impl TryFrom<&ContentPart> for Source {
    type Error = &'static str;

    fn try_from(value: &ContentPart) -> Result<Self, Self::Error> {
        match value {
            ContentPart::Source {
                id,
                source,
                provider_metadata,
            } => Ok(Self {
                kind: SourceMarker::Source,
                id: id.clone(),
                source: source.clone(),
                provider_metadata: provider_metadata.clone(),
            }),
            _ => Err("content part is not a source"),
        }
    }
}

/// AI SDK-style shared image-provider metadata root.
pub type ImageModelProviderMetadata = ProviderMetadata;

/// AI SDK-style shared video-provider metadata root.
pub type VideoModelProviderMetadata = ProviderMetadata;

/// AI SDK-style generated file returned by text helper output parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeneratedFile {
    /// File content as a base64 encoded string.
    pub base64: String,
    /// IANA media type of the file.
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
}

impl GeneratedFile {
    /// Create a generated file from base64 content.
    pub fn from_base64(base64: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            base64: base64.into(),
            media_type: media_type.into(),
        }
    }

    /// Create a generated file from bytes.
    pub fn from_bytes(data: impl AsRef<[u8]>, media_type: impl Into<String>) -> Self {
        Self::from_base64(STANDARD.encode(data.as_ref()), media_type)
    }

    /// Return base64 content.
    pub fn base64(&self) -> &str {
        self.base64.as_str()
    }

    /// Decode the file into bytes.
    pub fn uint8_array(&self) -> Result<Vec<u8>, base64::DecodeError> {
        STANDARD.decode(&self.base64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextOutputMarker {
    Text,
}

impl Default for TextOutputMarker {
    fn default() -> Self {
        Self::Text
    }
}

impl Serialize for TextOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("text")
    }
}

impl<'de> Deserialize<'de> for TextOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "text" {
            Ok(Self::Text)
        } else {
            Err(serde::de::Error::custom("expected text type marker"))
        }
    }
}

/// AI SDK-style text output content part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextOutput {
    #[serde(rename = "type", default)]
    marker: TextOutputMarker,
    /// Generated text.
    pub text: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextOutput {
    /// Create a text output content part.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            marker: TextOutputMarker::Text,
            text: text.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "text"
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CustomOutputMarker {
    Custom,
}

impl Default for CustomOutputMarker {
    fn default() -> Self {
        Self::Custom
    }
}

impl Serialize for CustomOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("custom")
    }
}

impl<'de> Deserialize<'de> for CustomOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "custom" {
            Ok(Self::Custom)
        } else {
            Err(serde::de::Error::custom("expected custom type marker"))
        }
    }
}

/// AI SDK-style custom output content part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomOutput {
    #[serde(rename = "type", default)]
    marker: CustomOutputMarker,
    /// Provider-specific custom output kind, e.g. `openai.compaction`.
    pub kind: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl CustomOutput {
    /// Create a custom output content part.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            marker: CustomOutputMarker::Custom,
            kind: kind.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "custom"
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileOutputMarker {
    File,
}

impl Default for FileOutputMarker {
    fn default() -> Self {
        Self::File
    }
}

impl Serialize for FileOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("file")
    }
}

impl<'de> Deserialize<'de> for FileOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "file" {
            Ok(Self::File)
        } else {
            Err(serde::de::Error::custom("expected file type marker"))
        }
    }
}

/// AI SDK-style generated file output content part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileOutput {
    #[serde(rename = "type", default)]
    marker: FileOutputMarker,
    /// Generated file payload.
    pub file: GeneratedFile,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl FileOutput {
    /// Create a generated file output content part.
    pub fn new(file: GeneratedFile) -> Self {
        Self {
            marker: FileOutputMarker::File,
            file,
            provider_metadata: None,
        }
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "file"
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningOutputMarker {
    Reasoning,
}

impl Default for ReasoningOutputMarker {
    fn default() -> Self {
        Self::Reasoning
    }
}

impl Serialize for ReasoningOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("reasoning")
    }
}

impl<'de> Deserialize<'de> for ReasoningOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "reasoning" {
            Ok(Self::Reasoning)
        } else {
            Err(serde::de::Error::custom("expected reasoning type marker"))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningFileOutputMarker {
    ReasoningFile,
}

impl Default for ReasoningFileOutputMarker {
    fn default() -> Self {
        Self::ReasoningFile
    }
}

impl Serialize for ReasoningFileOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("reasoning-file")
    }
}

impl<'de> Deserialize<'de> for ReasoningFileOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "reasoning-file" {
            Ok(Self::ReasoningFile)
        } else {
            Err(serde::de::Error::custom(
                "expected reasoning-file type marker",
            ))
        }
    }
}

/// AI SDK-style reasoning output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReasoningOutput {
    #[serde(rename = "type", default)]
    marker: ReasoningOutputMarker,
    /// Reasoning text.
    pub text: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl ReasoningOutput {
    /// Create a reasoning output part.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            marker: ReasoningOutputMarker::Reasoning,
            text: text.into(),
            provider_metadata: None,
        }
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning"
    }
}

/// AI SDK-style reasoning-file output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReasoningFileOutput {
    #[serde(rename = "type", default)]
    marker: ReasoningFileOutputMarker,
    /// Generated file attached to model reasoning.
    pub file: GeneratedFile,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl ReasoningFileOutput {
    /// Create a reasoning-file output part.
    pub fn new(file: GeneratedFile) -> Self {
        Self {
            marker: ReasoningFileOutputMarker::ReasoningFile,
            file,
            provider_metadata: None,
        }
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning-file"
    }
}

/// AI SDK-style typed tool call view returned by higher-level text helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallMarker {
    ToolCall,
}

impl Default for ToolCallMarker {
    fn default() -> Self {
        Self::ToolCall
    }
}

impl Serialize for ToolCallMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-call")
    }
}

impl<'de> Deserialize<'de> for ToolCallMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-call" {
            Ok(Self::ToolCall)
        } else {
            Err(serde::de::Error::custom("expected tool-call type marker"))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall<NAME = String, INPUT = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ToolCallMarker,
    /// ID of the tool call. This ID is used to match the tool call with the tool result.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool being called.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Tool arguments.
    pub input: INPUT,
    /// Whether the tool call will be executed by the provider.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Whether the tool is dynamic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    /// Whether the tool call is invalid, e.g. caused by invalid input or an unknown tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalid: Option<bool>,
    /// Error payload explaining why the tool call is invalid when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JSONValue>,
    /// Human-readable title for the tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl<NAME, INPUT> ToolCall<NAME, INPUT> {
    /// Create a typed tool call.
    pub fn new(tool_call_id: impl Into<String>, tool_name: NAME, input: INPUT) -> Self {
        Self {
            marker: ToolCallMarker::ToolCall,
            tool_call_id: tool_call_id.into(),
            tool_name,
            input,
            provider_executed: None,
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_metadata: None,
        }
    }

    /// Mark whether the tool call is provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Mark whether the tool call is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Mark whether the tool call is invalid.
    pub const fn with_invalid(mut self, invalid: bool) -> Self {
        self.invalid = Some(invalid);
        self
    }

    /// Attach an invalid-tool-call error payload.
    pub fn with_error(mut self, error: impl Into<JSONValue>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Attach a human-readable title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-call"
    }
}

/// AI SDK-style typed tool result view returned by higher-level text helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolResultMarker {
    ToolResult,
}

impl Default for ToolResultMarker {
    fn default() -> Self {
        Self::ToolResult
    }
}

impl Serialize for ToolResultMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-result")
    }
}

impl<'de> Deserialize<'de> for ToolResultMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-result" {
            Ok(Self::ToolResult)
        } else {
            Err(serde::de::Error::custom("expected tool-result type marker"))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    #[serde(rename = "type", default)]
    marker: ToolResultMarker,
    /// ID of the tool call. This ID is used to match the tool call with the tool result.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool that was called.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Tool input.
    pub input: INPUT,
    /// Tool output.
    pub output: OUTPUT,
    /// Whether the tool result was executed by the provider.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Whether the tool is dynamic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Whether the result is preliminary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
    /// Human-readable title for the tool result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl<NAME, INPUT, OUTPUT> ToolResult<NAME, INPUT, OUTPUT> {
    /// Create a typed tool result.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: NAME,
        input: INPUT,
        output: OUTPUT,
    ) -> Self {
        Self {
            marker: ToolResultMarker::ToolResult,
            tool_call_id: tool_call_id.into(),
            tool_name,
            input,
            output,
            provider_executed: None,
            dynamic: None,
            provider_metadata: None,
            preliminary: None,
            title: None,
        }
    }

    /// Mark whether the tool result is provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Mark whether the tool is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Mark whether the tool result is preliminary.
    pub const fn with_preliminary(mut self, preliminary: bool) -> Self {
        self.preliminary = Some(preliminary);
        self
    }

    /// Attach a human-readable title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-result"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolErrorMarker {
    ToolError,
}

impl Default for ToolErrorMarker {
    fn default() -> Self {
        Self::ToolError
    }
}

impl Serialize for ToolErrorMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-error")
    }
}

impl<'de> Deserialize<'de> for ToolErrorMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-error" {
            Ok(Self::ToolError)
        } else {
            Err(serde::de::Error::custom("expected tool-error type marker"))
        }
    }
}

/// AI SDK-style `tool-error` output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolError<NAME = String, INPUT = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ToolErrorMarker,
    /// ID of the tool call.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool that failed.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Tool input.
    pub input: INPUT,
    /// Error payload.
    pub error: JSONValue,
    /// Whether the tool call was provider-executed.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Whether the tool is dynamic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    /// Human-readable title for the tool error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl<NAME, INPUT> ToolError<NAME, INPUT> {
    /// Create a tool error output part.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: NAME,
        input: INPUT,
        error: impl Into<JSONValue>,
    ) -> Self {
        Self {
            marker: ToolErrorMarker::ToolError,
            tool_call_id: tool_call_id.into(),
            tool_name,
            input,
            error: error.into(),
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            title: None,
        }
    }

    /// Mark whether the tool call was provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Mark whether the tool is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Attach a human-readable title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-error"
    }
}

/// AI SDK-style tool execution output union.
///
/// This mirrors `generate-text/tool-output.ts`: successful tool executions carry
/// `tool-result`, while failed executions carry `tool-error`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolOutput<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// Successful tool output.
    Result(ToolResult<NAME, INPUT, OUTPUT>),
    /// Failed tool output.
    Error(ToolError<NAME, INPUT>),
}

impl<NAME, INPUT, OUTPUT> ToolOutput<NAME, INPUT, OUTPUT> {
    /// Return the AI SDK output discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Result(output) => output.r#type(),
            Self::Error(output) => output.r#type(),
        }
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolResult<NAME, INPUT, OUTPUT>>
    for ToolOutput<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolResult<NAME, INPUT, OUTPUT>) -> Self {
        Self::Result(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolError<NAME, INPUT>> for ToolOutput<NAME, INPUT, OUTPUT> {
    fn from(value: ToolError<NAME, INPUT>) -> Self {
        Self::Error(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolOutputDeniedMarker {
    ToolOutputDenied,
}

impl Default for ToolOutputDeniedMarker {
    fn default() -> Self {
        Self::ToolOutputDenied
    }
}

impl Serialize for ToolOutputDeniedMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-output-denied")
    }
}

impl<'de> Deserialize<'de> for ToolOutputDeniedMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-output-denied" {
            Ok(Self::ToolOutputDenied)
        } else {
            Err(serde::de::Error::custom(
                "expected tool-output-denied type marker",
            ))
        }
    }
}

/// AI SDK-style `tool-output-denied` output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolOutputDenied<NAME = String> {
    #[serde(rename = "type", default)]
    marker: ToolOutputDeniedMarker,
    /// ID of the tool call.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool whose output was denied.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Whether the tool call was provider-executed.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Whether the tool is dynamic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
}

impl<NAME> ToolOutputDenied<NAME> {
    /// Create a tool-output-denied output part.
    pub fn new(tool_call_id: impl Into<String>, tool_name: NAME) -> Self {
        Self {
            marker: ToolOutputDeniedMarker::ToolOutputDenied,
            tool_call_id: tool_call_id.into(),
            tool_name,
            provider_executed: None,
            dynamic: None,
        }
    }

    /// Mark whether the tool call was provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Mark whether the tool is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-output-denied"
    }
}

/// Static tool-output-denied view. Rust keeps the same runtime carrier as the typed view.
pub type StaticToolOutputDenied<NAME = String> = ToolOutputDenied<NAME>;

/// Typed tool-output-denied view. Rust keeps the same runtime carrier as the static view.
pub type TypedToolOutputDenied<NAME = String> = ToolOutputDenied<NAME>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolApprovalRequestOutputMarker {
    ToolApprovalRequest,
}

impl Default for ToolApprovalRequestOutputMarker {
    fn default() -> Self {
        Self::ToolApprovalRequest
    }
}

impl Serialize for ToolApprovalRequestOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-approval-request")
    }
}

impl<'de> Deserialize<'de> for ToolApprovalRequestOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-approval-request" {
            Ok(Self::ToolApprovalRequest)
        } else {
            Err(serde::de::Error::custom(
                "expected tool-approval-request type marker",
            ))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolApprovalResponseOutputMarker {
    ToolApprovalResponse,
}

impl Default for ToolApprovalResponseOutputMarker {
    fn default() -> Self {
        Self::ToolApprovalResponse
    }
}

impl Serialize for ToolApprovalResponseOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-approval-response")
    }
}

impl<'de> Deserialize<'de> for ToolApprovalResponseOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-approval-response" {
            Ok(Self::ToolApprovalResponse)
        } else {
            Err(serde::de::Error::custom(
                "expected tool-approval-response type marker",
            ))
        }
    }
}

/// AI SDK-style `tool-approval-request` output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolApprovalRequestOutput<NAME = String, INPUT = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ToolApprovalRequestOutputMarker,
    /// ID of the tool approval request.
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    /// Tool call that the approval request is for.
    #[serde(rename = "toolCall")]
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Whether the tool was automatically approved or denied.
    #[serde(
        rename = "isAutomatic",
        alias = "is_automatic",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub is_automatic: Option<bool>,
}

impl<NAME, INPUT> ToolApprovalRequestOutput<NAME, INPUT> {
    /// Create a tool approval request output part.
    pub fn new(approval_id: impl Into<String>, tool_call: ToolCall<NAME, INPUT>) -> Self {
        Self {
            marker: ToolApprovalRequestOutputMarker::ToolApprovalRequest,
            approval_id: approval_id.into(),
            tool_call,
            is_automatic: None,
        }
    }

    /// Mark whether the request was generated automatically.
    pub const fn with_is_automatic(mut self, is_automatic: bool) -> Self {
        self.is_automatic = Some(is_automatic);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-request"
    }
}

/// AI SDK-style `tool-approval-response` output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolApprovalResponseOutput<NAME = String, INPUT = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ToolApprovalResponseOutputMarker,
    /// ID of the tool approval.
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    /// Tool call that the approval response is for.
    #[serde(rename = "toolCall")]
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Whether the approval was granted or denied.
    pub approved: bool,
    /// Optional reason for the approval or denial.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Whether the tool call is provider-executed.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
}

impl<NAME, INPUT> ToolApprovalResponseOutput<NAME, INPUT> {
    /// Create a tool approval response output part.
    pub fn new(
        approval_id: impl Into<String>,
        tool_call: ToolCall<NAME, INPUT>,
        approved: bool,
    ) -> Self {
        Self {
            marker: ToolApprovalResponseOutputMarker::ToolApprovalResponse,
            approval_id: approval_id.into(),
            tool_call,
            approved,
            reason: None,
            provider_executed: None,
        }
    }

    /// Attach an optional human-readable approval reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Mark whether the approval response refers to a provider-executed tool call.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-response"
    }
}

/// AI SDK-style `generateText` content part union.
///
/// This is the output-side content union from `packages/ai/src/generate-text/content-part.ts`.
/// It intentionally stays separate from the prompt/runtime `ContentPart`, whose file and provider
/// option shapes represent request-side content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum GenerateTextContentPart<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    Text(TextOutput),
    Custom(CustomOutput),
    Reasoning(ReasoningOutput),
    ReasoningFile(ReasoningFileOutput),
    Source(Source),
    File(FileOutput),
    ToolCall(ToolCall<NAME, INPUT>),
    ToolResult(ToolResult<NAME, INPUT, OUTPUT>),
    ToolError(ToolError<NAME, INPUT>),
    ToolApprovalRequest(ToolApprovalRequestOutput<NAME, INPUT>),
    ToolApprovalResponse(ToolApprovalResponseOutput<NAME, INPUT>),
}

impl<NAME, INPUT, OUTPUT> GenerateTextContentPart<NAME, INPUT, OUTPUT> {
    /// Return the AI SDK content part discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Text(part) => part.r#type(),
            Self::Custom(part) => part.r#type(),
            Self::Reasoning(part) => part.r#type(),
            Self::ReasoningFile(part) => part.r#type(),
            Self::Source(part) => part.r#type(),
            Self::File(part) => part.r#type(),
            Self::ToolCall(part) => part.r#type(),
            Self::ToolResult(part) => part.r#type(),
            Self::ToolError(part) => part.r#type(),
            Self::ToolApprovalRequest(part) => part.r#type(),
            Self::ToolApprovalResponse(part) => part.r#type(),
        }
    }
}

impl<NAME, INPUT, OUTPUT> From<TextOutput> for GenerateTextContentPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextOutput) -> Self {
        Self::Text(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<CustomOutput> for GenerateTextContentPart<NAME, INPUT, OUTPUT> {
    fn from(value: CustomOutput) -> Self {
        Self::Custom(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ReasoningOutput> for GenerateTextContentPart<NAME, INPUT, OUTPUT> {
    fn from(value: ReasoningOutput) -> Self {
        Self::Reasoning(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ReasoningFileOutput>
    for GenerateTextContentPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ReasoningFileOutput) -> Self {
        Self::ReasoningFile(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<Source> for GenerateTextContentPart<NAME, INPUT, OUTPUT> {
    fn from(value: Source) -> Self {
        Self::Source(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<FileOutput> for GenerateTextContentPart<NAME, INPUT, OUTPUT> {
    fn from(value: FileOutput) -> Self {
        Self::File(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolCall<NAME, INPUT>>
    for GenerateTextContentPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolCall<NAME, INPUT>) -> Self {
        Self::ToolCall(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolResult<NAME, INPUT, OUTPUT>>
    for GenerateTextContentPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolResult<NAME, INPUT, OUTPUT>) -> Self {
        Self::ToolResult(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolError<NAME, INPUT>>
    for GenerateTextContentPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolError<NAME, INPUT>) -> Self {
        Self::ToolError(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolApprovalRequestOutput<NAME, INPUT>>
    for GenerateTextContentPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolApprovalRequestOutput<NAME, INPUT>) -> Self {
        Self::ToolApprovalRequest(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolApprovalResponseOutput<NAME, INPUT>>
    for GenerateTextContentPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolApprovalResponseOutput<NAME, INPUT>) -> Self {
        Self::ToolApprovalResponse(value)
    }
}

/// AI SDK-style response message generated by a `generateText` step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ResponseMessage {
    Assistant(AssistantModelMessage),
    Tool(ToolModelMessage),
}

impl From<AssistantModelMessage> for ResponseMessage {
    fn from(value: AssistantModelMessage) -> Self {
        Self::Assistant(value)
    }
}

impl From<ToolModelMessage> for ResponseMessage {
    fn from(value: ToolModelMessage) -> Self {
        Self::Tool(value)
    }
}

/// AI SDK-style reasoning projection returned by `GenerateTextResult::reasoning`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum GenerateTextReasoningPart {
    Reasoning(ReasoningOutput),
    ReasoningFile(ReasoningFileOutput),
}

impl GenerateTextReasoningPart {
    /// Return the AI SDK reasoning part discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Reasoning(part) => part.r#type(),
            Self::ReasoningFile(part) => part.r#type(),
        }
    }
}

impl From<ReasoningOutput> for GenerateTextReasoningPart {
    fn from(value: ReasoningOutput) -> Self {
        Self::Reasoning(value)
    }
}

impl From<ReasoningFileOutput> for GenerateTextReasoningPart {
    fn from(value: ReasoningFileOutput) -> Self {
        Self::ReasoningFile(value)
    }
}

/// AI SDK-style step reasoning part returned by `StepResult::reasoning`.
///
/// This intentionally uses the provider-utils prompt-side shape (`providerOptions` and
/// `data`/`mediaType`) rather than the output-side `ReasoningOutput` shape used by the final
/// `GenerateTextResult::reasoning` projection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum GenerateTextStepReasoningPart {
    /// Text reasoning part.
    Reasoning {
        /// Reasoning text.
        text: String,
        /// Provider-specific request options derived from output provider metadata.
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },
    /// File reasoning part.
    ReasoningFile {
        /// Base64 or binary file data.
        data: DataContent,
        /// IANA media type of the file.
        #[serde(rename = "mediaType", alias = "media_type")]
        media_type: String,
        /// Provider-specific request options derived from output provider metadata.
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },
}

impl GenerateTextStepReasoningPart {
    /// Create a step reasoning text part.
    pub fn reasoning(text: impl Into<String>) -> Self {
        Self::Reasoning {
            text: text.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a step reasoning-file part.
    pub fn reasoning_file(data: impl Into<DataContent>, media_type: impl Into<String>) -> Self {
        Self::ReasoningFile {
            data: data.into(),
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Return the AI SDK reasoning part discriminator.
    pub const fn r#type(&self) -> &'static str {
        match self {
            Self::Reasoning { .. } => "reasoning",
            Self::ReasoningFile { .. } => "reasoning-file",
        }
    }

    /// Attach provider-specific options.
    pub fn with_provider_options(mut self, provider_options: ProviderOptionsMap) -> Self {
        match &mut self {
            Self::Reasoning {
                provider_options: options,
                ..
            }
            | Self::ReasoningFile {
                provider_options: options,
                ..
            } => *options = provider_options,
        }
        self
    }
}

fn provider_metadata_to_options(provider_metadata: Option<ProviderMetadata>) -> ProviderOptionsMap {
    let mut provider_options = ProviderOptionsMap::default();
    if let Some(provider_metadata) = provider_metadata {
        for (provider_id, value) in provider_metadata {
            provider_options.insert(provider_id, value);
        }
    }
    provider_options
}

impl From<ReasoningOutput> for GenerateTextStepReasoningPart {
    fn from(value: ReasoningOutput) -> Self {
        Self::Reasoning {
            text: value.text,
            provider_options: provider_metadata_to_options(value.provider_metadata),
        }
    }
}

impl From<ReasoningFileOutput> for GenerateTextStepReasoningPart {
    fn from(value: ReasoningFileOutput) -> Self {
        Self::ReasoningFile {
            data: DataContent::base64(value.file.base64),
            media_type: value.file.media_type,
            provider_options: provider_metadata_to_options(value.provider_metadata),
        }
    }
}

/// AI SDK-style model identity used by step results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextModelInfo {
    /// Provider identifier.
    pub provider: String,
    /// Model identifier.
    pub model_id: String,
}

impl GenerateTextModelInfo {
    /// Create model identity metadata.
    pub fn new(provider: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model_id: model_id.into(),
        }
    }
}

/// AI SDK-style response metadata envelope for text generation results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextResponseMetadata {
    /// Shared language-model response metadata.
    #[serde(flatten)]
    pub metadata: LanguageModelResponseMetadata,
    /// Response messages generated during the call.
    pub messages: Vec<ResponseMessage>,
    /// Raw response body when the provider exposes it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl GenerateTextResponseMetadata {
    /// Create a response metadata envelope.
    pub fn new(metadata: LanguageModelResponseMetadata) -> Self {
        Self {
            metadata,
            messages: Vec::new(),
            body: None,
        }
    }

    /// Attach generated response messages.
    pub fn with_messages(mut self, messages: Vec<ResponseMessage>) -> Self {
        self.messages = messages;
        self
    }

    /// Attach a raw response body.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }
}

/// Passive AI SDK-style result for one `generateText` step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextStepResult<NAME = String, INPUT = JSONValue, ToolOutput = ToolResultOutput> {
    /// Unique identifier for the generation call this step belongs to.
    pub call_id: String,
    /// Zero-based step index.
    pub step_number: u32,
    /// Model identity that produced this step.
    pub model: GenerateTextModelInfo,
    /// Tool context snapshot.
    pub tools_context: Context,
    /// Runtime context snapshot.
    pub runtime_context: Context,
    /// Content generated in this step.
    pub content: Vec<GenerateTextContentPart<NAME, INPUT, ToolOutput>>,
    /// Generated text for this step.
    pub text: String,
    /// Reasoning parts generated in this step.
    pub reasoning: Vec<GenerateTextStepReasoningPart>,
    /// Concatenated reasoning text when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_text: Option<String>,
    /// Files generated in this step.
    pub files: Vec<GeneratedFile>,
    /// Sources used in this step.
    pub sources: Vec<Source>,
    /// Tool calls made in this step.
    pub tool_calls: Vec<ToolCall<NAME, INPUT>>,
    /// Static tool calls made in this step.
    pub static_tool_calls: Vec<ToolCall<NAME, INPUT>>,
    /// Dynamic tool calls made in this step.
    pub dynamic_tool_calls: Vec<ToolCall<NAME, INPUT>>,
    /// Tool results produced in this step.
    pub tool_results: Vec<ToolResult<NAME, INPUT, ToolOutput>>,
    /// Static tool results produced in this step.
    pub static_tool_results: Vec<ToolResult<NAME, INPUT, ToolOutput>>,
    /// Dynamic tool results produced in this step.
    pub dynamic_tool_results: Vec<ToolResult<NAME, INPUT, ToolOutput>>,
    /// Unified finish reason.
    pub finish_reason: FinishReason,
    /// Raw provider finish reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_finish_reason: Option<String>,
    /// Token usage for this step.
    pub usage: LanguageModelUsage,
    /// Provider warnings for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<CallWarning>>,
    /// Request metadata for this step.
    pub request: LanguageModelRequestMetadata,
    /// Response metadata for this step.
    pub response: GenerateTextResponseMetadata,
    /// Provider-specific metadata for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

/// Passive AI SDK-style result envelope for a non-streaming `generateText` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextResult<
    OUTPUT = JSONValue,
    NAME = String,
    INPUT = JSONValue,
    ToolOutput = ToolResultOutput,
> {
    /// Content generated in the last step.
    pub content: Vec<GenerateTextContentPart<NAME, INPUT, ToolOutput>>,
    /// Text generated in the last step.
    pub text: String,
    /// Reasoning generated in the last step.
    pub reasoning: Vec<GenerateTextReasoningPart>,
    /// Concatenated reasoning text when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_text: Option<String>,
    /// Files generated in the last step.
    pub files: Vec<GeneratedFile>,
    /// Sources used in the last step.
    pub sources: Vec<Source>,
    /// Tool calls made in the last step.
    pub tool_calls: Vec<ToolCall<NAME, INPUT>>,
    /// Static tool calls made in the last step.
    pub static_tool_calls: Vec<ToolCall<NAME, INPUT>>,
    /// Dynamic tool calls made in the last step.
    pub dynamic_tool_calls: Vec<ToolCall<NAME, INPUT>>,
    /// Tool results produced in the last step.
    pub tool_results: Vec<ToolResult<NAME, INPUT, ToolOutput>>,
    /// Static tool results produced in the last step.
    pub static_tool_results: Vec<ToolResult<NAME, INPUT, ToolOutput>>,
    /// Dynamic tool results produced in the last step.
    pub dynamic_tool_results: Vec<ToolResult<NAME, INPUT, ToolOutput>>,
    /// Unified finish reason.
    pub finish_reason: FinishReason,
    /// Raw provider finish reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_finish_reason: Option<String>,
    /// Token usage for the last step.
    pub usage: LanguageModelUsage,
    /// Aggregated token usage across all steps.
    pub total_usage: LanguageModelUsage,
    /// Provider warnings for the call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<CallWarning>>,
    /// Request metadata for the call.
    pub request: LanguageModelRequestMetadata,
    /// Response metadata for the call.
    pub response: GenerateTextResponseMetadata,
    /// Provider-specific metadata for the call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Step details.
    pub steps: Vec<GenerateTextStepResult<NAME, INPUT, ToolOutput>>,
    /// Structured output projection.
    pub output: OUTPUT,
}

macro_rules! fixed_text_stream_type_marker {
    ($name:ident, $value:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum $name {
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
                        "expected text stream type marker `{}`, got `{value}`",
                        $value
                    )))
                }
            }
        }
    };
}

fixed_text_stream_type_marker!(TextStreamTextStartPartMarker, "text-start");
fixed_text_stream_type_marker!(TextStreamTextDeltaPartMarker, "text-delta");
fixed_text_stream_type_marker!(TextStreamTextEndPartMarker, "text-end");
fixed_text_stream_type_marker!(TextStreamReasoningStartPartMarker, "reasoning-start");
fixed_text_stream_type_marker!(TextStreamReasoningDeltaPartMarker, "reasoning-delta");
fixed_text_stream_type_marker!(TextStreamReasoningEndPartMarker, "reasoning-end");
fixed_text_stream_type_marker!(TextStreamCustomPartMarker, "custom");
fixed_text_stream_type_marker!(TextStreamToolInputStartPartMarker, "tool-input-start");
fixed_text_stream_type_marker!(TextStreamToolInputDeltaPartMarker, "tool-input-delta");
fixed_text_stream_type_marker!(TextStreamToolInputEndPartMarker, "tool-input-end");
fixed_text_stream_type_marker!(TextStreamFilePartMarker, "file");
fixed_text_stream_type_marker!(TextStreamReasoningFilePartMarker, "reasoning-file");
fixed_text_stream_type_marker!(TextStreamStartStepPartMarker, "start-step");
fixed_text_stream_type_marker!(TextStreamFinishStepPartMarker, "finish-step");
fixed_text_stream_type_marker!(TextStreamStartPartMarker, "start");
fixed_text_stream_type_marker!(TextStreamFinishPartMarker, "finish");
fixed_text_stream_type_marker!(TextStreamAbortPartMarker, "abort");
fixed_text_stream_type_marker!(TextStreamErrorPartMarker, "error");
fixed_text_stream_type_marker!(TextStreamRawPartMarker, "raw");

/// Text block start event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamTextStartPart {
    #[serde(rename = "type", default)]
    marker: TextStreamTextStartPartMarker,
    pub id: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamTextStartPart {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            marker: TextStreamTextStartPartMarker::Marker,
            id: id.into(),
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "text-start"
    }
}

/// Text delta event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamTextDeltaPart {
    #[serde(rename = "type", default)]
    marker: TextStreamTextDeltaPartMarker,
    pub id: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
    pub text: String,
}

impl TextStreamTextDeltaPart {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            marker: TextStreamTextDeltaPartMarker::Marker,
            id: id.into(),
            provider_metadata: None,
            text: text.into(),
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "text-delta"
    }
}

/// Text block end event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamTextEndPart {
    #[serde(rename = "type", default)]
    marker: TextStreamTextEndPartMarker,
    pub id: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamTextEndPart {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            marker: TextStreamTextEndPartMarker::Marker,
            id: id.into(),
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "text-end"
    }
}

/// Reasoning block start event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamReasoningStartPart {
    #[serde(rename = "type", default)]
    marker: TextStreamReasoningStartPartMarker,
    pub id: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamReasoningStartPart {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            marker: TextStreamReasoningStartPartMarker::Marker,
            id: id.into(),
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "reasoning-start"
    }
}

/// Reasoning delta event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamReasoningDeltaPart {
    #[serde(rename = "type", default)]
    marker: TextStreamReasoningDeltaPartMarker,
    pub id: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
    pub text: String,
}

impl TextStreamReasoningDeltaPart {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            marker: TextStreamReasoningDeltaPartMarker::Marker,
            id: id.into(),
            provider_metadata: None,
            text: text.into(),
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "reasoning-delta"
    }
}

/// Reasoning block end event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamReasoningEndPart {
    #[serde(rename = "type", default)]
    marker: TextStreamReasoningEndPartMarker,
    pub id: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamReasoningEndPart {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            marker: TextStreamReasoningEndPartMarker::Marker,
            id: id.into(),
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "reasoning-end"
    }
}

/// Provider custom event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamCustomPart {
    #[serde(rename = "type", default)]
    marker: TextStreamCustomPartMarker,
    pub kind: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamCustomPart {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            marker: TextStreamCustomPartMarker::Marker,
            kind: kind.into(),
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "custom"
    }
}

/// Tool input start event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamToolInputStartPart {
    #[serde(rename = "type", default)]
    marker: TextStreamToolInputStartPartMarker,
    pub id: String,
    #[serde(rename = "toolName", alias = "tool_name")]
    pub tool_name: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl TextStreamToolInputStartPart {
    pub fn new(id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            marker: TextStreamToolInputStartPartMarker::Marker,
            id: id.into(),
            tool_name: tool_name.into(),
            provider_metadata: None,
            provider_executed: None,
            dynamic: None,
            title: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "tool-input-start"
    }
}

/// Tool input delta event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamToolInputDeltaPart {
    #[serde(rename = "type", default)]
    marker: TextStreamToolInputDeltaPartMarker,
    pub id: String,
    pub delta: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamToolInputDeltaPart {
    pub fn new(id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            marker: TextStreamToolInputDeltaPartMarker::Marker,
            id: id.into(),
            delta: delta.into(),
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "tool-input-delta"
    }
}

/// Tool input end event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamToolInputEndPart {
    #[serde(rename = "type", default)]
    marker: TextStreamToolInputEndPartMarker,
    pub id: String,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamToolInputEndPart {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            marker: TextStreamToolInputEndPartMarker::Marker,
            id: id.into(),
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "tool-input-end"
    }
}

/// File event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamFilePart {
    #[serde(rename = "type", default)]
    marker: TextStreamFilePartMarker,
    pub file: GeneratedFile,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamFilePart {
    pub fn new(file: GeneratedFile) -> Self {
        Self {
            marker: TextStreamFilePartMarker::Marker,
            file,
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "file"
    }
}

/// Reasoning file event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamReasoningFilePart {
    #[serde(rename = "type", default)]
    marker: TextStreamReasoningFilePartMarker,
    pub file: GeneratedFile,
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamReasoningFilePart {
    pub fn new(file: GeneratedFile) -> Self {
        Self {
            marker: TextStreamReasoningFilePartMarker::Marker,
            file,
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "reasoning-file"
    }
}

/// AI SDK `TextStreamPart` tool-call alias.
pub type TextStreamToolCallPart<NAME = String, INPUT = JSONValue> = ToolCall<NAME, INPUT>;

/// AI SDK `TextStreamPart` source alias.
pub type TextStreamSourcePart = Source;

/// AI SDK `TextStreamPart` tool-result alias.
pub type TextStreamToolResultPart<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    ToolResult<NAME, INPUT, OUTPUT>;

/// AI SDK `TextStreamPart` tool-error alias.
pub type TextStreamToolErrorPart<NAME = String, INPUT = JSONValue> = ToolError<NAME, INPUT>;

/// AI SDK `TextStreamPart` tool-output-denied alias.
pub type TextStreamToolOutputDeniedPart<NAME = String> = ToolOutputDenied<NAME>;

/// AI SDK `TextStreamPart` tool-approval-request alias.
pub type TextStreamToolApprovalRequestPart<NAME = String, INPUT = JSONValue> =
    ToolApprovalRequestOutput<NAME, INPUT>;

/// AI SDK `TextStreamPart` tool-approval-response alias.
pub type TextStreamToolApprovalResponsePart<NAME = String, INPUT = JSONValue> =
    ToolApprovalResponseOutput<NAME, INPUT>;

/// Step start event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamStartStepPart {
    #[serde(rename = "type", default)]
    marker: TextStreamStartStepPartMarker,
    pub request: LanguageModelRequestMetadata,
    pub warnings: Vec<CallWarning>,
}

impl TextStreamStartStepPart {
    pub fn new(request: LanguageModelRequestMetadata, warnings: Vec<CallWarning>) -> Self {
        Self {
            marker: TextStreamStartStepPartMarker::Marker,
            request,
            warnings,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "start-step"
    }
}

/// Step finish event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TextStreamFinishStepPart {
    #[serde(rename = "type", default)]
    marker: TextStreamFinishStepPartMarker,
    pub response: LanguageModelResponseMetadata,
    pub usage: LanguageModelUsage,
    pub finish_reason: FinishReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextStreamFinishStepPart {
    pub fn new(
        response: LanguageModelResponseMetadata,
        usage: LanguageModelUsage,
        finish_reason: FinishReason,
    ) -> Self {
        Self {
            marker: TextStreamFinishStepPartMarker::Marker,
            response,
            usage,
            finish_reason,
            raw_finish_reason: None,
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "finish-step"
    }
}

/// Stream start event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamStartPart {
    #[serde(rename = "type", default)]
    marker: TextStreamStartPartMarker,
}

impl TextStreamStartPart {
    pub fn new() -> Self {
        Self {
            marker: TextStreamStartPartMarker::Marker,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "start"
    }
}

impl Default for TextStreamStartPart {
    fn default() -> Self {
        Self::new()
    }
}

/// Stream finish event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TextStreamFinishPart {
    #[serde(rename = "type", default)]
    marker: TextStreamFinishPartMarker,
    pub finish_reason: FinishReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_finish_reason: Option<String>,
    pub total_usage: LanguageModelUsage,
}

impl TextStreamFinishPart {
    pub fn new(finish_reason: FinishReason, total_usage: LanguageModelUsage) -> Self {
        Self {
            marker: TextStreamFinishPartMarker::Marker,
            finish_reason,
            raw_finish_reason: None,
            total_usage,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "finish"
    }
}

/// Abort event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextStreamAbortPart {
    #[serde(rename = "type", default)]
    marker: TextStreamAbortPartMarker,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl TextStreamAbortPart {
    pub fn new() -> Self {
        Self {
            marker: TextStreamAbortPartMarker::Marker,
            reason: None,
        }
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    pub const fn r#type(&self) -> &'static str {
        "abort"
    }
}

impl Default for TextStreamAbortPart {
    fn default() -> Self {
        Self::new()
    }
}

/// Error event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamErrorPart {
    #[serde(rename = "type", default)]
    marker: TextStreamErrorPartMarker,
    pub error: JSONValue,
}

impl TextStreamErrorPart {
    pub fn new(error: impl Into<JSONValue>) -> Self {
        Self {
            marker: TextStreamErrorPartMarker::Marker,
            error: error.into(),
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "error"
    }
}

/// Raw event from AI SDK `TextStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextStreamRawPart {
    #[serde(rename = "type", default)]
    marker: TextStreamRawPartMarker,
    #[serde(rename = "rawValue", alias = "raw_value")]
    pub raw_value: JSONValue,
}

impl TextStreamRawPart {
    pub fn new(raw_value: impl Into<JSONValue>) -> Self {
        Self {
            marker: TextStreamRawPartMarker::Marker,
            raw_value: raw_value.into(),
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "raw"
    }
}

/// AI SDK `TextStreamPart` union from `generate-text/stream-text-result.ts`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TextStreamPart<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    TextStart(TextStreamTextStartPart),
    TextEnd(TextStreamTextEndPart),
    TextDelta(TextStreamTextDeltaPart),
    ReasoningStart(TextStreamReasoningStartPart),
    ReasoningEnd(TextStreamReasoningEndPart),
    ReasoningDelta(TextStreamReasoningDeltaPart),
    Custom(TextStreamCustomPart),
    ToolInputStart(TextStreamToolInputStartPart),
    ToolInputEnd(TextStreamToolInputEndPart),
    ToolInputDelta(TextStreamToolInputDeltaPart),
    Source(Source),
    File(TextStreamFilePart),
    ReasoningFile(TextStreamReasoningFilePart),
    ToolCall(ToolCall<NAME, INPUT>),
    ToolResult(ToolResult<NAME, INPUT, OUTPUT>),
    ToolError(ToolError<NAME, INPUT>),
    ToolOutputDenied(ToolOutputDenied<NAME>),
    ToolApprovalRequest(ToolApprovalRequestOutput<NAME, INPUT>),
    ToolApprovalResponse(ToolApprovalResponseOutput<NAME, INPUT>),
    StartStep(TextStreamStartStepPart),
    FinishStep(TextStreamFinishStepPart),
    Start(TextStreamStartPart),
    Finish(TextStreamFinishPart),
    Abort(TextStreamAbortPart),
    Error(TextStreamErrorPart),
    Raw(TextStreamRawPart),
}

impl<NAME, INPUT, OUTPUT> TextStreamPart<NAME, INPUT, OUTPUT> {
    /// Return the AI SDK `TextStreamPart` discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::TextStart(part) => part.r#type(),
            Self::TextEnd(part) => part.r#type(),
            Self::TextDelta(part) => part.r#type(),
            Self::ReasoningStart(part) => part.r#type(),
            Self::ReasoningEnd(part) => part.r#type(),
            Self::ReasoningDelta(part) => part.r#type(),
            Self::Custom(part) => part.r#type(),
            Self::ToolInputStart(part) => part.r#type(),
            Self::ToolInputEnd(part) => part.r#type(),
            Self::ToolInputDelta(part) => part.r#type(),
            Self::Source(part) => part.r#type(),
            Self::File(part) => part.r#type(),
            Self::ReasoningFile(part) => part.r#type(),
            Self::ToolCall(part) => part.r#type(),
            Self::ToolResult(part) => part.r#type(),
            Self::ToolError(part) => part.r#type(),
            Self::ToolOutputDenied(part) => part.r#type(),
            Self::ToolApprovalRequest(part) => part.r#type(),
            Self::ToolApprovalResponse(part) => part.r#type(),
            Self::StartStep(part) => part.r#type(),
            Self::FinishStep(part) => part.r#type(),
            Self::Start(part) => part.r#type(),
            Self::Finish(part) => part.r#type(),
            Self::Abort(part) => part.r#type(),
            Self::Error(part) => part.r#type(),
            Self::Raw(part) => part.r#type(),
        }
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamTextStartPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamTextStartPart) -> Self {
        Self::TextStart(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamTextDeltaPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamTextDeltaPart) -> Self {
        Self::TextDelta(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamTextEndPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamTextEndPart) -> Self {
        Self::TextEnd(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamReasoningStartPart>
    for TextStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamReasoningStartPart) -> Self {
        Self::ReasoningStart(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamReasoningDeltaPart>
    for TextStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamReasoningDeltaPart) -> Self {
        Self::ReasoningDelta(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamReasoningEndPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamReasoningEndPart) -> Self {
        Self::ReasoningEnd(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamCustomPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamCustomPart) -> Self {
        Self::Custom(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamToolInputStartPart>
    for TextStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamToolInputStartPart) -> Self {
        Self::ToolInputStart(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamToolInputDeltaPart>
    for TextStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamToolInputDeltaPart) -> Self {
        Self::ToolInputDelta(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamToolInputEndPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamToolInputEndPart) -> Self {
        Self::ToolInputEnd(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<Source> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: Source) -> Self {
        Self::Source(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamFilePart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamFilePart) -> Self {
        Self::File(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamReasoningFilePart>
    for TextStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamReasoningFilePart) -> Self {
        Self::ReasoningFile(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolCall<NAME, INPUT>> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: ToolCall<NAME, INPUT>) -> Self {
        Self::ToolCall(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolResult<NAME, INPUT, OUTPUT>>
    for TextStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolResult<NAME, INPUT, OUTPUT>) -> Self {
        Self::ToolResult(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolError<NAME, INPUT>> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: ToolError<NAME, INPUT>) -> Self {
        Self::ToolError(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolOutputDenied<NAME>> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: ToolOutputDenied<NAME>) -> Self {
        Self::ToolOutputDenied(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolApprovalRequestOutput<NAME, INPUT>>
    for TextStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolApprovalRequestOutput<NAME, INPUT>) -> Self {
        Self::ToolApprovalRequest(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolApprovalResponseOutput<NAME, INPUT>>
    for TextStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolApprovalResponseOutput<NAME, INPUT>) -> Self {
        Self::ToolApprovalResponse(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamStartStepPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamStartStepPart) -> Self {
        Self::StartStep(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamFinishStepPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamFinishStepPart) -> Self {
        Self::FinishStep(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamStartPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamStartPart) -> Self {
        Self::Start(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamFinishPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamFinishPart) -> Self {
        Self::Finish(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamAbortPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamAbortPart) -> Self {
        Self::Abort(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamErrorPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamErrorPart) -> Self {
        Self::Error(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamRawPart> for TextStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamRawPart) -> Self {
        Self::Raw(value)
    }
}

/// Passive representation of AI SDK `StopCondition`.
///
/// The TypeScript surface accepts predicates/functions. Rust keeps this as JSON so
/// callers can record symbolic stop-condition configuration without pretending those
/// callbacks are executable across the spec boundary.
pub type StopCondition = JSONValue;

/// Common model information used across AI SDK callback events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CallbackModelInfo {
    /// Provider identifier.
    pub provider: String,
    /// Model identifier.
    pub model_id: String,
}

impl CallbackModelInfo {
    /// Create callback model information.
    pub fn new(provider: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model_id: model_id.into(),
        }
    }
}

/// Event passed to AI SDK `generateText` / `streamText` `onStart` callbacks.
///
/// This is a passive data view of `generate-text/core-events.ts`; no runtime callback
/// execution is implied by this type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextStartEvent<OUTPUT = JSONValue> {
    /// Unique generation call identifier.
    pub call_id: String,
    /// Operation identifier such as `ai.generateText` or `ai.streamText`.
    pub operation_id: String,
    /// Provider and model identity flattened like AI SDK callback payloads.
    #[serde(flatten)]
    pub model: CallbackModelInfo,
    /// Tools available for this generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool choice strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Tool names enabled for this generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_tools: Option<Vec<String>>,
    /// Maximum retry count.
    pub max_retries: u32,
    /// Timeout configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<TimeoutConfiguration>,
    /// Additional request headers. `None` values serialize as JSON null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Provider-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
    /// Symbolic stop-condition configuration.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_when: Vec<StopCondition>,
    /// Structured output specification or metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OUTPUT>,
    /// Tool context snapshot.
    pub tools_context: Context,
    /// Runtime context snapshot.
    pub runtime_context: Context,
    /// Model call options flattened into the callback payload.
    #[serde(flatten)]
    pub call_options: LanguageModelCallOptions,
    /// Standardized prompt flattened into the callback payload.
    #[serde(flatten)]
    pub prompt: StandardizedPrompt,
}

/// Event passed to AI SDK `generateText` / `streamText` `onStepStart` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextStepStartEvent<
    OUTPUT = JSONValue,
    NAME = String,
    INPUT = JSONValue,
    ToolOutputValue = ToolResultOutput,
> {
    /// Unique generation call identifier.
    pub call_id: String,
    /// Provider and model identity flattened like AI SDK callback payloads.
    #[serde(flatten)]
    pub model: CallbackModelInfo,
    /// Zero-based step index.
    pub step_number: u32,
    /// Tools available for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool choice strategy for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Tool names enabled for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_tools: Option<Vec<String>>,
    /// Previous step results.
    pub steps: Vec<GenerateTextStepResult<NAME, INPUT, ToolOutputValue>>,
    /// Provider-specific options for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
    /// Structured output specification or metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OUTPUT>,
    /// Runtime context snapshot.
    pub runtime_context: Context,
    /// Tool context snapshot.
    pub tools_context: Context,
    /// Standardized prompt flattened into the callback payload.
    #[serde(flatten)]
    pub prompt: StandardizedPrompt,
}

/// Event passed to AI SDK `onStepFinish` callbacks.
pub type GenerateTextStepEndEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    GenerateTextStepResult<NAME, INPUT, OUTPUT>;

/// Event passed to AI SDK `onFinish` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateTextEndEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// Final step result flattened into the callback payload.
    #[serde(flatten)]
    pub step: GenerateTextStepResult<NAME, INPUT, OUTPUT>,
    /// All step results.
    pub steps: Vec<GenerateTextStepResult<NAME, INPUT, OUTPUT>>,
    /// Aggregated token usage across all steps.
    pub total_usage: LanguageModelUsage,
}

/// Stream lifecycle marker type used by `StreamTextChunkEvent`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StreamTextLifecycleChunkType {
    /// First streamed chunk marker.
    #[serde(rename = "ai.stream.firstChunk")]
    FirstChunk,
    /// Stream finish marker.
    #[serde(rename = "ai.stream.finish")]
    Finish,
}

/// Stream lifecycle marker emitted through AI SDK `onChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StreamTextLifecycleChunk {
    /// Lifecycle marker discriminator.
    #[serde(rename = "type")]
    pub kind: StreamTextLifecycleChunkType,
    /// Unique generation call identifier.
    pub call_id: String,
    /// Zero-based step index.
    pub step_number: u32,
    /// Optional telemetry attributes.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, JSONValue>,
}

impl StreamTextLifecycleChunk {
    /// Create an `ai.stream.firstChunk` lifecycle marker.
    pub fn first_chunk(call_id: impl Into<String>, step_number: u32) -> Self {
        Self {
            kind: StreamTextLifecycleChunkType::FirstChunk,
            call_id: call_id.into(),
            step_number,
            attributes: HashMap::new(),
        }
    }

    /// Create an `ai.stream.finish` lifecycle marker.
    pub fn finish(call_id: impl Into<String>, step_number: u32) -> Self {
        Self {
            kind: StreamTextLifecycleChunkType::Finish,
            call_id: call_id.into(),
            step_number,
            attributes: HashMap::new(),
        }
    }

    /// Attach one lifecycle attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<JSONValue>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Return the lifecycle discriminator.
    pub const fn r#type(&self) -> &'static str {
        match self.kind {
            StreamTextLifecycleChunkType::FirstChunk => "ai.stream.firstChunk",
            StreamTextLifecycleChunkType::Finish => "ai.stream.finish",
        }
    }
}

/// Chunk payload passed to AI SDK `streamText` `onChunk` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum StreamTextChunk<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// Regular `TextStreamPart`.
    Part(TextStreamPart<NAME, INPUT, OUTPUT>),
    /// Stream lifecycle marker.
    Lifecycle(StreamTextLifecycleChunk),
}

impl<NAME, INPUT, OUTPUT> StreamTextChunk<NAME, INPUT, OUTPUT> {
    /// Return the chunk discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Part(part) => part.r#type(),
            Self::Lifecycle(part) => part.r#type(),
        }
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamPart<NAME, INPUT, OUTPUT>>
    for StreamTextChunk<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamPart<NAME, INPUT, OUTPUT>) -> Self {
        Self::Part(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<StreamTextLifecycleChunk> for StreamTextChunk<NAME, INPUT, OUTPUT> {
    fn from(value: StreamTextLifecycleChunk) -> Self {
        Self::Lifecycle(value)
    }
}

/// Event passed to AI SDK `streamText` `onChunk` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamTextChunkEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// Stream chunk or lifecycle marker.
    pub chunk: StreamTextChunk<NAME, INPUT, OUTPUT>,
}

impl<NAME, INPUT, OUTPUT> StreamTextChunkEvent<NAME, INPUT, OUTPUT> {
    /// Create a stream chunk event.
    pub fn new(chunk: impl Into<StreamTextChunk<NAME, INPUT, OUTPUT>>) -> Self {
        Self {
            chunk: chunk.into(),
        }
    }
}

/// Event passed to AI SDK tool execution start callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionStartEvent<NAME = String, INPUT = JSONValue> {
    /// Unique generation call identifier.
    pub call_id: String,
    /// Messages sent to the model before the assistant tool-call response.
    pub messages: Vec<ModelMessage>,
    /// Tool call that is about to execute.
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Validated tool context when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_context: Option<JSONValue>,
}

/// Event passed to AI SDK tool execution end callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionEndEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// Unique generation call identifier.
    pub call_id: String,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// Messages sent to the model before the assistant tool-call response.
    pub messages: Vec<ModelMessage>,
    /// Tool call that finished executing.
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Validated tool context when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_context: Option<JSONValue>,
    /// Successful or failed tool output.
    pub tool_output: ToolOutput<NAME, INPUT, OUTPUT>,
}

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use GenerateTextStartEvent instead.")]
pub type OnStartEvent<OUTPUT = JSONValue> = GenerateTextStartEvent<OUTPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use GenerateTextStepStartEvent instead.")]
pub type OnStepStartEvent<
    OUTPUT = JSONValue,
    NAME = String,
    INPUT = JSONValue,
    ToolOutputValue = ToolResultOutput,
> = GenerateTextStepStartEvent<OUTPUT, NAME, INPUT, ToolOutputValue>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use StreamTextChunkEvent instead.")]
pub type OnChunkEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    StreamTextChunkEvent<NAME, INPUT, OUTPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use GenerateTextStepEndEvent instead.")]
pub type OnStepFinishEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    GenerateTextStepEndEvent<NAME, INPUT, OUTPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use GenerateTextEndEvent instead.")]
pub type OnFinishEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    GenerateTextEndEvent<NAME, INPUT, OUTPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use ToolExecutionStartEvent instead.")]
pub type OnToolCallStartEvent<NAME = String, INPUT = JSONValue> =
    ToolExecutionStartEvent<NAME, INPUT>;

/// Deprecated AI SDK callback event aliases kept for source compatibility.
#[deprecated(note = "Use ToolExecutionEndEvent instead.")]
pub type OnToolCallFinishEvent<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    ToolExecutionEndEvent<NAME, INPUT, OUTPUT>;

/// A cloneable cancellation handle for request-scoped abort semantics.
#[derive(Clone, Debug, Default)]
pub struct CancelHandle {
    token: CancellationToken,
}

impl CancelHandle {
    /// Create a new cancellation handle.
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Future that resolves when cancellation is requested.
    pub fn cancelled(&self) -> WaitForCancellationFuture<'_> {
        self.token.cancelled()
    }

    /// Clone the underlying cancellation token for integrations that need it directly.
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

/// Structured timeout configuration details for request-facing controls.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TimeoutConfigurationSettings {
    /// Total timeout in milliseconds.
    #[serde(rename = "totalMs", skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<u64>,
    /// Per-step timeout in milliseconds.
    #[serde(rename = "stepMs", skip_serializing_if = "Option::is_none")]
    pub step_ms: Option<u64>,
    /// Timeout between stream chunks in milliseconds.
    #[serde(rename = "chunkMs", skip_serializing_if = "Option::is_none")]
    pub chunk_ms: Option<u64>,
    /// Default timeout for all tools in milliseconds.
    #[serde(rename = "toolMs", skip_serializing_if = "Option::is_none")]
    pub tool_ms: Option<u64>,
    /// Per-tool timeout overrides keyed by the AI SDK-style `{toolName}Ms` suffix form.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tools: HashMap<String, u64>,
}

impl TimeoutConfigurationSettings {
    /// Create empty timeout settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set total timeout in milliseconds.
    pub const fn with_total_ms(mut self, total_ms: u64) -> Self {
        self.total_ms = Some(total_ms);
        self
    }

    /// Set per-step timeout in milliseconds.
    pub const fn with_step_ms(mut self, step_ms: u64) -> Self {
        self.step_ms = Some(step_ms);
        self
    }

    /// Set between-chunk timeout in milliseconds.
    pub const fn with_chunk_ms(mut self, chunk_ms: u64) -> Self {
        self.chunk_ms = Some(chunk_ms);
        self
    }

    /// Set default tool timeout in milliseconds.
    pub const fn with_tool_ms(mut self, tool_ms: u64) -> Self {
        self.tool_ms = Some(tool_ms);
        self
    }

    /// Set a tool-specific timeout in milliseconds.
    pub fn with_tool_timeout_ms(mut self, tool_name: impl AsRef<str>, timeout_ms: u64) -> Self {
        self.tools
            .insert(format!("{}Ms", tool_name.as_ref()), timeout_ms);
        self
    }

    /// Resolve the tool timeout for the given tool name.
    pub fn tool_timeout_ms(&self, tool_name: &str) -> Option<u64> {
        self.tools
            .get(&format!("{tool_name}Ms"))
            .copied()
            .or(self.tool_ms)
    }
}

/// AI SDK-style timeout configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TimeoutConfiguration {
    /// A single total timeout in milliseconds.
    Millis(u64),
    /// Structured timeout configuration.
    Settings(TimeoutConfigurationSettings),
}

impl TimeoutConfiguration {
    /// Create a total-timeout configuration from milliseconds.
    pub const fn from_millis(total_ms: u64) -> Self {
        Self::Millis(total_ms)
    }

    /// Create a structured timeout configuration.
    pub fn settings(settings: TimeoutConfigurationSettings) -> Self {
        Self::Settings(settings)
    }

    /// Total timeout in milliseconds.
    pub const fn total_timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Millis(total_ms) => Some(*total_ms),
            Self::Settings(settings) => settings.total_ms,
        }
    }

    /// Per-step timeout in milliseconds.
    pub const fn step_timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Millis(_) => None,
            Self::Settings(settings) => settings.step_ms,
        }
    }

    /// Per-chunk timeout in milliseconds.
    pub const fn chunk_timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Millis(_) => None,
            Self::Settings(settings) => settings.chunk_ms,
        }
    }

    /// Per-tool timeout in milliseconds.
    pub fn tool_timeout_ms(&self, tool_name: &str) -> Option<u64> {
        match self {
            Self::Millis(_) => None,
            Self::Settings(settings) => settings.tool_timeout_ms(tool_name),
        }
    }

    /// Total timeout as a `Duration`.
    pub fn total_timeout(&self) -> Option<Duration> {
        self.total_timeout_ms().map(Duration::from_millis)
    }

    /// Step timeout as a `Duration`.
    pub fn step_timeout(&self) -> Option<Duration> {
        self.step_timeout_ms().map(Duration::from_millis)
    }

    /// Chunk timeout as a `Duration`.
    pub fn chunk_timeout(&self) -> Option<Duration> {
        self.chunk_timeout_ms().map(Duration::from_millis)
    }
}

/// AI SDK-style helper for extracting the total timeout in milliseconds.
pub const fn get_total_timeout_ms(timeout: Option<&TimeoutConfiguration>) -> Option<u64> {
    match timeout {
        Some(timeout) => timeout.total_timeout_ms(),
        None => None,
    }
}

/// AI SDK-style helper for extracting the step timeout in milliseconds.
pub const fn get_step_timeout_ms(timeout: Option<&TimeoutConfiguration>) -> Option<u64> {
    match timeout {
        Some(timeout) => timeout.step_timeout_ms(),
        None => None,
    }
}

/// AI SDK-style helper for extracting the chunk timeout in milliseconds.
pub const fn get_chunk_timeout_ms(timeout: Option<&TimeoutConfiguration>) -> Option<u64> {
    match timeout {
        Some(timeout) => timeout.chunk_timeout_ms(),
        None => None,
    }
}

/// AI SDK-style helper for extracting one tool timeout in milliseconds.
pub fn get_tool_timeout_ms(timeout: Option<&TimeoutConfiguration>, tool_name: &str) -> Option<u64> {
    timeout.and_then(|timeout| timeout.tool_timeout_ms(tool_name))
}

/// AI SDK-style request-facing transport controls.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// Maximum number of retries. `0` disables retries.
    pub max_retries: Option<u32>,
    /// Request-scoped abort signal.
    pub abort_signal: Option<CancelHandle>,
    /// Additional HTTP headers. `None` values are filtered when materialized.
    pub headers: HashMap<String, Option<String>>,
    /// Timeout configuration.
    pub timeout: Option<TimeoutConfiguration>,
}

impl RequestOptions {
    /// Create empty request options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max retries.
    pub const fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Set the abort signal.
    pub fn with_abort_signal(mut self, abort_signal: CancelHandle) -> Self {
        self.abort_signal = Some(abort_signal);
        self
    }

    /// Set a request header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), Some(value.into()));
        self
    }

    /// Mark a header as intentionally omitted.
    pub fn without_header(mut self, key: impl Into<String>) -> Self {
        self.headers.insert(key.into(), None);
        self
    }

    /// Set timeout configuration.
    pub fn with_timeout(mut self, timeout: TimeoutConfiguration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Materialize only the headers with concrete values.
    pub fn effective_headers(&self) -> HashMap<String, String> {
        self.headers
            .iter()
            .filter_map(|(key, value)| value.clone().map(|value| (key.clone(), value)))
            .collect()
    }

    /// Total timeout as a `Duration`.
    pub fn total_timeout(&self) -> Option<Duration> {
        self.timeout
            .as_ref()
            .and_then(TimeoutConfiguration::total_timeout)
    }

    /// Step timeout as a `Duration`.
    pub fn step_timeout(&self) -> Option<Duration> {
        self.timeout
            .as_ref()
            .and_then(TimeoutConfiguration::step_timeout)
    }

    /// Chunk timeout as a `Duration`.
    pub fn chunk_timeout(&self) -> Option<Duration> {
        self.timeout
            .as_ref()
            .and_then(TimeoutConfiguration::chunk_timeout)
    }

    /// Per-tool timeout in milliseconds.
    pub fn tool_timeout_ms(&self, tool_name: &str) -> Option<u64> {
        self.timeout
            .as_ref()
            .and_then(|timeout| timeout.tool_timeout_ms(tool_name))
    }

    /// Convert retries into total attempts, where `0` retries means `1` attempt.
    pub fn max_attempts(&self) -> Option<u32> {
        self.max_retries.map(|retries| retries.saturating_add(1))
    }
}

/// AI SDK-style reasoning level for language-model call options.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LanguageModelReasoning {
    /// Use the provider's default reasoning level.
    ProviderDefault,
    /// Disable reasoning when supported.
    None,
    /// Minimal reasoning.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort.
    #[serde(rename = "xhigh")]
    XHigh,
}

/// AI SDK-style model-facing generation controls.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LanguageModelCallOptions {
    /// Maximum output tokens requested by the caller.
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling.
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<f64>,
    /// Presence penalty.
    #[serde(rename = "presencePenalty", skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    #[serde(rename = "frequencyPenalty", skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Stop sequences.
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Deterministic seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Cross-provider reasoning level.
    ///
    /// Siumai does not yet have a stable cross-provider request lane for this field, so
    /// `From<CommonParams>` leaves it empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<LanguageModelReasoning>,
}

impl From<super::CommonParams> for LanguageModelCallOptions {
    fn from(value: super::CommonParams) -> Self {
        Self::from(&value)
    }
}

impl From<&super::CommonParams> for LanguageModelCallOptions {
    fn from(value: &super::CommonParams) -> Self {
        Self {
            max_output_tokens: value.max_completion_tokens.or(value.max_tokens),
            temperature: value.temperature,
            top_p: value.top_p,
            top_k: value.top_k,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            stop_sequences: value.stop_sequences.clone(),
            seed: value.seed,
            reasoning: None,
        }
    }
}

/// Deprecated AI SDK-style combined call settings view.
///
/// Prefer using `LanguageModelCallOptions` together with `RequestOptions`.
#[deprecated(note = "Use `LanguageModelCallOptions` together with `RequestOptions` instead.")]
#[derive(Debug, Clone, Default)]
pub struct CallSettings {
    /// Maximum output tokens requested by the caller.
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    pub top_p: Option<f64>,
    /// Top-k sampling.
    pub top_k: Option<f64>,
    /// Presence penalty.
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Deterministic seed.
    pub seed: Option<u64>,
    /// Cross-provider reasoning level.
    pub reasoning: Option<LanguageModelReasoning>,
    /// Maximum number of retries. `0` disables retries.
    pub max_retries: Option<u32>,
    /// Request-scoped abort signal.
    pub abort_signal: Option<CancelHandle>,
    /// Additional HTTP headers. `None` values are filtered when materialized.
    pub headers: HashMap<String, Option<String>>,
}

#[allow(deprecated)]
impl CallSettings {
    /// Create empty call settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max output tokens.
    pub const fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    /// Set temperature.
    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set nucleus sampling.
    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k sampling.
    pub const fn with_top_k(mut self, top_k: f64) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set presence penalty.
    pub const fn with_presence_penalty(mut self, presence_penalty: f64) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Set frequency penalty.
    pub const fn with_frequency_penalty(mut self, frequency_penalty: f64) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Set stop sequences.
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    /// Set deterministic seed.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set reasoning level.
    pub fn with_reasoning(mut self, reasoning: LanguageModelReasoning) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set max retries.
    pub const fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Set the abort signal.
    pub fn with_abort_signal(mut self, abort_signal: CancelHandle) -> Self {
        self.abort_signal = Some(abort_signal);
        self
    }

    /// Set a request header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), Some(value.into()));
        self
    }

    /// Mark a header as intentionally omitted.
    pub fn without_header(mut self, key: impl Into<String>) -> Self {
        self.headers.insert(key.into(), None);
        self
    }

    /// Project onto `LanguageModelCallOptions`.
    pub fn language_model_call_options(&self) -> LanguageModelCallOptions {
        self.into()
    }

    /// Project onto `RequestOptions`.
    pub fn request_options(&self) -> RequestOptions {
        self.into()
    }

    /// Materialize only the headers with concrete values.
    pub fn effective_headers(&self) -> HashMap<String, String> {
        self.headers
            .iter()
            .filter_map(|(key, value)| value.clone().map(|value| (key.clone(), value)))
            .collect()
    }

    /// Convert retries into total attempts, where `0` retries means `1` attempt.
    pub fn max_attempts(&self) -> Option<u32> {
        self.max_retries.map(|retries| retries.saturating_add(1))
    }
}

#[allow(deprecated)]
impl From<LanguageModelCallOptions> for CallSettings {
    fn from(value: LanguageModelCallOptions) -> Self {
        Self {
            max_output_tokens: value.max_output_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            top_k: value.top_k,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            stop_sequences: value.stop_sequences,
            seed: value.seed,
            reasoning: value.reasoning,
            max_retries: None,
            abort_signal: None,
            headers: HashMap::new(),
        }
    }
}

#[allow(deprecated)]
impl From<RequestOptions> for CallSettings {
    fn from(value: RequestOptions) -> Self {
        Self {
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop_sequences: None,
            seed: None,
            reasoning: None,
            max_retries: value.max_retries,
            abort_signal: value.abort_signal,
            headers: value.headers,
        }
    }
}

#[allow(deprecated)]
impl From<&super::CommonParams> for CallSettings {
    fn from(value: &super::CommonParams) -> Self {
        LanguageModelCallOptions::from(value).into()
    }
}

#[allow(deprecated)]
impl From<super::CommonParams> for CallSettings {
    fn from(value: super::CommonParams) -> Self {
        Self::from(&value)
    }
}

#[allow(deprecated)]
impl From<&CallSettings> for LanguageModelCallOptions {
    fn from(value: &CallSettings) -> Self {
        Self {
            max_output_tokens: value.max_output_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            top_k: value.top_k,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            stop_sequences: value.stop_sequences.clone(),
            seed: value.seed,
            reasoning: value.reasoning.clone(),
        }
    }
}

#[allow(deprecated)]
impl From<CallSettings> for LanguageModelCallOptions {
    fn from(value: CallSettings) -> Self {
        Self::from(&value)
    }
}

#[allow(deprecated)]
impl From<&CallSettings> for RequestOptions {
    fn from(value: &CallSettings) -> Self {
        RequestOptions {
            max_retries: value.max_retries,
            abort_signal: value.abort_signal.clone(),
            headers: value.headers.clone(),
            timeout: None,
        }
    }
}

#[allow(deprecated)]
impl From<CallSettings> for RequestOptions {
    fn from(value: CallSettings) -> Self {
        Self::from(&value)
    }
}

/// AI SDK-style language-model input token detail shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LanguageModelInputTokenDetails {
    /// Non-cached input tokens.
    #[serde(rename = "noCacheTokens", skip_serializing_if = "Option::is_none")]
    pub no_cache_tokens: Option<u32>,
    /// Cached input tokens read.
    #[serde(rename = "cacheReadTokens", skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    /// Cached input tokens written.
    #[serde(rename = "cacheWriteTokens", skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
}

impl LanguageModelInputTokenDetails {
    fn is_empty(&self) -> bool {
        self.no_cache_tokens.is_none()
            && self.cache_read_tokens.is_none()
            && self.cache_write_tokens.is_none()
    }
}

/// AI SDK-style language-model output token detail shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LanguageModelOutputTokenDetails {
    /// Text/output-visible tokens.
    #[serde(rename = "textTokens", skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u32>,
    /// Reasoning tokens.
    #[serde(rename = "reasoningTokens", skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

impl LanguageModelOutputTokenDetails {
    fn is_empty(&self) -> bool {
        self.text_tokens.is_none() && self.reasoning_tokens.is_none()
    }
}

/// AI SDK-style language-model usage shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LanguageModelUsage {
    /// Total input tokens.
    #[serde(rename = "inputTokens", skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    /// Detailed input-token breakdown.
    #[serde(
        rename = "inputTokenDetails",
        default,
        skip_serializing_if = "LanguageModelInputTokenDetails::is_empty"
    )]
    pub input_token_details: LanguageModelInputTokenDetails,
    /// Total output tokens.
    #[serde(rename = "outputTokens", skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    /// Detailed output-token breakdown.
    #[serde(
        rename = "outputTokenDetails",
        default,
        skip_serializing_if = "LanguageModelOutputTokenDetails::is_empty"
    )]
    pub output_token_details: LanguageModelOutputTokenDetails,
    /// Total tokens across input and output.
    #[serde(rename = "totalTokens", skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    /// Deprecated reasoning-token compatibility alias.
    #[serde(rename = "reasoningTokens", skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    /// Deprecated cached-input-token compatibility alias.
    #[serde(rename = "cachedInputTokens", skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u32>,
    /// Raw provider usage payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Map<String, JSONValue>>,
}

impl LanguageModelUsage {
    /// Merge another language-model usage payload into this one.
    pub fn merge(&mut self, other: &Self) {
        self.input_tokens = add_optional_u32(self.input_tokens, other.input_tokens);
        self.input_token_details.no_cache_tokens = add_optional_u32(
            self.input_token_details.no_cache_tokens,
            other.input_token_details.no_cache_tokens,
        );
        self.input_token_details.cache_read_tokens = add_optional_u32(
            self.input_token_details.cache_read_tokens,
            other.input_token_details.cache_read_tokens,
        );
        self.input_token_details.cache_write_tokens = add_optional_u32(
            self.input_token_details.cache_write_tokens,
            other.input_token_details.cache_write_tokens,
        );
        self.output_tokens = add_optional_u32(self.output_tokens, other.output_tokens);
        self.output_token_details.text_tokens = add_optional_u32(
            self.output_token_details.text_tokens,
            other.output_token_details.text_tokens,
        );
        self.output_token_details.reasoning_tokens = add_optional_u32(
            self.output_token_details.reasoning_tokens,
            other.output_token_details.reasoning_tokens,
        );
        self.total_tokens = add_optional_u32(self.total_tokens, other.total_tokens);
        self.reasoning_tokens = add_optional_u32(self.reasoning_tokens, other.reasoning_tokens);
        self.cached_input_tokens =
            add_optional_u32(self.cached_input_tokens, other.cached_input_tokens);
        self.raw = None;
    }
}

/// AI SDK-style helper for creating an empty language-model usage payload.
pub fn create_null_language_model_usage() -> LanguageModelUsage {
    LanguageModelUsage::default()
}

/// AI SDK-style helper for adding two language-model usage payloads.
///
/// Raw provider usage is intentionally dropped on the aggregated result, matching the
/// upstream helper's normalized aggregate shape.
pub fn add_language_model_usage(
    usage1: &LanguageModelUsage,
    usage2: &LanguageModelUsage,
) -> LanguageModelUsage {
    let mut merged = usage1.clone();
    merged.merge(usage2);
    merged
}

impl From<Usage> for LanguageModelUsage {
    fn from(value: Usage) -> Self {
        Self::from(&value)
    }
}

impl From<&Usage> for LanguageModelUsage {
    fn from(value: &Usage) -> Self {
        let input_tokens = value.normalized_input_tokens();
        let output_tokens = value.normalized_output_tokens();

        Self {
            input_tokens: input_tokens.total,
            input_token_details: LanguageModelInputTokenDetails {
                no_cache_tokens: input_tokens.no_cache,
                cache_read_tokens: input_tokens.cache_read,
                cache_write_tokens: input_tokens.cache_write,
            },
            output_tokens: output_tokens.total,
            output_token_details: LanguageModelOutputTokenDetails {
                text_tokens: output_tokens.text,
                reasoning_tokens: output_tokens.reasoning,
            },
            total_tokens: add_optional_u32(input_tokens.total, output_tokens.total),
            reasoning_tokens: output_tokens.reasoning,
            cached_input_tokens: input_tokens.cache_read,
            raw: value.raw.clone(),
        }
    }
}

/// AI SDK-style embedding usage shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingModelUsage {
    /// Tokens used by the embedding call.
    pub tokens: u32,
}

impl EmbeddingModelUsage {
    /// Create embedding usage from a token count.
    pub const fn new(tokens: u32) -> Self {
        Self { tokens }
    }
}

impl From<EmbeddingUsage> for EmbeddingModelUsage {
    fn from(value: EmbeddingUsage) -> Self {
        Self {
            tokens: value.prompt_tokens,
        }
    }
}

impl From<&EmbeddingUsage> for EmbeddingModelUsage {
    fn from(value: &EmbeddingUsage) -> Self {
        Self {
            tokens: value.prompt_tokens,
        }
    }
}

/// AI SDK-style image-model usage shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ImageModelUsage {
    /// Input prompt tokens when the provider reports them.
    #[serde(rename = "inputTokens", skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    /// Output tokens when the provider reports them.
    #[serde(rename = "outputTokens", skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    /// Total tokens when the provider reports them.
    #[serde(rename = "totalTokens", skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

impl ImageModelUsage {
    /// Create image-model usage from optional token totals.
    pub const fn new(
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
        total_tokens: Option<u32>,
    ) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens,
        }
    }

    /// Merge another image usage payload into this one.
    pub fn merge(&mut self, other: &Self) {
        self.input_tokens = add_optional_u32(self.input_tokens, other.input_tokens);
        self.output_tokens = add_optional_u32(self.output_tokens, other.output_tokens);
        self.total_tokens = add_optional_u32(self.total_tokens, other.total_tokens);
    }
}

/// AI SDK-style helper for adding two image-model usage payloads.
pub fn add_image_model_usage(
    usage1: &ImageModelUsage,
    usage2: &ImageModelUsage,
) -> ImageModelUsage {
    let mut merged = usage1.clone();
    merged.merge(usage2);
    merged
}

/// AI SDK-style request metadata for language-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LanguageModelRequestMetadata {
    /// Serialized request body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl From<HttpRequestInfo> for LanguageModelRequestMetadata {
    fn from(value: HttpRequestInfo) -> Self {
        Self {
            body: value.body.map(string_body_to_json_value),
        }
    }
}

impl From<&HttpRequestInfo> for LanguageModelRequestMetadata {
    fn from(value: &HttpRequestInfo) -> Self {
        Self {
            body: value.body.clone().map(string_body_to_json_value),
        }
    }
}

/// AI SDK-style response metadata for language-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelResponseMetadata {
    /// Response identifier.
    pub id: String,
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl TryFrom<ResponseMetadata> for LanguageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: ResponseMetadata) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&ResponseMetadata> for LanguageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &ResponseMetadata) -> Result<Self, Self::Error> {
        Ok(Self {
            id: value.id.clone().ok_or("missing response id")?,
            timestamp: value.created.ok_or("missing response timestamp")?,
            model_id: value.model.clone().ok_or("missing response model id")?,
            headers: value.headers.clone(),
        })
    }
}

/// AI SDK-style response metadata for image-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl TryFrom<HttpResponseInfo> for ImageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for ImageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
        })
    }
}

/// AI SDK-style response metadata for video-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VideoModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Provider-specific metadata for this call when available.
    #[serde(rename = "providerMetadata", skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<VideoModelProviderMetadata>,
}

impl VideoModelResponseMetadata {
    /// Attach provider metadata when it is not empty.
    pub fn with_provider_metadata(mut self, provider_metadata: VideoModelProviderMetadata) -> Self {
        if !provider_metadata.is_empty() {
            self.provider_metadata = Some(provider_metadata);
        }
        self
    }
}

impl TryFrom<HttpResponseInfo> for VideoModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for VideoModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
            provider_metadata: None,
        })
    }
}

/// AI SDK-style response metadata for speech-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeechModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Raw response body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl TryFrom<HttpResponseInfo> for SpeechModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for SpeechModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
            body: None,
        })
    }
}

/// AI SDK-style response metadata for transcription-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptionModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl TryFrom<HttpResponseInfo> for TranscriptionModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for TranscriptionModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
        })
    }
}

fn add_optional_u32(left: Option<u32>, right: Option<u32>) -> Option<u32> {
    match (left, right) {
        (None, None) => None,
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
    }
}

fn string_body_to_json_value(body: String) -> JSONValue {
    serde_json::from_str(&body).unwrap_or(JSONValue::String(body))
}

#[cfg(test)]
mod tests {
    use super::super::AssistantContent;
    use super::*;

    #[test]
    fn source_shape_matches_language_model_source_contract() {
        let mut provider_metadata = ProviderMetadataMap::new();
        provider_metadata.insert("anthropic".to_string(), serde_json::json!({ "foo": "bar" }));

        let source = Source::url_with_title("source-0", "https://example.com", "Example")
            .with_provider_metadata(provider_metadata);

        let value = serde_json::to_value(&source).expect("serialize source");
        assert_eq!(value["type"], serde_json::json!("source"));
        assert_eq!(value["sourceType"], serde_json::json!("url"));
        assert_eq!(value["id"], serde_json::json!("source-0"));
        assert_eq!(value["url"], serde_json::json!("https://example.com"));
        assert_eq!(value["title"], serde_json::json!("Example"));
        assert_eq!(
            value["providerMetadata"]["anthropic"],
            serde_json::json!({ "foo": "bar" })
        );

        let roundtrip: Source = serde_json::from_value(value).expect("deserialize source");
        assert_eq!(roundtrip.r#type(), "source");
        assert_eq!(roundtrip.source_type(), "url");
        assert_eq!(roundtrip, source);

        let content_part: ContentPart = source.clone().into();
        let converted = Source::try_from(&content_part).expect("convert source content part");
        assert_eq!(converted, source);
    }

    #[test]
    fn source_rejects_non_source_type_marker() {
        let error = serde_json::from_value::<Source>(serde_json::json!({
            "type": "text",
            "sourceType": "url",
            "id": "source-0",
            "url": "https://example.com"
        }))
        .expect_err("non-source marker should be rejected");

        assert!(error.to_string().contains("expected source type marker"));
    }

    #[test]
    fn language_model_request_metadata_parses_json_body_and_falls_back_to_string() {
        let json = LanguageModelRequestMetadata::from(HttpRequestInfo {
            body: Some("{\"ok\":true}".to_string()),
        });
        let text = LanguageModelRequestMetadata::from(HttpRequestInfo {
            body: Some("plain-text".to_string()),
        });

        assert_eq!(json.body, Some(serde_json::json!({ "ok": true })));
        assert_eq!(text.body, Some(JSONValue::String("plain-text".to_string())));
    }

    #[test]
    fn language_model_response_metadata_requires_main_fields() {
        let metadata = ResponseMetadata {
            id: Some("resp_123".to_string()),
            model: Some("gpt-4o".to_string()),
            created: Some(
                DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                    .expect("valid timestamp")
                    .with_timezone(&Utc),
            ),
            provider: "openai".to_string(),
            request_id: None,
            headers: Some(HashMap::from([(
                "x-request-id".to_string(),
                "req_123".to_string(),
            )])),
        };

        let converted =
            LanguageModelResponseMetadata::try_from(&metadata).expect("convert response metadata");

        assert_eq!(converted.id, "resp_123");
        assert_eq!(converted.model_id, "gpt-4o");
        assert_eq!(
            converted
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-request-id"))
                .map(String::as_str),
            Some("req_123")
        );
    }

    #[test]
    fn image_and_transcription_response_metadata_require_model_id() {
        let response = HttpResponseInfo {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: Some("model-1".to_string()),
            headers: HashMap::from([("x-test".to_string(), "1".to_string())]),
        };

        let image =
            ImageModelResponseMetadata::try_from(&response).expect("convert image metadata");
        let transcription = TranscriptionModelResponseMetadata::try_from(&response)
            .expect("convert transcription metadata");

        assert_eq!(image.model_id, "model-1");
        assert_eq!(transcription.model_id, "model-1");
    }

    #[test]
    fn cancel_handle_and_request_timeout_helpers_work() {
        let cancel = CancelHandle::new();
        assert!(!cancel.is_cancelled());
        cancel.cancel();
        assert!(cancel.is_cancelled());

        let timeout = TimeoutConfiguration::settings(
            TimeoutConfigurationSettings::new()
                .with_total_ms(3_000)
                .with_step_ms(1_000)
                .with_chunk_ms(250)
                .with_tool_ms(900)
                .with_tool_timeout_ms("weather", 1200),
        );

        let request = RequestOptions::new()
            .with_max_retries(2)
            .with_abort_signal(cancel)
            .with_header("x-test", "1")
            .without_header("x-drop")
            .with_timeout(timeout.clone());

        assert_eq!(timeout.total_timeout_ms(), Some(3_000));
        assert_eq!(timeout.step_timeout_ms(), Some(1_000));
        assert_eq!(timeout.chunk_timeout_ms(), Some(250));
        assert_eq!(timeout.tool_timeout_ms("weather"), Some(1_200));
        assert_eq!(timeout.tool_timeout_ms("other"), Some(900));
        assert_eq!(request.max_attempts(), Some(3));
        assert_eq!(
            request.effective_headers(),
            HashMap::from([("x-test".to_string(), "1".to_string())])
        );
        assert_eq!(request.total_timeout(), Some(Duration::from_millis(3_000)));
        assert_eq!(request.step_timeout(), Some(Duration::from_millis(1_000)));
        assert_eq!(request.chunk_timeout(), Some(Duration::from_millis(250)));
        assert_eq!(request.tool_timeout_ms("weather"), Some(1_200));
        assert!(
            request
                .abort_signal
                .as_ref()
                .is_some_and(CancelHandle::is_cancelled)
        );
    }

    #[test]
    fn embedding_and_image_usage_shared_shapes_are_available() {
        let embedding = EmbeddingModelUsage::from(EmbeddingUsage::new(12, 12));
        let mut image = ImageModelUsage::new(Some(3), Some(5), Some(8));
        image.merge(&ImageModelUsage::new(Some(2), None, Some(2)));
        let added_image =
            add_image_model_usage(&image, &ImageModelUsage::new(None, Some(1), Some(1)));
        let call_options = LanguageModelCallOptions::from(&super::super::CommonParams {
            model: "gpt-5".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(128),
            max_completion_tokens: Some(256),
            top_p: Some(0.9),
            top_k: Some(40.0),
            stop_sequences: Some(vec!["END".to_string()]),
            seed: Some(7),
            frequency_penalty: Some(0.2),
            presence_penalty: Some(0.1),
        });
        let mut language = LanguageModelUsage::from(
            Usage::builder()
                .with_input_tokens(super::super::UsageInputTokens {
                    total: Some(10),
                    no_cache: Some(7),
                    cache_read: Some(2),
                    cache_write: Some(1),
                })
                .with_output_tokens(super::super::UsageOutputTokens {
                    total: Some(6),
                    text: Some(4),
                    reasoning: Some(2),
                })
                .with_raw_usage_value(serde_json::json!({
                    "provider_total_tokens": 16
                }))
                .build(),
        );
        language.merge(&LanguageModelUsage {
            input_tokens: Some(2),
            input_token_details: LanguageModelInputTokenDetails {
                no_cache_tokens: Some(1),
                cache_read_tokens: Some(1),
                cache_write_tokens: None,
            },
            output_tokens: Some(1),
            output_token_details: LanguageModelOutputTokenDetails {
                text_tokens: Some(1),
                reasoning_tokens: None,
            },
            total_tokens: Some(3),
            reasoning_tokens: None,
            cached_input_tokens: Some(1),
            raw: Some(serde_json::Map::new()),
        });
        let added_language = add_language_model_usage(
            &create_null_language_model_usage(),
            &LanguageModelUsage {
                input_tokens: Some(4),
                input_token_details: LanguageModelInputTokenDetails::default(),
                output_tokens: Some(3),
                output_token_details: LanguageModelOutputTokenDetails::default(),
                total_tokens: Some(7),
                reasoning_tokens: None,
                cached_input_tokens: None,
                raw: Some(serde_json::Map::new()),
            },
        );

        assert_eq!(embedding.tokens, 12);
        assert_eq!(image.input_tokens, Some(5));
        assert_eq!(image.output_tokens, Some(5));
        assert_eq!(image.total_tokens, Some(10));
        assert_eq!(added_image.input_tokens, Some(5));
        assert_eq!(added_image.output_tokens, Some(6));
        assert_eq!(added_image.total_tokens, Some(11));
        assert_eq!(call_options.max_output_tokens, Some(256));
        assert_eq!(call_options.temperature, Some(0.7));
        assert_eq!(call_options.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(call_options.reasoning, None);
        assert_eq!(language.input_tokens, Some(12));
        assert_eq!(language.output_tokens, Some(7));
        assert_eq!(language.total_tokens, Some(19));
        assert_eq!(language.input_token_details.no_cache_tokens, Some(8));
        assert_eq!(language.cached_input_tokens, Some(3));
        assert_eq!(language.output_token_details.reasoning_tokens, Some(2));
        assert_eq!(language.raw, None);
        assert_eq!(added_language.input_tokens, Some(4));
        assert_eq!(added_language.output_tokens, Some(3));
        assert_eq!(added_language.total_tokens, Some(7));
        assert_eq!(added_language.raw, None);
    }

    #[test]
    fn video_model_response_metadata_attaches_non_empty_provider_metadata() {
        let metadata = VideoModelResponseMetadata::try_from(&HttpResponseInfo {
            timestamp: Utc::now(),
            model_id: Some("video-model".to_string()),
            headers: HashMap::from([("x-video".to_string(), "1".to_string())]),
        })
        .expect("valid video response metadata")
        .with_provider_metadata(HashMap::from([(
            "fake-video".to_string(),
            serde_json::json!({ "taskId": "task-1" }),
        )]));

        assert_eq!(metadata.model_id, "video-model");
        assert_eq!(
            metadata
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-video")),
            Some(&"1".to_string())
        );
        assert_eq!(
            metadata
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("fake-video"))
                .and_then(|value| value.get("taskId"))
                .and_then(serde_json::Value::as_str),
            Some("task-1")
        );
    }

    #[test]
    fn timeout_helper_functions_follow_ai_sdk_semantics() {
        let timeout = TimeoutConfiguration::settings(
            TimeoutConfigurationSettings::new()
                .with_total_ms(1_000)
                .with_step_ms(200)
                .with_chunk_ms(50)
                .with_tool_ms(300)
                .with_tool_timeout_ms("search", 450),
        );

        assert_eq!(get_total_timeout_ms(Some(&timeout)), Some(1_000));
        assert_eq!(get_step_timeout_ms(Some(&timeout)), Some(200));
        assert_eq!(get_chunk_timeout_ms(Some(&timeout)), Some(50));
        assert_eq!(get_tool_timeout_ms(Some(&timeout), "search"), Some(450));
        assert_eq!(get_tool_timeout_ms(Some(&timeout), "other"), Some(300));
        assert_eq!(get_total_timeout_ms(None), None);
    }

    #[test]
    fn provider_utils_style_tool_and_context_types_are_available() {
        let mut context = Context::new();
        context.insert("tenant".to_string(), serde_json::json!("acme"));

        let mut provider_options = ProviderOptions::default();
        provider_options.insert("openai", serde_json::json!({ "serviceTier": "flex" }));

        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "itemId": "item_1" }),
        )]);

        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
        )
        .with_provider_executed(true)
        .with_dynamic(true)
        .with_invalid(true)
        .with_error(serde_json::json!({ "message": "unknown tool" }))
        .with_title("Search")
        .with_provider_metadata(provider_metadata.clone());

        let tool_result = ToolResult::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
            ToolResultOutput::json(serde_json::json!({ "ok": true })),
        )
        .with_provider_executed(true)
        .with_dynamic(true)
        .with_preliminary(true)
        .with_title("Search result")
        .with_provider_metadata(provider_metadata);

        assert_eq!(context.get("tenant"), Some(&serde_json::json!("acme")));
        assert_eq!(
            provider_options
                .get("openai")
                .and_then(|value| value.get("serviceTier"))
                .and_then(serde_json::Value::as_str),
            Some("flex")
        );
        assert_eq!(tool_call.tool_call_id, "call_1");
        assert_eq!(tool_call.provider_executed, Some(true));
        assert_eq!(tool_call.invalid, Some(true));
        assert_eq!(tool_result.tool_call_id, "call_1");
        assert_eq!(tool_result.dynamic, Some(true));

        let tool_call_json = serde_json::to_value(&tool_call).expect("serialize tool call");
        assert_eq!(tool_call.r#type(), "tool-call");
        assert_eq!(tool_call_json["type"], serde_json::json!("tool-call"));
        assert_eq!(
            tool_call_json["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("item_1")
        );
        assert_eq!(tool_call_json["title"], serde_json::json!("Search"));
        assert_eq!(tool_call_json["invalid"], serde_json::json!(true));
        assert_eq!(
            tool_call_json["error"],
            serde_json::json!({ "message": "unknown tool" })
        );

        let tool_result_json = serde_json::to_value(&tool_result).expect("serialize tool result");
        assert_eq!(tool_result.r#type(), "tool-result");
        assert_eq!(tool_result_json["type"], serde_json::json!("tool-result"));
        assert_eq!(
            tool_result_json["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("item_1")
        );
        assert_eq!(
            tool_result_json["title"],
            serde_json::json!("Search result")
        );
        assert_eq!(tool_result_json["preliminary"], serde_json::json!(true));
    }

    #[test]
    fn generate_text_basic_content_outputs_match_ai_sdk_shape() {
        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "itemId": "msg_1" }),
        )]);

        let text = TextOutput::new("hello").with_provider_metadata(provider_metadata.clone());
        assert_eq!(text.r#type(), "text");
        assert_eq!(
            serde_json::to_value(&text).expect("serialize text output"),
            serde_json::json!({
                "type": "text",
                "text": "hello",
                "providerMetadata": {
                    "openai": { "itemId": "msg_1" }
                }
            })
        );

        let custom = CustomOutput::new("openai.compaction")
            .with_provider_metadata(provider_metadata.clone());
        assert_eq!(custom.r#type(), "custom");
        assert_eq!(
            serde_json::to_value(&custom).expect("serialize custom output"),
            serde_json::json!({
                "type": "custom",
                "kind": "openai.compaction",
                "providerMetadata": {
                    "openai": { "itemId": "msg_1" }
                }
            })
        );

        let file = FileOutput::new(GeneratedFile::from_bytes(b"hello", "text/plain"))
            .with_provider_metadata(provider_metadata);
        assert_eq!(file.r#type(), "file");
        assert_eq!(
            serde_json::to_value(&file).expect("serialize file output"),
            serde_json::json!({
                "type": "file",
                "file": {
                    "base64": "aGVsbG8=",
                    "mediaType": "text/plain"
                },
                "providerMetadata": {
                    "openai": { "itemId": "msg_1" }
                }
            })
        );

        let wrong_type = serde_json::json!({
            "type": "image",
            "file": {
                "base64": "aGVsbG8=",
                "mediaType": "text/plain"
            }
        });
        assert!(serde_json::from_value::<FileOutput>(wrong_type).is_err());
    }

    #[test]
    fn generate_text_content_part_union_roundtrips_ai_sdk_shape() {
        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
        )
        .with_dynamic(true);
        let parts: Vec<GenerateTextContentPart> = vec![
            TextOutput::new("hello").into(),
            CustomOutput::new("openai.compaction").into(),
            ReasoningOutput::new("thinking").into(),
            ReasoningFileOutput::new(GeneratedFile::from_bytes(b"trace", "text/plain")).into(),
            Source::url("source_1", "https://example.com").into(),
            FileOutput::new(GeneratedFile::from_bytes(b"hello", "text/plain")).into(),
            tool_call.clone().into(),
            ToolResult::new(
                "call_1",
                "search".to_string(),
                serde_json::json!({ "q": "rust" }),
                ToolResultOutput::json(serde_json::json!({ "ok": true })),
            )
            .into(),
            ToolError::new(
                "call_2",
                "fetch".to_string(),
                serde_json::json!({ "url": "https://example.com" }),
                serde_json::json!({ "message": "timeout" }),
            )
            .into(),
            ToolApprovalRequestOutput::new("approval_1", tool_call.clone()).into(),
            ToolApprovalResponseOutput::new("approval_1", tool_call, true).into(),
        ];

        let part_types: Vec<&'static str> =
            parts.iter().map(GenerateTextContentPart::r#type).collect();
        assert_eq!(
            part_types,
            vec![
                "text",
                "custom",
                "reasoning",
                "reasoning-file",
                "source",
                "file",
                "tool-call",
                "tool-result",
                "tool-error",
                "tool-approval-request",
                "tool-approval-response"
            ]
        );

        let json = serde_json::to_value(&parts).expect("serialize generate text content parts");
        assert_eq!(json[0]["type"], serde_json::json!("text"));
        assert_eq!(
            json[5]["file"]["mediaType"],
            serde_json::json!("text/plain")
        );
        assert_eq!(json[6]["type"], serde_json::json!("tool-call"));
        assert_eq!(json[7]["type"], serde_json::json!("tool-result"));
        assert_eq!(json[9]["toolCall"]["type"], serde_json::json!("tool-call"));

        let roundtrip: Vec<GenerateTextContentPart> =
            serde_json::from_value(json).expect("deserialize generate text content parts");
        let roundtrip_types: Vec<&'static str> = roundtrip
            .iter()
            .map(GenerateTextContentPart::r#type)
            .collect();
        assert_eq!(roundtrip_types, part_types);
    }

    #[test]
    fn generate_text_result_envelope_matches_ai_sdk_shape() {
        let timestamp = DateTime::parse_from_rfc3339("2026-04-24T00:00:00Z")
            .expect("timestamp")
            .with_timezone(&Utc);
        let response_metadata = LanguageModelResponseMetadata {
            id: "resp_1".to_string(),
            timestamp,
            model_id: "gpt-test".to_string(),
            headers: None,
        };
        let response = GenerateTextResponseMetadata::new(response_metadata)
            .with_messages(vec![
                AssistantModelMessage::new(AssistantContent::text("hello")).into(),
            ])
            .with_body(serde_json::json!({ "id": "resp_1" }));
        let request = LanguageModelRequestMetadata {
            body: Some(serde_json::json!({ "messages": [] })),
        };
        let usage = LanguageModelUsage {
            input_tokens: Some(3),
            output_tokens: Some(2),
            total_tokens: Some(5),
            ..LanguageModelUsage::default()
        };
        let source = Source::url("source_1", "https://example.com");
        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
        );
        let tool_result = ToolResult::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
            ToolResultOutput::json(serde_json::json!({ "ok": true })),
        );
        let reasoning_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "reasoningId": "rs_1" }),
        )]);
        let content: Vec<GenerateTextContentPart> = vec![
            TextOutput::new("hello").into(),
            source.clone().into(),
            tool_call.clone().into(),
            tool_result.clone().into(),
        ];
        let reasoning_output =
            ReasoningOutput::new("because").with_provider_metadata(reasoning_metadata.clone());
        let reasoning = vec![reasoning_output.clone().into()];
        let step_reasoning = vec![
            GenerateTextStepReasoningPart::from(reasoning_output),
            GenerateTextStepReasoningPart::from(
                ReasoningFileOutput::new(GeneratedFile::from_base64("dHJhY2U=", "text/plain"))
                    .with_provider_metadata(reasoning_metadata),
            ),
        ];
        let files = vec![GeneratedFile::from_bytes(b"hello", "text/plain")];
        let model = GenerateTextModelInfo::new("openai", "gpt-test");
        let step = GenerateTextStepResult {
            call_id: "call_root".to_string(),
            step_number: 0,
            model,
            tools_context: Context::new(),
            runtime_context: Context::new(),
            content: content.clone(),
            text: "hello".to_string(),
            reasoning: step_reasoning,
            reasoning_text: Some("because".to_string()),
            files: files.clone(),
            sources: vec![source.clone()],
            tool_calls: vec![tool_call.clone()],
            static_tool_calls: vec![tool_call.clone()],
            dynamic_tool_calls: Vec::new(),
            tool_results: vec![tool_result.clone()],
            static_tool_results: vec![tool_result.clone()],
            dynamic_tool_results: Vec::new(),
            finish_reason: FinishReason::Stop,
            raw_finish_reason: Some("stop".to_string()),
            usage: usage.clone(),
            warnings: None,
            request: request.clone(),
            response: response.clone(),
            provider_metadata: None,
        };
        let result = GenerateTextResult {
            content,
            text: "hello".to_string(),
            reasoning,
            reasoning_text: Some("because".to_string()),
            files,
            sources: vec![source],
            tool_calls: vec![tool_call.clone()],
            static_tool_calls: vec![tool_call],
            dynamic_tool_calls: Vec::new(),
            tool_results: vec![tool_result.clone()],
            static_tool_results: vec![tool_result],
            dynamic_tool_results: Vec::new(),
            finish_reason: FinishReason::Stop,
            raw_finish_reason: Some("stop".to_string()),
            usage: usage.clone(),
            total_usage: usage,
            warnings: None,
            request,
            response,
            provider_metadata: None,
            steps: vec![step],
            output: serde_json::json!("hello"),
        };

        let json = serde_json::to_value(&result).expect("serialize generate text result");
        assert_eq!(json["content"][0]["type"], serde_json::json!("text"));
        assert_eq!(json["reasoning"][0]["type"], serde_json::json!("reasoning"));
        assert_eq!(
            json["reasoning"][0]["providerMetadata"]["openai"]["reasoningId"],
            serde_json::json!("rs_1")
        );
        assert_eq!(
            json["files"][0]["mediaType"],
            serde_json::json!("text/plain")
        );
        assert_eq!(json["finishReason"], serde_json::json!("stop"));
        assert_eq!(json["rawFinishReason"], serde_json::json!("stop"));
        assert_eq!(json["totalUsage"]["totalTokens"], serde_json::json!(5));
        assert_eq!(
            json["response"]["messages"][0]["role"],
            serde_json::json!("assistant")
        );
        assert_eq!(json["steps"][0]["callId"], serde_json::json!("call_root"));
        assert_eq!(
            json["steps"][0]["model"]["modelId"],
            serde_json::json!("gpt-test")
        );
        assert_eq!(
            json["steps"][0]["reasoning"][0]["providerOptions"]["openai"]["reasoningId"],
            serde_json::json!("rs_1")
        );
        assert_eq!(
            json["steps"][0]["reasoning"][1],
            serde_json::json!({
                "type": "reasoning-file",
                "data": "dHJhY2U=",
                "mediaType": "text/plain",
                "providerOptions": {
                    "openai": { "reasoningId": "rs_1" }
                }
            })
        );

        let roundtrip: GenerateTextResult =
            serde_json::from_value(json).expect("deserialize generate text result");
        assert_eq!(roundtrip.text, "hello");
        assert_eq!(roundtrip.steps.len(), 1);
        assert_eq!(roundtrip.content[2].r#type(), "tool-call");
        assert_eq!(roundtrip.steps[0].reasoning[1].r#type(), "reasoning-file");
    }

    #[test]
    fn text_stream_parts_match_ai_sdk_stream_text_result_shape() {
        let timestamp = DateTime::parse_from_rfc3339("2026-04-24T00:00:00Z")
            .expect("timestamp")
            .with_timezone(&Utc);
        let response = LanguageModelResponseMetadata {
            id: "resp_1".to_string(),
            timestamp,
            model_id: "gpt-test".to_string(),
            headers: None,
        };
        let request = LanguageModelRequestMetadata {
            body: Some(serde_json::json!({ "messages": [] })),
        };
        let usage = LanguageModelUsage {
            input_tokens: Some(3),
            output_tokens: Some(2),
            total_tokens: Some(5),
            ..LanguageModelUsage::default()
        };
        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "responseId": "resp_1" }),
        )]);
        let mut finish_step =
            TextStreamFinishStepPart::new(response, usage.clone(), FinishReason::Stop);
        finish_step.raw_finish_reason = Some("stop".to_string());
        finish_step.provider_metadata = Some(provider_metadata);
        let mut finish = TextStreamFinishPart::new(FinishReason::Stop, usage);
        finish.raw_finish_reason = Some("stop".to_string());

        let parts: Vec<TextStreamPart> = vec![
            TextStreamStartPart::new().into(),
            TextStreamTextStartPart::new("text_1").into(),
            TextStreamTextDeltaPart::new("text_1", "hello").into(),
            TextStreamReasoningDeltaPart::new("reasoning_1", "because").into(),
            TextStreamToolInputStartPart::new("call_1", "search").into(),
            ToolCall::new(
                "call_1",
                "search".to_string(),
                serde_json::json!({ "q": "rust" }),
            )
            .into(),
            TextStreamFilePart::new(GeneratedFile::from_bytes(b"hello", "text/plain")).into(),
            TextStreamStartStepPart::new(request, Vec::new()).into(),
            finish_step.into(),
            finish.into(),
            TextStreamAbortPart::new().with_reason("user").into(),
            TextStreamRawPart::new(serde_json::json!({ "chunk": 1 })).into(),
        ];

        let json = serde_json::to_value(&parts).expect("serialize text stream parts");
        assert_eq!(json[0]["type"], serde_json::json!("start"));
        assert_eq!(json[2]["type"], serde_json::json!("text-delta"));
        assert_eq!(json[2]["text"], serde_json::json!("hello"));
        assert!(json[2]["delta"].is_null());
        assert_eq!(json[3]["type"], serde_json::json!("reasoning-delta"));
        assert_eq!(json[3]["text"], serde_json::json!("because"));
        assert_eq!(json[5]["type"], serde_json::json!("tool-call"));
        assert_eq!(
            json[6]["file"]["mediaType"],
            serde_json::json!("text/plain")
        );
        assert_eq!(json[8]["type"], serde_json::json!("finish-step"));
        assert_eq!(json[8]["finishReason"], serde_json::json!("stop"));
        assert_eq!(json[8]["rawFinishReason"], serde_json::json!("stop"));
        assert_eq!(
            json[8]["providerMetadata"]["openai"]["responseId"],
            serde_json::json!("resp_1")
        );
        assert_eq!(json[9]["type"], serde_json::json!("finish"));
        assert_eq!(json[9]["totalUsage"]["totalTokens"], serde_json::json!(5));
        assert!(json[9]["usage"].is_null());
        assert_eq!(json[10]["reason"], serde_json::json!("user"));
        assert_eq!(json[11]["rawValue"]["chunk"], serde_json::json!(1));

        let roundtrip: Vec<TextStreamPart> =
            serde_json::from_value(json).expect("deserialize text stream parts");
        assert_eq!(roundtrip[2].r#type(), "text-delta");
        assert_eq!(roundtrip[8].r#type(), "finish-step");
        assert_eq!(roundtrip[11].r#type(), "raw");
    }

    #[test]
    fn generate_text_callback_start_events_match_ai_sdk_shape() {
        let mut provider_options = ProviderOptionsMap::new();
        provider_options.insert("openai", serde_json::json!({ "reasoningEffort": "low" }));
        let mut tools_context = Context::new();
        tools_context.insert("tenant".to_string(), serde_json::json!("docs"));
        let mut runtime_context = Context::new();
        runtime_context.insert("traceId".to_string(), serde_json::json!("trace_1"));
        let prompt = StandardizedPrompt {
            system: None,
            messages: vec![ModelMessage::Assistant(AssistantModelMessage::new(
                AssistantContent::text("ready"),
            ))],
        };

        let start_event = GenerateTextStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.generateText".to_string(),
            model: CallbackModelInfo::new("openai", "gpt-test"),
            tools: Some(vec![Tool::function(
                "search",
                "Search docs",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    }
                }),
            )]),
            tool_choice: Some(ToolChoice::tool("search")),
            active_tools: Some(vec!["search".to_string()]),
            max_retries: 2,
            timeout: Some(TimeoutConfiguration::settings(
                TimeoutConfigurationSettings::new().with_total_ms(30_000),
            )),
            headers: Some(HashMap::from([
                ("x-test".to_string(), Some("1".to_string())),
                ("x-drop".to_string(), None),
            ])),
            provider_options: Some(provider_options.clone()),
            stop_when: vec![serde_json::json!({ "type": "step-count", "maxSteps": 2 })],
            output: Some(serde_json::json!({ "type": "object" })),
            tools_context: tools_context.clone(),
            runtime_context: runtime_context.clone(),
            call_options: LanguageModelCallOptions {
                temperature: Some(0.2),
                ..LanguageModelCallOptions::default()
            },
            prompt: prompt.clone(),
        };

        let json = serde_json::to_value(&start_event).expect("serialize start event");

        assert_eq!(json["callId"], serde_json::json!("call_1"));
        assert_eq!(json["operationId"], serde_json::json!("ai.generateText"));
        assert_eq!(json["provider"], serde_json::json!("openai"));
        assert_eq!(json["modelId"], serde_json::json!("gpt-test"));
        assert_eq!(json["toolChoice"]["toolName"], serde_json::json!("search"));
        assert_eq!(json["activeTools"][0], serde_json::json!("search"));
        assert_eq!(
            json["providerOptions"]["openai"]["reasoningEffort"],
            serde_json::json!("low")
        );
        assert_eq!(json["toolsContext"]["tenant"], serde_json::json!("docs"));
        assert_eq!(
            json["runtimeContext"]["traceId"],
            serde_json::json!("trace_1")
        );
        assert_eq!(json["maxRetries"], serde_json::json!(2));
        assert_eq!(json["timeout"]["totalMs"], serde_json::json!(30_000));
        assert_eq!(json["headers"]["x-test"], serde_json::json!("1"));
        assert!(json["headers"]["x-drop"].is_null());
        assert_eq!(json["messages"][0]["role"], serde_json::json!("assistant"));
        assert_eq!(json["temperature"], serde_json::json!(0.2));

        let step_event = GenerateTextStepStartEvent {
            call_id: "call_1".to_string(),
            model: CallbackModelInfo::new("openai", "gpt-test"),
            step_number: 1,
            tools: None,
            tool_choice: Some(ToolChoice::Auto),
            active_tools: Some(vec!["search".to_string()]),
            steps: Vec::<GenerateTextStepResult>::new(),
            provider_options: Some(provider_options),
            output: Some(serde_json::json!({ "type": "object" })),
            runtime_context,
            tools_context,
            prompt,
        };
        let step_json = serde_json::to_value(&step_event).expect("serialize step start event");

        assert_eq!(step_json["callId"], serde_json::json!("call_1"));
        assert_eq!(step_json["stepNumber"], serde_json::json!(1));
        assert_eq!(step_json["toolChoice"], serde_json::json!("auto"));
        assert_eq!(step_json["activeTools"][0], serde_json::json!("search"));
        assert!(step_json["steps"].as_array().is_some_and(Vec::is_empty));
    }

    #[test]
    fn stream_text_chunk_event_accepts_parts_and_lifecycle_markers() {
        let part_event: StreamTextChunkEvent = StreamTextChunkEvent::new(TextStreamPart::from(
            TextStreamTextDeltaPart::new("text_1", "hello"),
        ));
        let part_json = serde_json::to_value(&part_event).expect("serialize part chunk event");

        assert_eq!(part_json["chunk"]["type"], serde_json::json!("text-delta"));
        assert_eq!(part_json["chunk"]["text"], serde_json::json!("hello"));

        let lifecycle_event: StreamTextChunkEvent = StreamTextChunkEvent::new(
            StreamTextLifecycleChunk::first_chunk("call_1", 0)
                .with_attribute("phase", serde_json::json!("first")),
        );
        let lifecycle_json =
            serde_json::to_value(&lifecycle_event).expect("serialize lifecycle chunk event");

        assert_eq!(
            lifecycle_json["chunk"]["type"],
            serde_json::json!("ai.stream.firstChunk")
        );
        assert_eq!(
            lifecycle_json["chunk"]["callId"],
            serde_json::json!("call_1")
        );
        assert_eq!(lifecycle_json["chunk"]["stepNumber"], serde_json::json!(0));
        assert_eq!(
            lifecycle_json["chunk"]["attributes"]["phase"],
            serde_json::json!("first")
        );

        let roundtrip: StreamTextChunkEvent =
            serde_json::from_value(lifecycle_json).expect("deserialize lifecycle chunk event");
        assert_eq!(roundtrip.chunk.r#type(), "ai.stream.firstChunk");
    }

    #[test]
    fn generated_file_and_reasoning_outputs_match_ai_sdk_shape() {
        let file = GeneratedFile::from_bytes(b"hello", "text/plain");
        assert_eq!(file.base64(), "aGVsbG8=");
        assert_eq!(file.uint8_array().expect("decode generated file"), b"hello");
        assert_eq!(
            serde_json::to_value(&file).expect("serialize generated file"),
            serde_json::json!({
                "base64": "aGVsbG8=",
                "mediaType": "text/plain"
            })
        );

        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "reasoningId": "rs_1" }),
        )]);
        let reasoning = ReasoningOutput::new("internal reasoning")
            .with_provider_metadata(provider_metadata.clone());
        let reasoning_json = serde_json::to_value(&reasoning).expect("serialize reasoning output");
        assert_eq!(reasoning.r#type(), "reasoning");
        assert_eq!(
            reasoning_json,
            serde_json::json!({
                "type": "reasoning",
                "text": "internal reasoning",
                "providerMetadata": {
                    "openai": { "reasoningId": "rs_1" }
                }
            })
        );
        let reasoning_roundtrip: ReasoningOutput =
            serde_json::from_value(reasoning_json).expect("deserialize reasoning output");
        assert_eq!(reasoning_roundtrip.text, "internal reasoning");

        let reasoning_file =
            ReasoningFileOutput::new(file).with_provider_metadata(provider_metadata);
        let reasoning_file_json =
            serde_json::to_value(&reasoning_file).expect("serialize reasoning-file output");
        assert_eq!(reasoning_file.r#type(), "reasoning-file");
        assert_eq!(
            reasoning_file_json,
            serde_json::json!({
                "type": "reasoning-file",
                "file": {
                    "base64": "aGVsbG8=",
                    "mediaType": "text/plain"
                },
                "providerMetadata": {
                    "openai": { "reasoningId": "rs_1" }
                }
            })
        );
        let wrong_type = serde_json::json!({
            "type": "reasoning-text",
            "text": "nope"
        });
        assert!(serde_json::from_value::<ReasoningOutput>(wrong_type).is_err());
    }

    #[test]
    fn tool_error_and_output_denied_match_ai_sdk_shape() {
        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "itemId": "item_error" }),
        )]);
        let tool_error = ToolError::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
            serde_json::json!({ "message": "timeout" }),
        )
        .with_provider_executed(true)
        .with_provider_metadata(provider_metadata)
        .with_dynamic(true)
        .with_title("Search failed");
        let tool_error_json = serde_json::to_value(&tool_error).expect("serialize tool error");

        assert_eq!(tool_error.r#type(), "tool-error");
        assert_eq!(
            tool_error_json,
            serde_json::json!({
                "type": "tool-error",
                "toolCallId": "call_1",
                "toolName": "search",
                "input": { "q": "rust" },
                "error": { "message": "timeout" },
                "providerExecuted": true,
                "providerMetadata": {
                    "openai": { "itemId": "item_error" }
                },
                "dynamic": true,
                "title": "Search failed"
            })
        );
        let roundtrip: ToolError =
            serde_json::from_value(tool_error_json).expect("deserialize tool error");
        assert_eq!(roundtrip.tool_call_id, "call_1");

        let denied = ToolOutputDenied::new("call_2", "delete".to_string())
            .with_provider_executed(false)
            .with_dynamic(false);
        let denied_json = serde_json::to_value(&denied).expect("serialize output denied");

        assert_eq!(denied.r#type(), "tool-output-denied");
        assert_eq!(
            denied_json,
            serde_json::json!({
                "type": "tool-output-denied",
                "toolCallId": "call_2",
                "toolName": "delete",
                "providerExecuted": false,
                "dynamic": false
            })
        );
        let wrong_type = serde_json::json!({
            "type": "tool-denied",
            "toolCallId": "call_2",
            "toolName": "delete"
        });
        assert!(serde_json::from_value::<ToolOutputDenied>(wrong_type).is_err());
    }

    #[test]
    fn tool_execution_events_and_tool_output_match_ai_sdk_shape() {
        let messages = vec![ModelMessage::Assistant(AssistantModelMessage::new(
            AssistantContent::text("calling search"),
        ))];
        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "query": "rust" }),
        );
        let start_event = ToolExecutionStartEvent {
            call_id: "call_1".to_string(),
            messages: messages.clone(),
            tool_call: tool_call.clone(),
            tool_context: Some(serde_json::json!({ "tenant": "docs" })),
        };
        let start_json =
            serde_json::to_value(&start_event).expect("serialize tool execution start event");

        assert_eq!(start_json["callId"], serde_json::json!("call_1"));
        assert_eq!(
            start_json["toolCall"]["toolCallId"],
            serde_json::json!("call_1")
        );
        assert_eq!(
            start_json["toolContext"]["tenant"],
            serde_json::json!("docs")
        );
        assert_eq!(
            start_json["messages"][0]["role"],
            serde_json::json!("assistant")
        );

        let tool_output = ToolOutput::from(ToolResult::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "query": "rust" }),
            ToolResultOutput::json(serde_json::json!({ "answer": "ok" })),
        ));
        let end_event = ToolExecutionEndEvent {
            call_id: "call_1".to_string(),
            duration_ms: 42,
            messages,
            tool_call,
            tool_context: Some(serde_json::json!({ "tenant": "docs" })),
            tool_output,
        };
        let end_json =
            serde_json::to_value(&end_event).expect("serialize tool execution end event");

        assert_eq!(end_json["durationMs"], serde_json::json!(42));
        assert_eq!(
            end_json["toolCall"]["toolName"],
            serde_json::json!("search")
        );
        assert_eq!(end_json["toolContext"]["tenant"], serde_json::json!("docs"));
        assert_eq!(
            end_json["toolOutput"]["type"],
            serde_json::json!("tool-result")
        );
        assert_eq!(
            end_json["toolOutput"]["output"]["value"]["answer"],
            serde_json::json!("ok")
        );

        let roundtrip: ToolExecutionEndEvent =
            serde_json::from_value(end_json).expect("deserialize tool execution end event");
        assert_eq!(roundtrip.tool_output.r#type(), "tool-result");

        let error_output: ToolOutput = ToolError::new(
            "call_2",
            "search".to_string(),
            serde_json::json!({ "query": "rust" }),
            serde_json::json!({ "message": "timeout" }),
        )
        .into();
        assert_eq!(error_output.r#type(), "tool-error");
    }

    #[test]
    fn generate_text_tool_approval_outputs_match_ai_sdk_shape() {
        let tool_call = ToolCall::new(
            "call_approval",
            "dangerous_tool".to_string(),
            serde_json::json!({ "path": "/tmp/file" }),
        )
        .with_provider_executed(true)
        .with_title("Dangerous tool");

        let request =
            ToolApprovalRequestOutput::new("approval_1", tool_call.clone()).with_is_automatic(true);
        let request_json = serde_json::to_value(&request).expect("serialize request output");

        assert_eq!(request.r#type(), "tool-approval-request");
        assert_eq!(
            request_json,
            serde_json::json!({
                "type": "tool-approval-request",
                "approvalId": "approval_1",
                "toolCall": {
                    "type": "tool-call",
                    "toolCallId": "call_approval",
                    "toolName": "dangerous_tool",
                    "input": { "path": "/tmp/file" },
                    "providerExecuted": true,
                    "title": "Dangerous tool"
                },
                "isAutomatic": true
            })
        );
        let roundtrip: ToolApprovalRequestOutput =
            serde_json::from_value(request_json).expect("deserialize request output");
        assert_eq!(roundtrip.approval_id, "approval_1");
        assert_eq!(roundtrip.tool_call.tool_call_id, "call_approval");

        let response = ToolApprovalResponseOutput::new("approval_1", tool_call, false)
            .with_reason("denied by policy")
            .with_provider_executed(true);
        let response_json = serde_json::to_value(&response).expect("serialize response output");

        assert_eq!(response.r#type(), "tool-approval-response");
        assert_eq!(
            response_json,
            serde_json::json!({
                "type": "tool-approval-response",
                "approvalId": "approval_1",
                "toolCall": {
                    "type": "tool-call",
                    "toolCallId": "call_approval",
                    "toolName": "dangerous_tool",
                    "input": { "path": "/tmp/file" },
                    "providerExecuted": true,
                    "title": "Dangerous tool"
                },
                "approved": false,
                "reason": "denied by policy",
                "providerExecuted": true
            })
        );
        let wrong_type = serde_json::json!({
            "type": "tool-approval",
            "approvalId": "approval_1",
            "toolCall": {
                "toolCallId": "call_approval",
                "toolName": "dangerous_tool",
                "input": {}
            }
        });
        assert!(serde_json::from_value::<ToolApprovalRequestOutput>(wrong_type).is_err());
    }

    #[test]
    #[allow(deprecated)]
    fn call_settings_projects_onto_call_and_request_options() {
        let abort_signal = CancelHandle::new();
        let settings = CallSettings::new()
            .with_max_output_tokens(256)
            .with_temperature(0.4)
            .with_top_p(0.8)
            .with_stop_sequences(vec!["END".to_string()])
            .with_seed(7)
            .with_reasoning(LanguageModelReasoning::Medium)
            .with_max_retries(2)
            .with_abort_signal(abort_signal.clone())
            .with_header("x-test", "1")
            .without_header("x-drop");

        let call_options = settings.language_model_call_options();
        let request_options = settings.request_options();

        assert_eq!(call_options.max_output_tokens, Some(256));
        assert_eq!(call_options.temperature, Some(0.4));
        assert_eq!(call_options.top_p, Some(0.8));
        assert_eq!(call_options.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(call_options.seed, Some(7));
        assert_eq!(call_options.reasoning, Some(LanguageModelReasoning::Medium));
        assert_eq!(request_options.max_retries, Some(2));
        assert_eq!(request_options.max_attempts(), Some(3));
        assert!(
            request_options
                .abort_signal
                .as_ref()
                .is_some_and(|signal| !signal.is_cancelled())
        );
        assert_eq!(
            request_options.effective_headers().get("x-test"),
            Some(&"1".to_string())
        );
        assert!(!request_options.effective_headers().contains_key("x-drop"));
        assert!(request_options.timeout.is_none());
    }
}
