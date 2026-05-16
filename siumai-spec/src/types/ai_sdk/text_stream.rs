use crate::types::{FinishReason, ToolResultOutput};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{
    CallWarning, GeneratedFile, JSONValue, LanguageModelRequestMetadata,
    LanguageModelResponseMetadata, LanguageModelUsage, ProviderMetadata, Source,
    ToolApprovalRequestOutput, ToolApprovalResponseOutput, ToolCall, ToolError, ToolOutputDenied,
    ToolResult,
};

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
                S: serde::Serializer,
            {
                serializer.serialize_str($value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                if value == $value {
                    Ok(Self::Marker)
                } else {
                    Err(serde::de::Error::custom(format!(
                        "expected stream type marker `{}`, got `{value}`",
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
fixed_text_stream_type_marker!(
    LanguageModelStreamModelCallStartPartMarker,
    "model-call-start"
);
fixed_text_stream_type_marker!(LanguageModelStreamModelCallEndPartMarker, "model-call-end");
fixed_text_stream_type_marker!(
    LanguageModelStreamModelCallResponseMetadataPartMarker,
    "model-call-response-metadata"
);
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

/// Model-call start event from AI SDK `LanguageModelStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelStreamModelCallStartPart {
    #[serde(rename = "type", default)]
    marker: LanguageModelStreamModelCallStartPartMarker,
    pub warnings: Vec<CallWarning>,
}

impl LanguageModelStreamModelCallStartPart {
    pub fn new(warnings: Vec<CallWarning>) -> Self {
        Self {
            marker: LanguageModelStreamModelCallStartPartMarker::Marker,
            warnings,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "model-call-start"
    }
}

/// Model-call finish event from AI SDK `LanguageModelStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelStreamModelCallEndPart {
    #[serde(rename = "type", default)]
    marker: LanguageModelStreamModelCallEndPartMarker,
    pub finish_reason: FinishReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_finish_reason: Option<String>,
    pub usage: LanguageModelUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelStreamModelCallEndPart {
    pub fn new(finish_reason: FinishReason, usage: LanguageModelUsage) -> Self {
        Self {
            marker: LanguageModelStreamModelCallEndPartMarker::Marker,
            finish_reason,
            raw_finish_reason: None,
            usage,
            provider_metadata: None,
        }
    }

    pub const fn r#type(&self) -> &'static str {
        "model-call-end"
    }
}

/// Response metadata event from AI SDK `LanguageModelStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelStreamModelCallResponseMetadataPart {
    #[serde(rename = "type", default)]
    marker: LanguageModelStreamModelCallResponseMetadataPartMarker,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

impl LanguageModelStreamModelCallResponseMetadataPart {
    pub fn new() -> Self {
        Self {
            marker: LanguageModelStreamModelCallResponseMetadataPartMarker::Marker,
            id: None,
            timestamp: None,
            model_id: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    pub const fn r#type(&self) -> &'static str {
        "model-call-response-metadata"
    }
}

impl Default for LanguageModelStreamModelCallResponseMetadataPart {
    fn default() -> Self {
        Self::new()
    }
}

/// AI SDK `LanguageModelStreamPart` union from `generate-text/stream-language-model-call.ts`.
///
/// This is the single-model-call stream lane used before `streamText` enriches the stream with
/// step lifecycle and final result events. It intentionally excludes `TextStreamPart` variants
/// that upstream removes (`finish`, `tool-output-denied`, `start-step`, `finish-step`, `start`,
/// and `abort`) while adding the `model-call-*` metadata events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LanguageModelStreamPart<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    ModelCallStart(LanguageModelStreamModelCallStartPart),
    ModelCallResponseMetadata(LanguageModelStreamModelCallResponseMetadataPart),
    ModelCallEnd(LanguageModelStreamModelCallEndPart),
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
    ToolApprovalRequest(ToolApprovalRequestOutput<NAME, INPUT>),
    ToolApprovalResponse(ToolApprovalResponseOutput<NAME, INPUT>),
    Error(TextStreamErrorPart),
    Raw(TextStreamRawPart),
}

/// Rust-cased alias for AI SDK's experimental language-model stream part export.
pub type ExperimentalLanguageModelStreamPart<
    NAME = String,
    INPUT = JSONValue,
    OUTPUT = ToolResultOutput,
> = LanguageModelStreamPart<NAME, INPUT, OUTPUT>;

/// AI SDK-compatible alias for `Experimental_LanguageModelStreamPart`.
#[allow(non_camel_case_types)]
pub type Experimental_LanguageModelStreamPart<
    NAME = String,
    INPUT = JSONValue,
    OUTPUT = ToolResultOutput,
> = LanguageModelStreamPart<NAME, INPUT, OUTPUT>;

impl<NAME, INPUT, OUTPUT> LanguageModelStreamPart<NAME, INPUT, OUTPUT> {
    /// Return the AI SDK `LanguageModelStreamPart` discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::ModelCallStart(part) => part.r#type(),
            Self::ModelCallResponseMetadata(part) => part.r#type(),
            Self::ModelCallEnd(part) => part.r#type(),
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
            Self::ToolApprovalRequest(part) => part.r#type(),
            Self::ToolApprovalResponse(part) => part.r#type(),
            Self::Error(part) => part.r#type(),
            Self::Raw(part) => part.r#type(),
        }
    }
}

impl<NAME, INPUT, OUTPUT> From<LanguageModelStreamModelCallStartPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: LanguageModelStreamModelCallStartPart) -> Self {
        Self::ModelCallStart(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<LanguageModelStreamModelCallResponseMetadataPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: LanguageModelStreamModelCallResponseMetadataPart) -> Self {
        Self::ModelCallResponseMetadata(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<LanguageModelStreamModelCallEndPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: LanguageModelStreamModelCallEndPart) -> Self {
        Self::ModelCallEnd(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamTextStartPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamTextStartPart) -> Self {
        Self::TextStart(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamTextDeltaPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamTextDeltaPart) -> Self {
        Self::TextDelta(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamTextEndPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamTextEndPart) -> Self {
        Self::TextEnd(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamReasoningStartPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamReasoningStartPart) -> Self {
        Self::ReasoningStart(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamReasoningDeltaPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamReasoningDeltaPart) -> Self {
        Self::ReasoningDelta(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamReasoningEndPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamReasoningEndPart) -> Self {
        Self::ReasoningEnd(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamCustomPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamCustomPart) -> Self {
        Self::Custom(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamToolInputStartPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamToolInputStartPart) -> Self {
        Self::ToolInputStart(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamToolInputDeltaPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamToolInputDeltaPart) -> Self {
        Self::ToolInputDelta(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamToolInputEndPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamToolInputEndPart) -> Self {
        Self::ToolInputEnd(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<Source> for LanguageModelStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: Source) -> Self {
        Self::Source(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamFilePart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamFilePart) -> Self {
        Self::File(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamReasoningFilePart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamReasoningFilePart) -> Self {
        Self::ReasoningFile(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolCall<NAME, INPUT>>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolCall<NAME, INPUT>) -> Self {
        Self::ToolCall(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolResult<NAME, INPUT, OUTPUT>>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolResult<NAME, INPUT, OUTPUT>) -> Self {
        Self::ToolResult(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolError<NAME, INPUT>>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolError<NAME, INPUT>) -> Self {
        Self::ToolError(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolApprovalRequestOutput<NAME, INPUT>>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolApprovalRequestOutput<NAME, INPUT>) -> Self {
        Self::ToolApprovalRequest(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolApprovalResponseOutput<NAME, INPUT>>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolApprovalResponseOutput<NAME, INPUT>) -> Self {
        Self::ToolApprovalResponse(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamErrorPart>
    for LanguageModelStreamPart<NAME, INPUT, OUTPUT>
{
    fn from(value: TextStreamErrorPart) -> Self {
        Self::Error(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<TextStreamRawPart> for LanguageModelStreamPart<NAME, INPUT, OUTPUT> {
    fn from(value: TextStreamRawPart) -> Self {
        Self::Raw(value)
    }
}
