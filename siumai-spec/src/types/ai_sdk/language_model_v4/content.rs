use crate::types::{SourcePart, ToolResultOutput};
use serde::{Deserialize, Serialize};

use super::super::shared::{
    deserialize_ai_sdk_non_null_json_value, serialize_ai_sdk_non_null_json_value,
};
use super::super::source::SourceMarker;
use super::super::{
    CustomOutput, FileOutput, GeneratedFile, JSONValue, ProviderMetadata, ReasoningFileOutput,
    ReasoningOutput, Source, TextOutput, ToolCall, ToolResult,
};
use super::shared::*;

/// AI SDK V4 generated text content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4Text {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4TextMarker,
    /// Generated text.
    pub text: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_language_model_v4_provider_metadata",
        serialize_with = "serialize_optional_language_model_v4_provider_metadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelV4Text {
    /// Create generated text content.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            marker: LanguageModelV4TextMarker::Marker,
            text: text.into(),
            provider_metadata: None,
        }
    }

    /// Project a stable text output onto the V4 provider content shape.
    pub fn from_text_output(output: &TextOutput) -> Self {
        Self {
            marker: LanguageModelV4TextMarker::Marker,
            text: output.text.clone(),
            provider_metadata: language_model_v4_provider_metadata_from_stable(
                &output.provider_metadata,
            ),
        }
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK V4 content discriminator.
    pub const fn r#type(&self) -> &'static str {
        "text"
    }
}

impl From<TextOutput> for LanguageModelV4Text {
    fn from(value: TextOutput) -> Self {
        Self::from_text_output(&value)
    }
}

impl From<&TextOutput> for LanguageModelV4Text {
    fn from(value: &TextOutput) -> Self {
        Self::from_text_output(value)
    }
}

/// AI SDK V4 generated reasoning content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4Reasoning {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ReasoningMarker,
    /// Reasoning text.
    pub text: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_language_model_v4_provider_metadata",
        serialize_with = "serialize_optional_language_model_v4_provider_metadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelV4Reasoning {
    /// Create generated reasoning content.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            marker: LanguageModelV4ReasoningMarker::Marker,
            text: text.into(),
            provider_metadata: None,
        }
    }

    /// Project a stable reasoning output onto the V4 provider content shape.
    pub fn from_reasoning_output(output: &ReasoningOutput) -> Self {
        Self {
            marker: LanguageModelV4ReasoningMarker::Marker,
            text: output.text.clone(),
            provider_metadata: language_model_v4_provider_metadata_from_stable(
                &output.provider_metadata,
            ),
        }
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK V4 content discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning"
    }
}

impl From<ReasoningOutput> for LanguageModelV4Reasoning {
    fn from(value: ReasoningOutput) -> Self {
        Self::from_reasoning_output(&value)
    }
}

impl From<&ReasoningOutput> for LanguageModelV4Reasoning {
    fn from(value: &ReasoningOutput) -> Self {
        Self::from_reasoning_output(value)
    }
}

/// AI SDK V4 provider-specific content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4CustomContent {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4CustomMarker,
    /// Provider-specific custom content kind, in `{provider}.{provider-type}` format.
    #[serde(
        deserialize_with = "deserialize_language_model_v4_custom_kind",
        serialize_with = "serialize_language_model_v4_custom_kind"
    )]
    pub kind: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_language_model_v4_provider_metadata",
        serialize_with = "serialize_optional_language_model_v4_provider_metadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelV4CustomContent {
    /// Create a V4 generated custom content part.
    ///
    /// Use `try_new` when the kind comes from untrusted input and should be checked before
    /// serialization.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            marker: LanguageModelV4CustomMarker::Marker,
            kind: kind.into(),
            provider_metadata: None,
        }
    }

    /// Create a V4 generated custom content part and validate the AI SDK kind format.
    pub fn try_new(kind: impl Into<String>) -> Option<Self> {
        let content = Self::new(kind);
        is_language_model_v4_custom_kind(&content.kind).then_some(content)
    }

    /// Project a stable custom output onto the V4 provider content shape.
    pub fn try_from_custom_output(output: &CustomOutput) -> Option<Self> {
        Some(Self {
            marker: LanguageModelV4CustomMarker::Marker,
            kind: is_language_model_v4_custom_kind(&output.kind).then(|| output.kind.clone())?,
            provider_metadata: language_model_v4_provider_metadata_from_stable(
                &output.provider_metadata,
            ),
        })
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK V4 content discriminator.
    pub const fn r#type(&self) -> &'static str {
        "custom"
    }
}

/// AI SDK V4 generated source citation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4Source {
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
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_language_model_v4_provider_metadata",
        serialize_with = "serialize_optional_language_model_v4_provider_metadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelV4Source {
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

    /// Project a stable source onto the V4 provider content shape.
    pub fn from_source(source: &Source) -> Self {
        Self {
            kind: SourceMarker::Source,
            id: source.id.clone(),
            source: source.source.clone(),
            provider_metadata: language_model_v4_provider_metadata_from_stable(
                &source.provider_metadata,
            ),
        }
    }

    /// Return the AI SDK V4 source marker.
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

impl From<Source> for LanguageModelV4Source {
    fn from(value: Source) -> Self {
        Self::from_source(&value)
    }
}

impl From<&Source> for LanguageModelV4Source {
    fn from(value: &Source) -> Self {
        Self::from_source(value)
    }
}

impl From<LanguageModelV4Source> for Source {
    fn from(value: LanguageModelV4Source) -> Self {
        Self {
            kind: SourceMarker::Source,
            id: value.id,
            source: value.source,
            provider_metadata: value.provider_metadata,
        }
    }
}

/// AI SDK V4 generated file content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4File {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4FileMarker,
    /// IANA media type of the generated file.
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
    /// Generated file data as a base64/string payload or bytes.
    pub data: LanguageModelV4GeneratedFileData,
    /// Additional provider-specific metadata for the file part.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_language_model_v4_provider_metadata",
        serialize_with = "serialize_optional_language_model_v4_provider_metadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelV4File {
    /// Create generated file content.
    pub fn new(
        data: impl Into<LanguageModelV4GeneratedFileData>,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            marker: LanguageModelV4FileMarker::Marker,
            media_type: media_type.into(),
            data: data.into(),
            provider_metadata: None,
        }
    }

    /// Project a high-level generated file onto the model-facing V4 file shape.
    pub fn from_generated_file(file: GeneratedFile) -> Self {
        Self::new(file.base64, file.media_type)
    }

    /// Project a stable file output onto the V4 provider content shape.
    pub fn from_file_output(output: &FileOutput) -> Self {
        let mut content = Self::from_generated_file(output.file.clone());
        content.provider_metadata =
            language_model_v4_provider_metadata_from_stable(&output.provider_metadata);
        content
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK V4 content discriminator.
    pub const fn r#type(&self) -> &'static str {
        "file"
    }
}

impl From<FileOutput> for LanguageModelV4File {
    fn from(value: FileOutput) -> Self {
        Self::from_file_output(&value)
    }
}

impl From<&FileOutput> for LanguageModelV4File {
    fn from(value: &FileOutput) -> Self {
        Self::from_file_output(value)
    }
}

/// AI SDK V4 generated reasoning-file content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4ReasoningFile {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ReasoningFileMarker,
    /// IANA media type of the generated file.
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
    /// Generated file data as a base64/string payload or bytes.
    pub data: LanguageModelV4GeneratedFileData,
    /// Additional provider-specific metadata for the reasoning file part.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_language_model_v4_provider_metadata",
        serialize_with = "serialize_optional_language_model_v4_provider_metadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelV4ReasoningFile {
    /// Create generated reasoning-file content.
    pub fn new(
        data: impl Into<LanguageModelV4GeneratedFileData>,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            marker: LanguageModelV4ReasoningFileMarker::Marker,
            media_type: media_type.into(),
            data: data.into(),
            provider_metadata: None,
        }
    }

    /// Project a high-level generated file onto the model-facing V4 reasoning-file shape.
    pub fn from_generated_file(file: GeneratedFile) -> Self {
        Self::new(file.base64, file.media_type)
    }

    /// Project a stable reasoning-file output onto the V4 provider content shape.
    pub fn from_reasoning_file_output(output: &ReasoningFileOutput) -> Self {
        let mut content = Self::from_generated_file(output.file.clone());
        content.provider_metadata =
            language_model_v4_provider_metadata_from_stable(&output.provider_metadata);
        content
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK V4 content discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning-file"
    }
}

impl From<ReasoningFileOutput> for LanguageModelV4ReasoningFile {
    fn from(value: ReasoningFileOutput) -> Self {
        Self::from_reasoning_file_output(&value)
    }
}

impl From<&ReasoningFileOutput> for LanguageModelV4ReasoningFile {
    fn from(value: &ReasoningFileOutput) -> Self {
        Self::from_reasoning_file_output(value)
    }
}

/// AI SDK V4 generated tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4ToolCall {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ToolCallMarker,
    /// Unique tool-call id.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Tool name.
    #[serde(rename = "toolName")]
    pub tool_name: String,
    /// Stringified JSON object with tool call arguments.
    pub input: String,
    /// Whether the tool call will be executed by the provider.
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
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_language_model_v4_provider_metadata",
        serialize_with = "serialize_optional_language_model_v4_provider_metadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelV4ToolCall {
    /// Create a model-facing tool call from already stringified JSON arguments.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: impl Into<String>,
    ) -> Self {
        Self {
            marker: LanguageModelV4ToolCallMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input: input.into(),
            provider_executed: None,
            dynamic: None,
            provider_metadata: None,
        }
    }

    /// Create a model-facing tool call by stringifying JSON-serializable input.
    pub fn from_json_input(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: impl Serialize,
    ) -> Result<Self, serde_json::Error> {
        Ok(Self::new(
            tool_call_id,
            tool_name,
            serde_json::to_string(&input)?,
        ))
    }

    /// Project a stable tool call onto the V4 provider content shape.
    pub fn from_tool_call<NAME, INPUT>(
        tool_call: &ToolCall<NAME, INPUT>,
    ) -> Result<Self, serde_json::Error>
    where
        NAME: ToString,
        INPUT: Serialize,
    {
        let mut content = Self::from_json_input(
            tool_call.tool_call_id.clone(),
            tool_call.tool_name.to_string(),
            &tool_call.input,
        )?;
        content.provider_executed = tool_call.provider_executed;
        content.dynamic = tool_call.dynamic;
        content.provider_metadata =
            language_model_v4_provider_metadata_from_stable(&tool_call.provider_metadata);
        Ok(content)
    }

    /// Mark whether the tool call will be executed by the provider.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Mark whether the tool is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK V4 content discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-call"
    }
}

/// AI SDK V4 provider-executed tool result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4ToolResult {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ToolResultMarker,
    /// Tool-call id associated with this result.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Tool name.
    #[serde(rename = "toolName")]
    pub tool_name: String,
    /// JSON-serializable non-null result payload.
    #[serde(
        deserialize_with = "deserialize_ai_sdk_non_null_json_value",
        serialize_with = "serialize_ai_sdk_non_null_json_value"
    )]
    pub result: JSONValue,
    /// Whether the result represents an error.
    #[serde(
        rename = "isError",
        alias = "is_error",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub is_error: Option<bool>,
    /// Whether the result is preliminary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
    /// Whether the tool is dynamic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_language_model_v4_provider_metadata",
        serialize_with = "serialize_optional_language_model_v4_provider_metadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelV4ToolResult {
    /// Create a model-facing provider-executed tool result.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: impl Into<JSONValue>,
    ) -> Self {
        Self {
            marker: LanguageModelV4ToolResultMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            result: result.into(),
            is_error: None,
            preliminary: None,
            dynamic: None,
            provider_metadata: None,
        }
    }

    /// Project a stable provider-executed tool result onto the V4 provider content shape.
    pub fn from_tool_result<NAME, INPUT, OUTPUT>(
        tool_result: &ToolResult<NAME, INPUT, OUTPUT>,
    ) -> Option<Self>
    where
        NAME: ToString,
        OUTPUT: Clone + Into<ToolResultOutput>,
    {
        let output = tool_result.output.clone().into();
        let (result, is_error) = match output {
            ToolResultOutput::Json { value, .. } => (value, false),
            ToolResultOutput::Text { value, .. } => (serde_json::Value::String(value), false),
            ToolResultOutput::ErrorText { value, .. } => (serde_json::Value::String(value), true),
            ToolResultOutput::ErrorJson { value, .. } => (value, true),
            ToolResultOutput::Content { .. } | ToolResultOutput::ExecutionDenied { .. } => {
                return None;
            }
        };
        if result.is_null() {
            return None;
        }

        let mut content = Self::new(
            tool_result.tool_call_id.clone(),
            tool_result.tool_name.to_string(),
            result,
        );
        content.is_error = is_error.then_some(true);
        content.preliminary = tool_result.preliminary;
        content.dynamic = tool_result.dynamic;
        content.provider_metadata =
            language_model_v4_provider_metadata_from_stable(&tool_result.provider_metadata);
        Some(content)
    }

    /// Mark whether the result represents an error.
    pub const fn with_is_error(mut self, is_error: bool) -> Self {
        self.is_error = Some(is_error);
        self
    }

    /// Mark whether the result is preliminary.
    pub const fn with_preliminary(mut self, preliminary: bool) -> Self {
        self.preliminary = Some(preliminary);
        self
    }

    /// Mark whether the tool is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK V4 content discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-result"
    }
}

/// AI SDK V4 provider-emitted tool approval request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelV4ToolApprovalRequest {
    #[serde(rename = "type", default)]
    marker: LanguageModelV4ToolApprovalRequestMarker,
    /// Approval request id.
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    /// Tool-call id associated with this approval request.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_language_model_v4_provider_metadata",
        serialize_with = "serialize_optional_language_model_v4_provider_metadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl LanguageModelV4ToolApprovalRequest {
    /// Create a provider-emitted tool approval request.
    pub fn new(approval_id: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            marker: LanguageModelV4ToolApprovalRequestMarker::Marker,
            approval_id: approval_id.into(),
            tool_call_id: tool_call_id.into(),
            provider_metadata: None,
        }
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK V4 content discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-request"
    }
}

/// AI SDK V4 generated content union.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LanguageModelV4Content {
    Text(LanguageModelV4Text),
    Reasoning(LanguageModelV4Reasoning),
    Custom(LanguageModelV4CustomContent),
    ReasoningFile(LanguageModelV4ReasoningFile),
    File(LanguageModelV4File),
    ToolApprovalRequest(LanguageModelV4ToolApprovalRequest),
    Source(LanguageModelV4Source),
    ToolCall(LanguageModelV4ToolCall),
    ToolResult(LanguageModelV4ToolResult),
}

impl LanguageModelV4Content {
    /// Return the AI SDK V4 content discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Text(part) => part.r#type(),
            Self::Reasoning(part) => part.r#type(),
            Self::Custom(part) => part.r#type(),
            Self::ReasoningFile(part) => part.r#type(),
            Self::File(part) => part.r#type(),
            Self::ToolApprovalRequest(part) => part.r#type(),
            Self::Source(part) => part.r#type(),
            Self::ToolCall(part) => part.r#type(),
            Self::ToolResult(part) => part.r#type(),
        }
    }
}

impl From<LanguageModelV4Text> for LanguageModelV4Content {
    fn from(value: LanguageModelV4Text) -> Self {
        Self::Text(value)
    }
}

impl From<TextOutput> for LanguageModelV4Content {
    fn from(value: TextOutput) -> Self {
        Self::Text(LanguageModelV4Text::from(value))
    }
}

impl From<&TextOutput> for LanguageModelV4Content {
    fn from(value: &TextOutput) -> Self {
        Self::Text(LanguageModelV4Text::from(value))
    }
}

impl From<LanguageModelV4Reasoning> for LanguageModelV4Content {
    fn from(value: LanguageModelV4Reasoning) -> Self {
        Self::Reasoning(value)
    }
}

impl From<ReasoningOutput> for LanguageModelV4Content {
    fn from(value: ReasoningOutput) -> Self {
        Self::Reasoning(LanguageModelV4Reasoning::from(value))
    }
}

impl From<&ReasoningOutput> for LanguageModelV4Content {
    fn from(value: &ReasoningOutput) -> Self {
        Self::Reasoning(LanguageModelV4Reasoning::from(value))
    }
}

impl From<LanguageModelV4CustomContent> for LanguageModelV4Content {
    fn from(value: LanguageModelV4CustomContent) -> Self {
        Self::Custom(value)
    }
}

impl From<LanguageModelV4ReasoningFile> for LanguageModelV4Content {
    fn from(value: LanguageModelV4ReasoningFile) -> Self {
        Self::ReasoningFile(value)
    }
}

impl From<ReasoningFileOutput> for LanguageModelV4Content {
    fn from(value: ReasoningFileOutput) -> Self {
        Self::ReasoningFile(LanguageModelV4ReasoningFile::from(value))
    }
}

impl From<&ReasoningFileOutput> for LanguageModelV4Content {
    fn from(value: &ReasoningFileOutput) -> Self {
        Self::ReasoningFile(LanguageModelV4ReasoningFile::from(value))
    }
}

impl From<LanguageModelV4File> for LanguageModelV4Content {
    fn from(value: LanguageModelV4File) -> Self {
        Self::File(value)
    }
}

impl From<FileOutput> for LanguageModelV4Content {
    fn from(value: FileOutput) -> Self {
        Self::File(LanguageModelV4File::from(value))
    }
}

impl From<&FileOutput> for LanguageModelV4Content {
    fn from(value: &FileOutput) -> Self {
        Self::File(LanguageModelV4File::from(value))
    }
}

impl From<LanguageModelV4ToolApprovalRequest> for LanguageModelV4Content {
    fn from(value: LanguageModelV4ToolApprovalRequest) -> Self {
        Self::ToolApprovalRequest(value)
    }
}

impl From<LanguageModelV4Source> for LanguageModelV4Content {
    fn from(value: LanguageModelV4Source) -> Self {
        Self::Source(value)
    }
}

impl From<Source> for LanguageModelV4Content {
    fn from(value: Source) -> Self {
        Self::Source(LanguageModelV4Source::from(value))
    }
}

impl From<&Source> for LanguageModelV4Content {
    fn from(value: &Source) -> Self {
        Self::Source(LanguageModelV4Source::from(value))
    }
}

impl From<LanguageModelV4ToolCall> for LanguageModelV4Content {
    fn from(value: LanguageModelV4ToolCall) -> Self {
        Self::ToolCall(value)
    }
}

impl From<LanguageModelV4ToolResult> for LanguageModelV4Content {
    fn from(value: LanguageModelV4ToolResult) -> Self {
        Self::ToolResult(value)
    }
}
