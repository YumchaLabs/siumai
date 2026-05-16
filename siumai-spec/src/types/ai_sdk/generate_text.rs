use crate::types::{
    AssistantModelMessage, ChatResponse, ContentPart, DataContent, FilePartSource, FinishReason,
    MediaSource, MessageContent, ModelMessage, ProviderOptionsMap, StandardizedPrompt,
    SystemPrompt, Tool, ToolChoice, ToolModelMessage, ToolResultOutput,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use super::{
    CallWarning, CallbackModelInfo, Context, CustomOutput, FileOutput, GeneratedFile, JSONValue,
    LanguageModelCallOptions, LanguageModelRequestMetadata, LanguageModelResponseMetadata,
    LanguageModelUsage, ProviderMetadata, ProviderOptions, ReasoningFileOutput, ReasoningOutput,
    Source, StopCondition, TextOutput, TextStreamPart, TimeoutConfiguration,
    ToolApprovalRequestOutput, ToolApprovalResponseOutput, ToolCall, ToolError, ToolResult,
};

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

/// Conversion failure when projecting legacy stable response content into generated output parts.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum GenerateTextContentPartProjectionError {
    /// The message content shape is not part of the generated text output projection.
    #[error(
        "message content `{content_type}` is not supported in generated output projection: {reason}"
    )]
    UnsupportedMessageContent {
        content_type: &'static str,
        reason: &'static str,
    },
    /// The content part is not losslessly representable as generated output.
    #[error("content part `{part_type}` is not supported in generated output projection: {reason}")]
    UnsupportedContentPart {
        part_type: &'static str,
        reason: &'static str,
    },
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

/// Project a stable response content part into the canonical non-V4 generated output content.
///
/// This is a response-side projection. It preserves `providerMetadata`, intentionally ignores
/// request-side `providerOptions`, and fails when a legacy `ContentPart` would lose information.
pub fn project_response_content_part_to_generate_text_content_part(
    part: &ContentPart,
) -> Result<GenerateTextContentPart, GenerateTextContentPartProjectionError> {
    match part {
        ContentPart::Text {
            text,
            provider_metadata,
            ..
        } => {
            let mut output = TextOutput::new(text.clone());
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::Custom {
            kind,
            provider_metadata,
            ..
        } => {
            let mut output = CustomOutput::new(kind.clone());
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::Reasoning {
            text,
            provider_metadata,
            ..
        } => {
            let mut output = ReasoningOutput::new(text.clone());
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::ReasoningFile {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            let file = generated_file_from_media_source(source, media_type, "reasoning-file")?;
            let mut output = ReasoningFileOutput::new(file);
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::Source { .. } => Ok(Source::try_from(part)
            .map(GenerateTextContentPart::Source)
            .map_err(
                |_| GenerateTextContentPartProjectionError::UnsupportedContentPart {
                    part_type: "source",
                    reason: "source content could not be projected",
                },
            )?),
        ContentPart::File {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            let file = generated_file_from_file_source(source, media_type, "file")?;
            let mut output = FileOutput::new(file);
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::ToolCall {
            tool_call_id,
            tool_name,
            arguments,
            provider_executed,
            dynamic,
            invalid,
            error,
            title,
            provider_metadata,
            ..
        } => {
            let mut output =
                ToolCall::new(tool_call_id.clone(), tool_name.clone(), arguments.clone());
            if let Some(provider_executed) = provider_executed {
                output = output.with_provider_executed(*provider_executed);
            }
            if let Some(dynamic) = dynamic {
                output = output.with_dynamic(*dynamic);
            }
            if let Some(invalid) = invalid {
                output = output.with_invalid(*invalid);
            }
            if let Some(error) = error {
                output = output.with_error(error.clone());
            }
            if let Some(title) = title {
                output = output.with_title(title.clone());
            }
            if let Some(provider_metadata) = provider_metadata.clone() {
                output = output.with_provider_metadata(provider_metadata);
            }
            Ok(output.into())
        }
        ContentPart::ToolResult {
            tool_call_id,
            tool_name,
            output,
            input,
            provider_executed,
            dynamic,
            preliminary,
            title,
            provider_metadata,
            ..
        } => {
            let Some(input) = input.clone() else {
                return Err(
                    GenerateTextContentPartProjectionError::UnsupportedContentPart {
                        part_type: "tool-result",
                        reason: "tool-result generated output requires original input",
                    },
                );
            };
            let mut projected = ToolResult::new(
                tool_call_id.clone(),
                tool_name.clone(),
                input,
                output.clone(),
            );
            if let Some(provider_executed) = provider_executed {
                projected = projected.with_provider_executed(*provider_executed);
            }
            if let Some(dynamic) = dynamic {
                projected = projected.with_dynamic(*dynamic);
            }
            if let Some(preliminary) = preliminary {
                projected = projected.with_preliminary(*preliminary);
            }
            if let Some(title) = title {
                projected = projected.with_title(title.clone());
            }
            if let Some(provider_metadata) = provider_metadata.clone() {
                projected = projected.with_provider_metadata(provider_metadata);
            }
            Ok(projected.into())
        }
        ContentPart::Image { .. } => Err(unsupported_generated_output_part(
            "image",
            "image content is ambiguous in generated text output projection",
        )),
        ContentPart::Audio { .. } => Err(unsupported_generated_output_part(
            "audio",
            "audio content is ambiguous in generated text output projection",
        )),
        ContentPart::ToolApprovalRequest { .. } => Err(unsupported_generated_output_part(
            "tool-approval-request",
            "tool approval request output requires the original tool call",
        )),
        ContentPart::ToolApprovalResponse { .. } => Err(unsupported_generated_output_part(
            "tool-approval-response",
            "tool approval response output requires the original tool call",
        )),
    }
}

/// Project stable response content into canonical non-V4 generated output content parts.
pub fn project_response_content_to_generate_text_content_parts(
    content: &MessageContent,
) -> Result<Vec<GenerateTextContentPart>, GenerateTextContentPartProjectionError> {
    match content {
        MessageContent::Text(text) => Ok(vec![TextOutput::new(text.clone()).into()]),
        MessageContent::MultiModal(parts) => parts
            .iter()
            .map(project_response_content_part_to_generate_text_content_part)
            .collect(),
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(_) => Err(
            GenerateTextContentPartProjectionError::UnsupportedMessageContent {
                content_type: "json",
                reason: "structured JSON content is not generated text output",
            },
        ),
    }
}

/// Project a chat response into canonical non-V4 generated output content parts.
pub fn project_chat_response_to_generate_text_content_parts(
    response: &ChatResponse,
) -> Result<Vec<GenerateTextContentPart>, GenerateTextContentPartProjectionError> {
    project_response_content_to_generate_text_content_parts(&response.content)
}

fn generated_file_from_media_source(
    source: &MediaSource,
    media_type: &str,
    part_type: &'static str,
) -> Result<GeneratedFile, GenerateTextContentPartProjectionError> {
    source.as_base64().map_or_else(
        || {
            Err(
                GenerateTextContentPartProjectionError::UnsupportedContentPart {
                    part_type,
                    reason: "generated file output requires base64 or binary data",
                },
            )
        },
        |base64| Ok(GeneratedFile::from_base64(base64, media_type)),
    )
}

fn generated_file_from_file_source(
    source: &FilePartSource,
    media_type: &str,
    part_type: &'static str,
) -> Result<GeneratedFile, GenerateTextContentPartProjectionError> {
    source.as_base64().map_or_else(
        || {
            Err(
                GenerateTextContentPartProjectionError::UnsupportedContentPart {
                    part_type,
                    reason: "generated file output requires base64 or binary data",
                },
            )
        },
        |base64| Ok(GeneratedFile::from_base64(base64, media_type)),
    )
}

fn unsupported_generated_output_part(
    part_type: &'static str,
    reason: &'static str,
) -> GenerateTextContentPartProjectionError {
    GenerateTextContentPartProjectionError::UnsupportedContentPart { part_type, reason }
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

/// AI SDK `StepResult` export. Rust keeps the same passive carrier as `GenerateTextStepResult`.
pub type StepResult<NAME = String, INPUT = JSONValue, ToolOutput = ToolResultOutput> =
    GenerateTextStepResult<NAME, INPUT, ToolOutput>;

/// AI SDK `DefaultStepResult` export. Rust keeps the same passive carrier as `StepResult`.
pub type DefaultStepResult<NAME = String, INPUT = JSONValue, ToolOutput = ToolResultOutput> =
    GenerateTextStepResult<NAME, INPUT, ToolOutput>;

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

/// Passive options passed to an AI SDK `PrepareStepFunction`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PrepareStepOptions<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// Steps that have already completed.
    pub steps: Vec<GenerateTextStepResult<NAME, INPUT, OUTPUT>>,
    /// Zero-based step index that is about to run.
    pub step_number: u32,
    /// Model selected for the step.
    pub model: CallbackModelInfo,
    /// Messages that will be sent for the current step.
    pub messages: Vec<ModelMessage>,
    /// Tool context snapshot.
    pub tools_context: Context,
    /// Runtime context snapshot.
    pub runtime_context: Context,
}

/// Passive result returned by an AI SDK `PrepareStepFunction`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PrepareStepResult {
    /// Optional model override for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<CallbackModelInfo>,
    /// Optional tool choice override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Optional active tool list override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_tools: Option<Vec<String>>,
    /// Optional system prompt override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    /// Optional full message list override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<ModelMessage>>,
    /// Optional tool context override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_context: Option<Context>,
    /// Optional runtime context override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_context: Option<Context>,
    /// Optional provider options override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
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
#[allow(clippy::large_enum_variant)]
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
