//! AI SDK-style prompt/message shared surface.
//!
//! These types mirror the higher-level `packages/ai/src/prompt/*` and
//! `packages/provider-utils/src/types/*model-message*` contracts. They are intentionally narrower
//! than Siumai's stable `ChatMessage` / `ContentPart` model: response-side metadata, developer
//! messages, audio/source parts, image detail, and various provider/runtime-specific tool
//! extensions remain outside this shared prompt contract.

use super::{
    ChatMessage, ChatRequest, ContentPart, FilePartSource, MediaSource, MessageContent,
    MessageMetadata, MessageRole, ProviderMetadataMap, ProviderOptionsMap, ToolResultOutput,
};
use base64::Engine;
use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

/// AI SDK-style inline data content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum DataContent {
    /// Base64-encoded data.
    Base64(String),
    /// Binary data.
    Binary(Vec<u8>),
}

impl DataContent {
    pub fn binary(data: impl Into<Vec<u8>>) -> Self {
        Self::Binary(data.into())
    }

    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64(data.into())
    }

    pub fn as_base64(&self) -> String {
        match self {
            Self::Base64(data) => data.clone(),
            Self::Binary(data) => base64::engine::general_purpose::STANDARD.encode(data),
        }
    }

    pub fn as_bytes(&self) -> Result<Vec<u8>, base64::DecodeError> {
        match self {
            Self::Base64(data) => base64::engine::general_purpose::STANDARD.decode(data),
            Self::Binary(data) => Ok(data.clone()),
        }
    }
}

/// Convert AI SDK-style data content into a base64 string.
pub fn convert_data_content_to_base64_string(content: &DataContent) -> String {
    content.as_base64()
}

impl From<&DataContent> for MediaSource {
    fn from(value: &DataContent) -> Self {
        match value {
            DataContent::Base64(data) => Self::base64(data.clone()),
            DataContent::Binary(data) => Self::binary(data.clone()),
        }
    }
}

impl From<DataContent> for MediaSource {
    fn from(value: DataContent) -> Self {
        Self::from(&value)
    }
}

impl From<&DataContent> for FilePartSource {
    fn from(value: &DataContent) -> Self {
        Self::Media(MediaSource::from(value))
    }
}

impl From<DataContent> for FilePartSource {
    fn from(value: DataContent) -> Self {
        Self::from(&value)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelMessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum PromptContentPartType {
    Text,
    Image,
    File,
    Reasoning,
    Custom,
    ReasoningFile,
    ToolCall,
    ToolResult,
    ToolApprovalRequest,
    ToolApprovalResponse,
}

const fn system_model_message_role() -> ModelMessageRole {
    ModelMessageRole::System
}

const fn user_model_message_role() -> ModelMessageRole {
    ModelMessageRole::User
}

const fn assistant_model_message_role() -> ModelMessageRole {
    ModelMessageRole::Assistant
}

const fn tool_model_message_role() -> ModelMessageRole {
    ModelMessageRole::Tool
}

const fn text_part_type() -> PromptContentPartType {
    PromptContentPartType::Text
}

const fn image_part_type() -> PromptContentPartType {
    PromptContentPartType::Image
}

const fn file_part_type() -> PromptContentPartType {
    PromptContentPartType::File
}

const fn reasoning_part_type() -> PromptContentPartType {
    PromptContentPartType::Reasoning
}

const fn custom_part_type() -> PromptContentPartType {
    PromptContentPartType::Custom
}

const fn reasoning_file_part_type() -> PromptContentPartType {
    PromptContentPartType::ReasoningFile
}

const fn tool_call_part_type() -> PromptContentPartType {
    PromptContentPartType::ToolCall
}

const fn tool_result_part_type() -> PromptContentPartType {
    PromptContentPartType::ToolResult
}

const fn tool_approval_request_part_type() -> PromptContentPartType {
    PromptContentPartType::ToolApprovalRequest
}

const fn tool_approval_response_part_type() -> PromptContentPartType {
    PromptContentPartType::ToolApprovalResponse
}

fn deserialize_exact_prompt_content_part_type<'de, D>(
    deserializer: D,
    expected: PromptContentPartType,
) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    let actual = PromptContentPartType::deserialize(deserializer)?;
    if actual == expected {
        Ok(actual)
    } else {
        Err(serde::de::Error::custom(format!(
            "expected prompt content part type `{}`",
            prompt_content_part_type_name(expected)
        )))
    }
}

fn deserialize_exact_model_message_role<'de, D>(
    deserializer: D,
    expected: ModelMessageRole,
) -> Result<ModelMessageRole, D::Error>
where
    D: Deserializer<'de>,
{
    let actual = ModelMessageRole::deserialize(deserializer)?;
    if actual == expected {
        Ok(actual)
    } else {
        Err(serde::de::Error::custom(format!(
            "expected model message role `{}`",
            model_message_role_name(expected)
        )))
    }
}

fn deserialize_text_part_type<'de, D>(deserializer: D) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, text_part_type())
}

fn deserialize_image_part_type<'de, D>(deserializer: D) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, image_part_type())
}

fn deserialize_file_part_type<'de, D>(deserializer: D) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, file_part_type())
}

fn deserialize_reasoning_part_type<'de, D>(
    deserializer: D,
) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, reasoning_part_type())
}

fn deserialize_custom_part_type<'de, D>(deserializer: D) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, custom_part_type())
}

fn deserialize_reasoning_file_part_type<'de, D>(
    deserializer: D,
) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, reasoning_file_part_type())
}

fn deserialize_tool_call_part_type<'de, D>(
    deserializer: D,
) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, tool_call_part_type())
}

fn deserialize_tool_result_part_type<'de, D>(
    deserializer: D,
) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, tool_result_part_type())
}

fn deserialize_tool_approval_request_part_type<'de, D>(
    deserializer: D,
) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, tool_approval_request_part_type())
}

fn deserialize_tool_approval_response_part_type<'de, D>(
    deserializer: D,
) -> Result<PromptContentPartType, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_prompt_content_part_type(deserializer, tool_approval_response_part_type())
}

fn deserialize_system_model_message_role<'de, D>(
    deserializer: D,
) -> Result<ModelMessageRole, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_model_message_role(deserializer, system_model_message_role())
}

fn deserialize_user_model_message_role<'de, D>(
    deserializer: D,
) -> Result<ModelMessageRole, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_model_message_role(deserializer, user_model_message_role())
}

fn deserialize_assistant_model_message_role<'de, D>(
    deserializer: D,
) -> Result<ModelMessageRole, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_model_message_role(deserializer, assistant_model_message_role())
}

fn deserialize_tool_model_message_role<'de, D>(
    deserializer: D,
) -> Result<ModelMessageRole, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_exact_model_message_role(deserializer, tool_model_message_role())
}

fn prompt_content_part_type_name(part_type: PromptContentPartType) -> &'static str {
    match part_type {
        PromptContentPartType::Text => "text",
        PromptContentPartType::Image => "image",
        PromptContentPartType::File => "file",
        PromptContentPartType::Reasoning => "reasoning",
        PromptContentPartType::Custom => "custom",
        PromptContentPartType::ReasoningFile => "reasoning-file",
        PromptContentPartType::ToolCall => "tool-call",
        PromptContentPartType::ToolResult => "tool-result",
        PromptContentPartType::ToolApprovalRequest => "tool-approval-request",
        PromptContentPartType::ToolApprovalResponse => "tool-approval-response",
    }
}

fn model_message_role_name(role: ModelMessageRole) -> &'static str {
    match role {
        ModelMessageRole::System => "system",
        ModelMessageRole::User => "user",
        ModelMessageRole::Assistant => "assistant",
        ModelMessageRole::Tool => "tool",
    }
}

/// Conversion failure when narrowing Siumai's richer chat model into the AI SDK prompt contract.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ModelMessageConversionError {
    /// The message role is outside the AI SDK prompt contract.
    #[error("message role `{role}` is not supported by the AI SDK prompt model")]
    UnsupportedMessageRole { role: String },

    /// The message content shape is incompatible with the target message role.
    #[error("message role `{role}` cannot be represented by the AI SDK prompt model: {reason}")]
    UnsupportedMessageContent { role: String, reason: &'static str },

    /// A content part falls outside the narrower AI SDK prompt content contract.
    #[error("content part `{part_type}` is not supported in `{context}` content: {reason}")]
    UnsupportedContentPart {
        context: &'static str,
        part_type: &'static str,
        reason: &'static str,
    },
}

/// Validation failure for AI SDK-style prompt input.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum PromptValidationError {
    /// Neither `prompt` nor `messages` was supplied.
    #[error("prompt or messages must be defined")]
    MissingPromptOrMessages,
    /// Both `prompt` and `messages` were supplied at once.
    #[error("prompt and messages cannot be defined at the same time")]
    BothPromptAndMessages,
    /// The normalized prompt would produce an empty message list.
    #[error("messages must not be empty")]
    EmptyMessages,
}

/// AI SDK-style text content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextPart {
    #[serde(rename = "type", deserialize_with = "deserialize_text_part_type")]
    part_type: PromptContentPartType,
    pub text: String,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl TextPart {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            part_type: text_part_type(),
            text: text.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style image content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImagePart {
    #[serde(rename = "type", deserialize_with = "deserialize_image_part_type")]
    part_type: PromptContentPartType,
    pub image: FilePartSource,
    #[serde(
        rename = "mediaType",
        alias = "media_type",
        skip_serializing_if = "Option::is_none"
    )]
    pub media_type: Option<String>,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl ImagePart {
    pub fn new(image: FilePartSource) -> Self {
        Self {
            part_type: image_part_type(),
            image,
            media_type: None,
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style file content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilePart {
    #[serde(rename = "type", deserialize_with = "deserialize_file_part_type")]
    part_type: PromptContentPartType,
    pub data: FilePartSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl FilePart {
    pub fn new(data: FilePartSource, media_type: impl Into<String>) -> Self {
        Self {
            part_type: file_part_type(),
            data,
            filename: None,
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style reasoning content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReasoningPart {
    #[serde(rename = "type", deserialize_with = "deserialize_reasoning_part_type")]
    part_type: PromptContentPartType,
    pub text: String,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl ReasoningPart {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            part_type: reasoning_part_type(),
            text: text.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style custom content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CustomPart {
    #[serde(rename = "type", deserialize_with = "deserialize_custom_part_type")]
    part_type: PromptContentPartType,
    pub kind: String,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl CustomPart {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            part_type: custom_part_type(),
            kind: kind.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style reasoning-file content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReasoningFilePart {
    #[serde(
        rename = "type",
        deserialize_with = "deserialize_reasoning_file_part_type"
    )]
    part_type: PromptContentPartType,
    pub data: MediaSource,
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl ReasoningFilePart {
    pub fn new(data: MediaSource, media_type: impl Into<String>) -> Self {
        Self {
            part_type: reasoning_file_part_type(),
            data,
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style tool-call content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallPart {
    #[serde(rename = "type", deserialize_with = "deserialize_tool_call_part_type")]
    part_type: PromptContentPartType,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    pub input: serde_json::Value,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
}

impl ToolCallPart {
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        Self {
            part_type: tool_call_part_type(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input,
            provider_options: ProviderOptionsMap::default(),
            provider_executed: None,
        }
    }
}

/// AI SDK-style tool-result content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResultPart {
    #[serde(
        rename = "type",
        deserialize_with = "deserialize_tool_result_part_type"
    )]
    part_type: PromptContentPartType,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    pub output: ToolResultOutput,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl ToolResultPart {
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        output: ToolResultOutput,
    ) -> Self {
        Self {
            part_type: tool_result_part_type(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output,
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style tool-approval-request content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolApprovalRequest {
    #[serde(
        rename = "type",
        deserialize_with = "deserialize_tool_approval_request_part_type"
    )]
    part_type: PromptContentPartType,
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
}

impl ToolApprovalRequest {
    pub fn new(approval_id: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            part_type: tool_approval_request_part_type(),
            approval_id: approval_id.into(),
            tool_call_id: tool_call_id.into(),
        }
    }
}

/// AI SDK-style tool-approval-response content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolApprovalResponse {
    #[serde(
        rename = "type",
        deserialize_with = "deserialize_tool_approval_response_part_type"
    )]
    part_type: PromptContentPartType,
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    pub approved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
}

impl ToolApprovalResponse {
    pub fn new(approval_id: impl Into<String>, approved: bool) -> Self {
        Self {
            part_type: tool_approval_response_part_type(),
            approval_id: approval_id.into(),
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
    pub fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UserContentPart {
    Text(TextPart),
    Image(ImagePart),
    File(FilePart),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum AssistantContentPart {
    Text(TextPart),
    Custom(CustomPart),
    File(FilePart),
    Reasoning(ReasoningPart),
    ReasoningFile(ReasoningFilePart),
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
    ToolApprovalRequest(ToolApprovalRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolContentPart {
    ToolResult(ToolResultPart),
    ToolApprovalResponse(ToolApprovalResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Parts(Vec<UserContentPart>),
}

impl UserContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    pub fn parts(parts: Vec<UserContentPart>) -> Self {
        Self::Parts(parts)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum AssistantContent {
    Text(String),
    Parts(Vec<AssistantContentPart>),
}

impl AssistantContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    pub fn parts(parts: Vec<AssistantContentPart>) -> Self {
        Self::Parts(parts)
    }
}

pub type ToolContent = Vec<ToolContentPart>;

/// AI SDK-style system model message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SystemModelMessage {
    #[serde(deserialize_with = "deserialize_system_model_message_role")]
    role: ModelMessageRole,
    pub content: String,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl SystemModelMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: system_model_message_role(),
            content: content.into(),
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style user model message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UserModelMessage {
    #[serde(deserialize_with = "deserialize_user_model_message_role")]
    role: ModelMessageRole,
    pub content: UserContent,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl UserModelMessage {
    pub fn new(content: UserContent) -> Self {
        Self {
            role: user_model_message_role(),
            content,
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style assistant model message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AssistantModelMessage {
    #[serde(deserialize_with = "deserialize_assistant_model_message_role")]
    role: ModelMessageRole,
    pub content: AssistantContent,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl AssistantModelMessage {
    pub fn new(content: AssistantContent) -> Self {
        Self {
            role: assistant_model_message_role(),
            content,
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style tool model message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolModelMessage {
    #[serde(deserialize_with = "deserialize_tool_model_message_role")]
    role: ModelMessageRole,
    pub content: ToolContent,
    #[serde(
        rename = "providerOptions",
        alias = "provider_options",
        default,
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options: ProviderOptionsMap,
}

impl ToolModelMessage {
    pub fn new(content: ToolContent) -> Self {
        Self {
            role: tool_model_message_role(),
            content,
            provider_options: ProviderOptionsMap::default(),
        }
    }
}

/// AI SDK-style model message union.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ModelMessage {
    System(SystemModelMessage),
    User(UserModelMessage),
    Assistant(AssistantModelMessage),
    Tool(ToolModelMessage),
}

impl ModelMessage {
    pub fn role(&self) -> ModelMessageRole {
        match self {
            Self::System(message) => message.role,
            Self::User(message) => message.role,
            Self::Assistant(message) => message.role,
            Self::Tool(message) => message.role,
        }
    }
}

/// AI SDK-style `prompt` field union.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PromptInput {
    Text(String),
    Messages(Vec<ModelMessage>),
}

/// AI SDK-style `system` field union.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Message(SystemModelMessage),
    Messages(Vec<SystemModelMessage>),
}

/// AI SDK-style prompt object.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Prompt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<ModelMessage>>,
}

impl Prompt {
    pub fn prompt_text(prompt: impl Into<String>) -> Self {
        Self {
            system: None,
            prompt: Some(PromptInput::Text(prompt.into())),
            messages: None,
        }
    }

    pub fn prompt_messages(prompt: Vec<ModelMessage>) -> Self {
        Self {
            system: None,
            prompt: Some(PromptInput::Messages(prompt)),
            messages: None,
        }
    }

    pub fn messages(messages: Vec<ModelMessage>) -> Self {
        Self {
            system: None,
            prompt: None,
            messages: Some(messages),
        }
    }

    pub fn with_system_text(mut self, system: impl Into<String>) -> Self {
        self.system = Some(SystemPrompt::Text(system.into()));
        self
    }

    pub fn with_system_message(mut self, system: SystemModelMessage) -> Self {
        self.system = Some(SystemPrompt::Message(system));
        self
    }

    pub fn with_system_messages(mut self, system: Vec<SystemModelMessage>) -> Self {
        self.system = Some(SystemPrompt::Messages(system));
        self
    }

    pub fn standardize(&self) -> Result<StandardizedPrompt, PromptValidationError> {
        let messages = match (&self.prompt, &self.messages) {
            (None, None) => return Err(PromptValidationError::MissingPromptOrMessages),
            (Some(_), Some(_)) => return Err(PromptValidationError::BothPromptAndMessages),
            (Some(PromptInput::Text(text)), None) => {
                vec![ModelMessage::User(UserModelMessage::new(
                    UserContent::Text(text.clone()),
                ))]
            }
            (Some(PromptInput::Messages(messages)), None) => messages.clone(),
            (None, Some(messages)) => messages.clone(),
        };

        if messages.is_empty() {
            return Err(PromptValidationError::EmptyMessages);
        }

        Ok(StandardizedPrompt {
            system: self.system.clone(),
            messages,
        })
    }

    pub fn to_chat_messages(&self) -> Result<Vec<ChatMessage>, PromptValidationError> {
        self.standardize().map(|prompt| prompt.to_chat_messages())
    }
}

/// Standardized prompt shape used after validating `Prompt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StandardizedPrompt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    pub messages: Vec<ModelMessage>,
}

impl StandardizedPrompt {
    pub fn to_chat_messages(&self) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        if let Some(system) = &self.system {
            match system {
                SystemPrompt::Text(text) => {
                    messages.push(ChatMessage::from(&SystemModelMessage::new(text.clone())));
                }
                SystemPrompt::Message(message) => messages.push(ChatMessage::from(message)),
                SystemPrompt::Messages(system_messages) => {
                    messages.extend(system_messages.iter().map(ChatMessage::from));
                }
            }
        }

        messages.extend(self.messages.iter().map(ChatMessage::from));
        messages
    }
}

impl TryFrom<&ChatRequest> for Prompt {
    type Error = ModelMessageConversionError;

    fn try_from(value: &ChatRequest) -> Result<Self, Self::Error> {
        let messages = value
            .messages
            .iter()
            .map(ModelMessage::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::messages(messages))
    }
}

impl TryFrom<ChatRequest> for Prompt {
    type Error = ModelMessageConversionError;

    fn try_from(value: ChatRequest) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&Prompt> for ChatRequest {
    type Error = PromptValidationError;

    fn try_from(value: &Prompt) -> Result<Self, Self::Error> {
        Ok(ChatRequest::new(value.to_chat_messages()?))
    }
}

impl TryFrom<Prompt> for ChatRequest {
    type Error = PromptValidationError;

    fn try_from(value: Prompt) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&ChatMessage> for ModelMessage {
    type Error = ModelMessageConversionError;

    fn try_from(value: &ChatMessage) -> Result<Self, Self::Error> {
        match value.role {
            MessageRole::System => Ok(Self::System(SystemModelMessage {
                role: system_model_message_role(),
                content: system_text_content(&value.content)?,
                provider_options: value.provider_options.clone(),
            })),
            MessageRole::User => Ok(Self::User(UserModelMessage {
                role: user_model_message_role(),
                content: user_content_from_message(value)?,
                provider_options: value.provider_options.clone(),
            })),
            MessageRole::Assistant => Ok(Self::Assistant(AssistantModelMessage {
                role: assistant_model_message_role(),
                content: assistant_content_from_message(value)?,
                provider_options: value.provider_options.clone(),
            })),
            MessageRole::Tool => Ok(Self::Tool(ToolModelMessage {
                role: tool_model_message_role(),
                content: tool_content_from_message(value)?,
                provider_options: value.provider_options.clone(),
            })),
            MessageRole::Developer => Err(ModelMessageConversionError::UnsupportedMessageRole {
                role: message_role_name(&value.role).to_string(),
            }),
        }
    }
}

impl TryFrom<ChatMessage> for ModelMessage {
    type Error = ModelMessageConversionError;

    fn try_from(value: ChatMessage) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl From<&SystemModelMessage> for ChatMessage {
    fn from(value: &SystemModelMessage) -> Self {
        ChatMessage {
            role: MessageRole::System,
            content: MessageContent::Text(value.content.clone()),
            provider_options: value.provider_options.clone(),
            metadata: MessageMetadata::default(),
        }
    }
}

impl From<SystemModelMessage> for ChatMessage {
    fn from(value: SystemModelMessage) -> Self {
        Self::from(&value)
    }
}

impl From<&UserModelMessage> for ChatMessage {
    fn from(value: &UserModelMessage) -> Self {
        ChatMessage {
            role: MessageRole::User,
            content: message_content_from_user_content(&value.content),
            provider_options: value.provider_options.clone(),
            metadata: MessageMetadata::default(),
        }
    }
}

impl From<UserModelMessage> for ChatMessage {
    fn from(value: UserModelMessage) -> Self {
        Self::from(&value)
    }
}

impl From<&AssistantModelMessage> for ChatMessage {
    fn from(value: &AssistantModelMessage) -> Self {
        ChatMessage {
            role: MessageRole::Assistant,
            content: message_content_from_assistant_content(&value.content),
            provider_options: value.provider_options.clone(),
            metadata: MessageMetadata::default(),
        }
    }
}

impl From<AssistantModelMessage> for ChatMessage {
    fn from(value: AssistantModelMessage) -> Self {
        Self::from(&value)
    }
}

impl From<&ToolModelMessage> for ChatMessage {
    fn from(value: &ToolModelMessage) -> Self {
        ChatMessage {
            role: MessageRole::Tool,
            content: MessageContent::MultiModal(
                value.content.iter().map(ContentPart::from).collect(),
            ),
            provider_options: value.provider_options.clone(),
            metadata: MessageMetadata::default(),
        }
    }
}

impl From<ToolModelMessage> for ChatMessage {
    fn from(value: ToolModelMessage) -> Self {
        Self::from(&value)
    }
}

impl From<&ModelMessage> for ChatMessage {
    fn from(value: &ModelMessage) -> Self {
        match value {
            ModelMessage::System(message) => ChatMessage::from(message),
            ModelMessage::User(message) => ChatMessage::from(message),
            ModelMessage::Assistant(message) => ChatMessage::from(message),
            ModelMessage::Tool(message) => ChatMessage::from(message),
        }
    }
}

impl From<ModelMessage> for ChatMessage {
    fn from(value: ModelMessage) -> Self {
        Self::from(&value)
    }
}

impl From<&UserContentPart> for ContentPart {
    fn from(value: &UserContentPart) -> Self {
        match value {
            UserContentPart::Text(part) => ContentPart::Text {
                text: part.text.clone(),
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            UserContentPart::Image(part) => ContentPart::Image {
                source: part.image.clone(),
                detail: None,
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            UserContentPart::File(part) => ContentPart::File {
                source: part.data.clone(),
                media_type: part.media_type.clone(),
                filename: part.filename.clone(),
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
        }
    }
}

impl From<UserContentPart> for ContentPart {
    fn from(value: UserContentPart) -> Self {
        Self::from(&value)
    }
}

impl From<&AssistantContentPart> for ContentPart {
    fn from(value: &AssistantContentPart) -> Self {
        match value {
            AssistantContentPart::Text(part) => ContentPart::Text {
                text: part.text.clone(),
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            AssistantContentPart::Custom(part) => ContentPart::Custom {
                kind: part.kind.clone(),
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            AssistantContentPart::File(part) => ContentPart::File {
                source: part.data.clone(),
                media_type: part.media_type.clone(),
                filename: part.filename.clone(),
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            AssistantContentPart::Reasoning(part) => ContentPart::Reasoning {
                text: part.text.clone(),
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            AssistantContentPart::ReasoningFile(part) => ContentPart::ReasoningFile {
                source: part.data.clone(),
                media_type: part.media_type.clone(),
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            AssistantContentPart::ToolCall(part) => ContentPart::ToolCall {
                tool_call_id: part.tool_call_id.clone(),
                tool_name: part.tool_name.clone(),
                arguments: part.input.clone(),
                provider_executed: part.provider_executed,
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            AssistantContentPart::ToolResult(part) => ContentPart::ToolResult {
                tool_call_id: part.tool_call_id.clone(),
                tool_name: part.tool_name.clone(),
                output: part.output.clone(),
                input: None,
                provider_executed: None,
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            AssistantContentPart::ToolApprovalRequest(part) => ContentPart::ToolApprovalRequest {
                approval_id: part.approval_id.clone(),
                tool_call_id: part.tool_call_id.clone(),
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        }
    }
}

impl From<AssistantContentPart> for ContentPart {
    fn from(value: AssistantContentPart) -> Self {
        Self::from(&value)
    }
}

impl From<&ToolContentPart> for ContentPart {
    fn from(value: &ToolContentPart) -> Self {
        match value {
            ToolContentPart::ToolResult(part) => ContentPart::ToolResult {
                tool_call_id: part.tool_call_id.clone(),
                tool_name: part.tool_name.clone(),
                output: part.output.clone(),
                input: None,
                provider_executed: None,
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: part.provider_options.clone(),
                provider_metadata: None,
            },
            ToolContentPart::ToolApprovalResponse(part) => ContentPart::ToolApprovalResponse {
                approval_id: part.approval_id.clone(),
                approved: part.approved,
                reason: part.reason.clone(),
                provider_executed: part.provider_executed,
                provider_options: ProviderOptionsMap::default(),
            },
        }
    }
}

impl From<ToolContentPart> for ContentPart {
    fn from(value: ToolContentPart) -> Self {
        Self::from(&value)
    }
}

fn message_content_from_user_content(content: &UserContent) -> MessageContent {
    match content {
        UserContent::Text(text) => MessageContent::Text(text.clone()),
        UserContent::Parts(parts) => {
            MessageContent::MultiModal(parts.iter().map(ContentPart::from).collect())
        }
    }
}

fn message_content_from_assistant_content(content: &AssistantContent) -> MessageContent {
    match content {
        AssistantContent::Text(text) => MessageContent::Text(text.clone()),
        AssistantContent::Parts(parts) => {
            MessageContent::MultiModal(parts.iter().map(ContentPart::from).collect())
        }
    }
}

fn system_text_content(content: &MessageContent) -> Result<String, ModelMessageConversionError> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::MultiModal(_) => {
            Err(ModelMessageConversionError::UnsupportedMessageContent {
                role: "system".to_string(),
                reason: "system messages must contain plain text",
            })
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(_) => Err(ModelMessageConversionError::UnsupportedMessageContent {
            role: "system".to_string(),
            reason: "structured JSON content is not part of the AI SDK prompt model",
        }),
    }
}

fn user_content_from_message(
    message: &ChatMessage,
) -> Result<UserContent, ModelMessageConversionError> {
    match &message.content {
        MessageContent::Text(text) => Ok(UserContent::Text(text.clone())),
        MessageContent::MultiModal(parts) => Ok(UserContent::Parts(
            parts
                .iter()
                .map(user_content_part_from_content_part)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(_) => Err(ModelMessageConversionError::UnsupportedMessageContent {
            role: "user".to_string(),
            reason: "structured JSON content is not part of the AI SDK prompt model",
        }),
    }
}

fn assistant_content_from_message(
    message: &ChatMessage,
) -> Result<AssistantContent, ModelMessageConversionError> {
    match &message.content {
        MessageContent::Text(text) => Ok(AssistantContent::Text(text.clone())),
        MessageContent::MultiModal(parts) => Ok(AssistantContent::Parts(
            parts
                .iter()
                .map(assistant_content_part_from_content_part)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(_) => Err(ModelMessageConversionError::UnsupportedMessageContent {
            role: "assistant".to_string(),
            reason: "structured JSON content is not part of the AI SDK prompt model",
        }),
    }
}

fn tool_content_from_message(
    message: &ChatMessage,
) -> Result<ToolContent, ModelMessageConversionError> {
    match &message.content {
        MessageContent::Text(_) => Err(ModelMessageConversionError::UnsupportedMessageContent {
            role: "tool".to_string(),
            reason: "tool messages must contain tool result or tool approval response parts",
        }),
        MessageContent::MultiModal(parts) => parts
            .iter()
            .map(tool_content_part_from_content_part)
            .collect::<Result<Vec<_>, _>>(),
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(_) => Err(ModelMessageConversionError::UnsupportedMessageContent {
            role: "tool".to_string(),
            reason: "structured JSON content is not part of the AI SDK prompt model",
        }),
    }
}

fn user_content_part_from_content_part(
    part: &ContentPart,
) -> Result<UserContentPart, ModelMessageConversionError> {
    match part {
        ContentPart::Text { .. } => Ok(UserContentPart::Text(text_part_from_content_part(part)?)),
        ContentPart::Image { .. } => {
            Ok(UserContentPart::Image(image_part_from_content_part(part)?))
        }
        ContentPart::File { .. } => Ok(UserContentPart::File(file_part_from_content_part(part)?)),
        _ => Err(unsupported_part(
            "user",
            part,
            "only text, image, and file parts are allowed",
        )),
    }
}

fn assistant_content_part_from_content_part(
    part: &ContentPart,
) -> Result<AssistantContentPart, ModelMessageConversionError> {
    match part {
        ContentPart::Text { .. } => Ok(AssistantContentPart::Text(text_part_from_content_part(
            part,
        )?)),
        ContentPart::Custom { .. } => Ok(AssistantContentPart::Custom(
            custom_part_from_content_part(part)?,
        )),
        ContentPart::File { .. } => Ok(AssistantContentPart::File(file_part_from_content_part(
            part,
        )?)),
        ContentPart::Reasoning { .. } => Ok(AssistantContentPart::Reasoning(
            reasoning_part_from_content_part(part)?,
        )),
        ContentPart::ReasoningFile { .. } => Ok(AssistantContentPart::ReasoningFile(
            reasoning_file_part_from_content_part(part)?,
        )),
        ContentPart::ToolCall { .. } => Ok(AssistantContentPart::ToolCall(
            tool_call_part_from_content_part(part)?,
        )),
        ContentPart::ToolResult { .. } => Ok(AssistantContentPart::ToolResult(
            tool_result_part_from_content_part(part)?,
        )),
        ContentPart::ToolApprovalRequest { .. } => Ok(AssistantContentPart::ToolApprovalRequest(
            tool_approval_request_from_content_part(part)?,
        )),
        _ => Err(unsupported_part(
            "assistant",
            part,
            "only text, custom, file, reasoning, reasoning-file, tool-call, tool-result, and tool-approval-request parts are allowed",
        )),
    }
}

fn tool_content_part_from_content_part(
    part: &ContentPart,
) -> Result<ToolContentPart, ModelMessageConversionError> {
    match part {
        ContentPart::ToolResult { .. } => Ok(ToolContentPart::ToolResult(
            tool_result_part_from_content_part(part)?,
        )),
        ContentPart::ToolApprovalResponse { .. } => Ok(ToolContentPart::ToolApprovalResponse(
            tool_approval_response_from_content_part(part)?,
        )),
        _ => Err(unsupported_part(
            "tool",
            part,
            "only tool-result and tool-approval-response parts are allowed",
        )),
    }
}

fn text_part_from_content_part(
    part: &ContentPart,
) -> Result<TextPart, ModelMessageConversionError> {
    match part {
        ContentPart::Text {
            text,
            provider_options,
            provider_metadata,
        } => {
            ensure_no_provider_metadata("text", provider_metadata)?;
            Ok(TextPart {
                part_type: text_part_type(),
                text: text.clone(),
                provider_options: provider_options.clone(),
            })
        }
        _ => Err(unsupported_part("unknown", part, "expected text part")),
    }
}

fn image_part_from_content_part(
    part: &ContentPart,
) -> Result<ImagePart, ModelMessageConversionError> {
    match part {
        ContentPart::Image {
            source,
            detail,
            provider_options,
            provider_metadata,
        } => {
            ensure_no_provider_metadata("image", provider_metadata)?;
            if detail.is_some() {
                return Err(ModelMessageConversionError::UnsupportedContentPart {
                    context: "user",
                    part_type: "image",
                    reason: "image detail is outside the AI SDK prompt content contract",
                });
            }

            Ok(ImagePart {
                part_type: image_part_type(),
                image: source.clone(),
                media_type: None,
                provider_options: provider_options.clone(),
            })
        }
        _ => Err(unsupported_part("unknown", part, "expected image part")),
    }
}

fn file_part_from_content_part(
    part: &ContentPart,
) -> Result<FilePart, ModelMessageConversionError> {
    match part {
        ContentPart::File {
            source,
            media_type,
            filename,
            provider_options,
            provider_metadata,
        } => {
            ensure_no_provider_metadata("file", provider_metadata)?;
            Ok(FilePart {
                part_type: file_part_type(),
                data: source.clone(),
                filename: filename.clone(),
                media_type: media_type.clone(),
                provider_options: provider_options.clone(),
            })
        }
        _ => Err(unsupported_part("unknown", part, "expected file part")),
    }
}

fn reasoning_part_from_content_part(
    part: &ContentPart,
) -> Result<ReasoningPart, ModelMessageConversionError> {
    match part {
        ContentPart::Reasoning {
            text,
            provider_options,
            provider_metadata,
        } => {
            ensure_no_provider_metadata("reasoning", provider_metadata)?;
            Ok(ReasoningPart {
                part_type: reasoning_part_type(),
                text: text.clone(),
                provider_options: provider_options.clone(),
            })
        }
        _ => Err(unsupported_part("unknown", part, "expected reasoning part")),
    }
}

fn custom_part_from_content_part(
    part: &ContentPart,
) -> Result<CustomPart, ModelMessageConversionError> {
    match part {
        ContentPart::Custom {
            kind,
            provider_options,
            provider_metadata,
        } => {
            ensure_no_provider_metadata("custom", provider_metadata)?;
            Ok(CustomPart {
                part_type: custom_part_type(),
                kind: kind.clone(),
                provider_options: provider_options.clone(),
            })
        }
        _ => Err(unsupported_part("unknown", part, "expected custom part")),
    }
}

fn reasoning_file_part_from_content_part(
    part: &ContentPart,
) -> Result<ReasoningFilePart, ModelMessageConversionError> {
    match part {
        ContentPart::ReasoningFile {
            source,
            media_type,
            provider_options,
            provider_metadata,
        } => {
            ensure_no_provider_metadata("reasoning-file", provider_metadata)?;
            Ok(ReasoningFilePart {
                part_type: reasoning_file_part_type(),
                data: source.clone(),
                media_type: media_type.clone(),
                provider_options: provider_options.clone(),
            })
        }
        _ => Err(unsupported_part(
            "unknown",
            part,
            "expected reasoning-file part",
        )),
    }
}

fn tool_call_part_from_content_part(
    part: &ContentPart,
) -> Result<ToolCallPart, ModelMessageConversionError> {
    match part {
        ContentPart::ToolCall {
            tool_call_id,
            tool_name,
            arguments,
            provider_executed,
            dynamic,
            invalid,
            error,
            title,
            provider_options,
            provider_metadata,
        } => {
            ensure_no_provider_metadata("tool-call", provider_metadata)?;
            ensure_none(
                "assistant",
                "tool-call",
                dynamic.as_ref(),
                "dynamic tool flags",
            )?;
            ensure_none(
                "assistant",
                "tool-call",
                invalid.as_ref(),
                "invalid tool flags",
            )?;
            ensure_none(
                "assistant",
                "tool-call",
                error.as_ref(),
                "invalid tool payloads",
            )?;
            ensure_none("assistant", "tool-call", title.as_ref(), "tool titles")?;

            Ok(ToolCallPart {
                part_type: tool_call_part_type(),
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                input: arguments.clone(),
                provider_options: provider_options.clone(),
                provider_executed: *provider_executed,
            })
        }
        _ => Err(unsupported_part("unknown", part, "expected tool-call part")),
    }
}

fn tool_result_part_from_content_part(
    part: &ContentPart,
) -> Result<ToolResultPart, ModelMessageConversionError> {
    match part {
        ContentPart::ToolResult {
            tool_call_id,
            tool_name,
            output,
            input,
            provider_executed,
            dynamic,
            preliminary,
            title,
            provider_options,
            provider_metadata,
        } => {
            ensure_no_provider_metadata("tool-result", provider_metadata)?;
            ensure_none(
                "assistant",
                "tool-result",
                input.as_ref(),
                "tool-result input",
            )?;
            ensure_none(
                "assistant",
                "tool-result",
                provider_executed.as_ref(),
                "provider-executed tool-result markers",
            )?;
            ensure_none(
                "assistant",
                "tool-result",
                dynamic.as_ref(),
                "dynamic tool flags",
            )?;
            ensure_none(
                "assistant",
                "tool-result",
                preliminary.as_ref(),
                "preliminary markers",
            )?;
            ensure_none("assistant", "tool-result", title.as_ref(), "tool titles")?;

            Ok(ToolResultPart {
                part_type: tool_result_part_type(),
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                output: output.clone(),
                provider_options: provider_options.clone(),
            })
        }
        _ => Err(unsupported_part(
            "unknown",
            part,
            "expected tool-result part",
        )),
    }
}

fn tool_approval_request_from_content_part(
    part: &ContentPart,
) -> Result<ToolApprovalRequest, ModelMessageConversionError> {
    match part {
        ContentPart::ToolApprovalRequest {
            approval_id,
            tool_call_id,
            provider_options,
            provider_metadata,
        } => {
            ensure_no_provider_metadata("tool-approval-request", provider_metadata)?;
            if !provider_options.is_empty() {
                return Err(ModelMessageConversionError::UnsupportedContentPart {
                    context: "assistant",
                    part_type: "tool-approval-request",
                    reason: "provider options are not part of the AI SDK tool-approval-request prompt contract",
                });
            }

            Ok(ToolApprovalRequest {
                part_type: tool_approval_request_part_type(),
                approval_id: approval_id.clone(),
                tool_call_id: tool_call_id.clone(),
            })
        }
        _ => Err(unsupported_part(
            "unknown",
            part,
            "expected tool-approval-request part",
        )),
    }
}

fn tool_approval_response_from_content_part(
    part: &ContentPart,
) -> Result<ToolApprovalResponse, ModelMessageConversionError> {
    match part {
        ContentPart::ToolApprovalResponse {
            approval_id,
            approved,
            reason,
            provider_executed,
            provider_options,
        } => {
            if !provider_options.is_empty() {
                return Err(ModelMessageConversionError::UnsupportedContentPart {
                    context: "tool",
                    part_type: "tool-approval-response",
                    reason: "provider options are not part of the AI SDK tool-approval-response prompt contract",
                });
            }

            Ok(ToolApprovalResponse {
                part_type: tool_approval_response_part_type(),
                approval_id: approval_id.clone(),
                approved: *approved,
                reason: reason.clone(),
                provider_executed: *provider_executed,
            })
        }
        _ => Err(unsupported_part(
            "unknown",
            part,
            "expected tool-approval-response part",
        )),
    }
}

fn ensure_no_provider_metadata(
    part_type: &'static str,
    provider_metadata: &Option<ProviderMetadataMap>,
) -> Result<(), ModelMessageConversionError> {
    if provider_metadata.is_some() {
        return Err(ModelMessageConversionError::UnsupportedContentPart {
            context: "prompt",
            part_type,
            reason: "provider metadata is response-side only",
        });
    }
    Ok(())
}

fn ensure_none<T>(
    context: &'static str,
    part_type: &'static str,
    value: Option<&T>,
    reason: &'static str,
) -> Result<(), ModelMessageConversionError> {
    if value.is_some() {
        return Err(ModelMessageConversionError::UnsupportedContentPart {
            context,
            part_type,
            reason,
        });
    }
    Ok(())
}

fn unsupported_part(
    context: &'static str,
    part: &ContentPart,
    reason: &'static str,
) -> ModelMessageConversionError {
    ModelMessageConversionError::UnsupportedContentPart {
        context,
        part_type: content_part_name(part),
        reason,
    }
}

fn content_part_name(part: &ContentPart) -> &'static str {
    match part {
        ContentPart::Text { .. } => "text",
        ContentPart::Image { .. } => "image",
        ContentPart::Audio { .. } => "audio",
        ContentPart::File { .. } => "file",
        ContentPart::ReasoningFile { .. } => "reasoning-file",
        ContentPart::Custom { .. } => "custom",
        ContentPart::Source { .. } => "source",
        ContentPart::ToolCall { .. } => "tool-call",
        ContentPart::ToolApprovalResponse { .. } => "tool-approval-response",
        ContentPart::ToolApprovalRequest { .. } => "tool-approval-request",
        ContentPart::ToolResult { .. } => "tool-result",
        ContentPart::Reasoning { .. } => "reasoning",
    }
}

fn message_role_name(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Developer => "developer",
        MessageRole::Tool => "tool",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_user_parts_roundtrip_through_model_message() {
        let message = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![
                ContentPart::text("hello"),
                ContentPart::file_provider_reference(
                    super::super::ProviderReference::single("openai", "file_123"),
                    "application/pdf",
                    Some("doc.pdf".to_string()),
                ),
            ]),
            provider_options: ProviderOptionsMap::default(),
            metadata: MessageMetadata::default(),
        };

        let model_message = ModelMessage::try_from(&message).expect("user model message");
        let roundtripped = ChatMessage::from(&model_message);

        assert_eq!(roundtripped.role, MessageRole::User);
        assert_eq!(roundtripped.content, message.content);
    }

    #[test]
    fn assistant_tool_call_with_runtime_only_extensions_is_rejected() {
        let message = ChatMessage {
            role: MessageRole::Assistant,
            content: MessageContent::MultiModal(vec![ContentPart::ToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "search".to_string(),
                arguments: serde_json::json!({ "q": "rust" }),
                provider_executed: None,
                dynamic: Some(true),
                invalid: None,
                error: None,
                title: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            provider_options: ProviderOptionsMap::default(),
            metadata: MessageMetadata::default(),
        };

        let err = ModelMessage::try_from(&message).expect_err("dynamic tool call must fail");
        assert!(matches!(
            err,
            ModelMessageConversionError::UnsupportedContentPart {
                context: "assistant",
                part_type: "tool-call",
                reason: "dynamic tool flags",
            }
        ));
    }

    #[test]
    fn developer_messages_are_not_part_of_model_message_contract() {
        let message = ChatMessage::developer("hidden").build();
        let err = ModelMessage::try_from(&message).expect_err("developer message must fail");

        assert!(matches!(
            err,
            ModelMessageConversionError::UnsupportedMessageRole { role } if role == "developer"
        ));
    }

    #[test]
    fn prompt_standardization_matches_ai_sdk_prompt_rules() {
        let prompt = Prompt::prompt_text("hello").with_system_text("rules");
        let standardized = prompt.standardize().expect("valid prompt");
        let chat_messages = standardized.to_chat_messages();

        assert_eq!(standardized.messages.len(), 1);
        assert_eq!(chat_messages.len(), 2);
        assert_eq!(chat_messages[0].role, MessageRole::System);
        assert_eq!(chat_messages[1].role, MessageRole::User);
        assert_eq!(chat_messages[1].content_text(), Some("hello"));
    }

    #[test]
    fn prompt_rejects_both_prompt_and_messages() {
        let prompt = Prompt {
            system: None,
            prompt: Some(PromptInput::Text("hello".to_string())),
            messages: Some(vec![ModelMessage::User(UserModelMessage::new(
                UserContent::Text("world".to_string()),
            ))]),
        };

        let err = prompt.standardize().expect_err("prompt must be invalid");
        assert_eq!(err, PromptValidationError::BothPromptAndMessages);
    }

    #[test]
    fn prompt_converts_to_chat_request() {
        let prompt = Prompt::messages(vec![
            ModelMessage::System(SystemModelMessage::new("rules")),
            ModelMessage::Assistant(AssistantModelMessage::new(AssistantContent::Text(
                "hello".to_string(),
            ))),
        ]);

        let request = ChatRequest::try_from(prompt).expect("chat request");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, MessageRole::System);
        assert_eq!(request.messages[1].role, MessageRole::Assistant);
    }

    #[test]
    fn data_content_helper_matches_ai_sdk_base64_projection() {
        let data = DataContent::binary([1_u8, 2, 3]);
        assert_eq!(convert_data_content_to_base64_string(&data), "AQID");
        assert_eq!(data.as_bytes().expect("bytes"), vec![1, 2, 3]);
    }

    #[test]
    fn text_part_rejects_wrong_discriminator_during_deserialization() {
        let err = serde_json::from_value::<TextPart>(serde_json::json!({
            "type": "image",
            "text": "hello"
        }))
        .expect_err("wrong part type must fail");

        assert!(
            err.to_string()
                .contains("expected prompt content part type `text`")
        );
    }

    #[test]
    fn system_model_message_rejects_wrong_role_during_deserialization() {
        let err = serde_json::from_value::<SystemModelMessage>(serde_json::json!({
            "role": "user",
            "content": "rules"
        }))
        .expect_err("wrong role must fail");

        assert!(
            err.to_string()
                .contains("expected model message role `system`")
        );
    }

    #[test]
    fn model_message_requires_explicit_role_when_deserializing() {
        let err = serde_json::from_value::<ModelMessage>(serde_json::json!({
            "content": "hello"
        }))
        .expect_err("missing role must fail");

        assert!(err.to_string().contains("data did not match any variant"));
    }

    #[test]
    fn tool_approval_response_builder_roundtrips_optional_fields() {
        let response = ToolApprovalResponse::new("approval_1", true)
            .with_reason("looks safe")
            .with_provider_executed(true);

        let value = serde_json::to_value(&response).expect("serialize tool approval response");
        assert_eq!(value["type"], serde_json::json!("tool-approval-response"));
        assert_eq!(value["approvalId"], serde_json::json!("approval_1"));
        assert_eq!(value["approved"], serde_json::json!(true));
        assert_eq!(value["reason"], serde_json::json!("looks safe"));
        assert_eq!(value["providerExecuted"], serde_json::json!(true));

        let roundtrip: ToolApprovalResponse =
            serde_json::from_value(value).expect("deserialize tool approval response");
        assert_eq!(roundtrip.approval_id, "approval_1");
        assert_eq!(roundtrip.reason.as_deref(), Some("looks safe"));
        assert_eq!(roundtrip.provider_executed, Some(true));
    }
}
