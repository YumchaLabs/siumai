//! AI SDK-aligned shared surface aliases and metadata helpers.
//!
//! These names intentionally mirror the shared `packages/ai/src/types/*` contract where
//! Siumai already has a stable equivalent or can expose a passive data structure honestly
//! without pretending the runtime wiring is more complete than it is today.

use super::chat::{ContentPart, SourcePart, UiMessage, UiMessagePart, UiMessageRole};
use super::{
    AssistantContent, AssistantContentPart, AssistantModelMessage, DataContent, EmbeddingUsage,
    FinishReason, FlexibleSchema, HttpRequestInfo, HttpResponseInfo, ModelMessage, PromptInput,
    ProviderMetadataMap, ProviderOptionsMap, ResponseMetadata, StandardizedPrompt, SystemPrompt,
    Tool, ToolChoice, ToolContentPart, ToolModelMessage, ToolResultOutput, Usage, Warning,
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

/// AI SDK provider `AISDKError` passive base error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AISDKError {
    /// Error name such as `AI_APICallError`.
    pub name: String,
    /// Human-readable error message.
    pub message: String,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
}

impl AISDKError {
    /// Create an AI SDK base error carrier.
    pub fn new(
        name: impl Into<String>,
        message: impl Into<String>,
        cause: Option<JSONValue>,
    ) -> Self {
        Self {
            name: name.into(),
            message: message.into(),
            cause,
        }
    }
}

/// AI SDK provider `APICallError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct APICallError {
    /// Human-readable error message.
    pub message: String,
    /// Request URL.
    pub url: String,
    /// Request body values when serializable.
    pub request_body_values: JSONValue,
    /// HTTP status code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_code: Option<u16>,
    /// HTTP response headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_headers: Option<HashMap<String, String>>,
    /// Raw response body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_body: Option<String>,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
    /// Whether the API call can be retried.
    pub is_retryable: bool,
    /// Parsed provider error payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<JSONValue>,
}

impl APICallError {
    /// Create an `APICallError` and apply AI SDK's default retryability rule.
    pub fn new(
        message: impl Into<String>,
        url: impl Into<String>,
        request_body_values: JSONValue,
        status_code: Option<u16>,
    ) -> Self {
        let is_retryable = matches!(status_code, Some(408 | 409 | 429))
            || status_code.is_some_and(|status_code| status_code >= 500);

        Self {
            message: message.into(),
            url: url.into(),
            request_body_values,
            status_code,
            response_headers: None,
            response_body: None,
            cause: None,
            is_retryable,
            data: None,
        }
    }
}

/// AI SDK provider `EmptyResponseBodyError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct EmptyResponseBodyError {
    /// Human-readable error message.
    pub message: String,
}

impl EmptyResponseBodyError {
    /// Create an `EmptyResponseBodyError` with the upstream default message.
    pub fn new() -> Self {
        Self {
            message: "Empty response body".to_string(),
        }
    }
}

impl Default for EmptyResponseBodyError {
    fn default() -> Self {
        Self::new()
    }
}

/// AI SDK provider `InvalidPromptError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InvalidPromptError {
    /// Human-readable error message.
    pub message: String,
    /// Invalid prompt payload.
    pub prompt: JSONValue,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
}

impl InvalidPromptError {
    /// Create an `InvalidPromptError` with the upstream default message prefix.
    pub fn new(prompt: JSONValue, message: impl Into<String>, cause: Option<JSONValue>) -> Self {
        Self {
            message: format!("Invalid prompt: {}", message.into()),
            prompt,
            cause,
        }
    }
}

/// AI SDK provider `InvalidResponseDataError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InvalidResponseDataError {
    /// Human-readable error message.
    pub message: String,
    /// Invalid response data payload.
    pub data: JSONValue,
}

impl InvalidResponseDataError {
    /// Create an `InvalidResponseDataError` with the upstream default message.
    pub fn new(data: JSONValue) -> Self {
        Self {
            message: format!("Invalid response data: {data}."),
            data,
        }
    }
}

/// AI SDK provider `JSONParseError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct JSONParseError {
    /// Human-readable error message.
    pub message: String,
    /// Original text that failed JSON parsing.
    pub text: String,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
}

impl JSONParseError {
    /// Create a `JSONParseError` with the upstream default message.
    pub fn new(text: impl Into<String>, cause: Option<JSONValue>) -> Self {
        let text = text.into();
        Self {
            message: format!(
                "JSON parsing failed: Text: {text}.\nError message: {}",
                ai_sdk_error_message(cause.as_ref())
            ),
            text,
            cause,
        }
    }
}

/// AI SDK provider `LoadAPIKeyError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct LoadAPIKeyError {
    /// Human-readable error message.
    pub message: String,
}

impl LoadAPIKeyError {
    /// Create a `LoadAPIKeyError`.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// AI SDK provider `LoadSettingError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct LoadSettingError {
    /// Human-readable error message.
    pub message: String,
}

impl LoadSettingError {
    /// Create a `LoadSettingError`.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// AI SDK provider `NoContentGeneratedError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NoContentGeneratedError {
    /// Human-readable error message.
    pub message: String,
}

impl NoContentGeneratedError {
    /// Create a `NoContentGeneratedError` with the upstream default message.
    pub fn new() -> Self {
        Self {
            message: "No content generated.".to_string(),
        }
    }
}

impl Default for NoContentGeneratedError {
    fn default() -> Self {
        Self::new()
    }
}

/// AI SDK provider model-family discriminator used by `NoSuchModelError`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NoSuchModelType {
    #[serde(rename = "languageModel")]
    LanguageModel,
    #[serde(rename = "embeddingModel")]
    EmbeddingModel,
    #[serde(rename = "imageModel")]
    ImageModel,
    #[serde(rename = "transcriptionModel")]
    TranscriptionModel,
    #[serde(rename = "speechModel")]
    SpeechModel,
    #[serde(rename = "rerankingModel")]
    RerankingModel,
    #[serde(rename = "videoModel")]
    VideoModel,
}

impl NoSuchModelType {
    /// Return the AI SDK string value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LanguageModel => "languageModel",
            Self::EmbeddingModel => "embeddingModel",
            Self::ImageModel => "imageModel",
            Self::TranscriptionModel => "transcriptionModel",
            Self::SpeechModel => "speechModel",
            Self::RerankingModel => "rerankingModel",
            Self::VideoModel => "videoModel",
        }
    }
}

/// AI SDK provider `NoSuchModelError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NoSuchModelError {
    /// Human-readable error message.
    pub message: String,
    /// Missing model id.
    pub model_id: String,
    /// Model family.
    pub model_type: NoSuchModelType,
}

impl NoSuchModelError {
    /// Create a `NoSuchModelError` with the upstream default message.
    pub fn new(model_id: impl Into<String>, model_type: NoSuchModelType) -> Self {
        let model_id = model_id.into();
        Self {
            message: format!("No such {}: {model_id}", model_type.as_str()),
            model_id,
            model_type,
        }
    }
}

/// AI SDK provider `NoSuchProviderReferenceError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NoSuchProviderReferenceError {
    /// Human-readable error message.
    pub message: String,
    /// Missing provider id.
    pub provider: String,
    /// Available provider references.
    pub reference: HashMap<String, String>,
}

impl NoSuchProviderReferenceError {
    /// Create a `NoSuchProviderReferenceError` with the upstream default message.
    pub fn new(provider: impl Into<String>, reference: HashMap<String, String>) -> Self {
        let provider = provider.into();
        let available = reference.keys().cloned().collect::<Vec<_>>().join(", ");
        Self {
            message: format!(
                "No provider reference found for provider '{provider}'. Available providers: {available}"
            ),
            provider,
            reference,
        }
    }
}

/// AI SDK provider `TooManyEmbeddingValuesForCallError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TooManyEmbeddingValuesForCallError {
    /// Human-readable error message.
    pub message: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Maximum values accepted by one provider call.
    pub max_embeddings_per_call: usize,
    /// Provided embedding input values.
    pub values: Vec<JSONValue>,
}

impl TooManyEmbeddingValuesForCallError {
    /// Create a `TooManyEmbeddingValuesForCallError` with the upstream default message.
    pub fn new(
        provider: impl Into<String>,
        model_id: impl Into<String>,
        max_embeddings_per_call: usize,
        values: Vec<JSONValue>,
    ) -> Self {
        let provider = provider.into();
        let model_id = model_id.into();
        Self {
            message: format!(
                "Too many values for a single embedding call. The {provider} model \
                 \"{model_id}\" can only embed up to {max_embeddings_per_call} values per call, \
                 but {} values were provided.",
                values.len()
            ),
            provider,
            model_id,
            max_embeddings_per_call,
            values,
        }
    }
}

/// AI SDK provider `TypeValidationContext` passive data.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TypeValidationContext {
    /// Field path in dot notation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// Entity name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_name: Option<String>,
    /// Entity identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
}

/// AI SDK provider `TypeValidationError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TypeValidationError {
    /// Human-readable error message.
    pub message: String,
    /// Value that failed validation.
    pub value: JSONValue,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
    /// Validation context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<TypeValidationContext>,
}

impl TypeValidationError {
    /// Create a `TypeValidationError` with the upstream default message.
    pub fn new(
        value: JSONValue,
        cause: Option<JSONValue>,
        context: Option<TypeValidationContext>,
    ) -> Self {
        let mut context_prefix = "Type validation failed".to_string();

        if let Some(field) = context.as_ref().and_then(|context| context.field.as_ref()) {
            context_prefix.push_str(" for ");
            context_prefix.push_str(field);
        }

        if context
            .as_ref()
            .is_some_and(|context| context.entity_name.is_some() || context.entity_id.is_some())
        {
            let mut parts = Vec::new();
            if let Some(entity_name) = context
                .as_ref()
                .and_then(|context| context.entity_name.as_ref())
            {
                parts.push(entity_name.clone());
            }
            if let Some(entity_id) = context
                .as_ref()
                .and_then(|context| context.entity_id.as_ref())
            {
                parts.push(format!("id: \"{entity_id}\""));
            }
            context_prefix.push_str(" (");
            context_prefix.push_str(&parts.join(", "));
            context_prefix.push(')');
        }

        Self {
            message: format!(
                "{context_prefix}: Value: {value}.\nError message: {}",
                ai_sdk_error_message(cause.as_ref())
            ),
            value,
            cause,
            context,
        }
    }
}

/// AI SDK provider `UnsupportedFunctionalityError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UnsupportedFunctionalityError {
    /// Human-readable error message.
    pub message: String,
    /// Unsupported functionality name.
    pub functionality: String,
}

impl UnsupportedFunctionalityError {
    /// Create an `UnsupportedFunctionalityError` with the upstream default message.
    pub fn new(functionality: impl Into<String>) -> Self {
        let functionality = functionality.into();
        Self {
            message: format!("'{functionality}' functionality not supported."),
            functionality,
        }
    }
}

/// AI SDK `InvalidArgumentError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InvalidArgumentError {
    /// Human-readable error message.
    pub message: String,
    /// Invalid parameter name.
    pub parameter: String,
    /// Invalid parameter value when serializable.
    pub value: JSONValue,
}

impl InvalidArgumentError {
    /// Create an `InvalidArgumentError` with the upstream default message prefix.
    pub fn new(parameter: impl Into<String>, value: JSONValue, message: impl Into<String>) -> Self {
        let parameter = parameter.into();
        let message = format!(
            "Invalid argument for parameter {parameter}: {}",
            message.into()
        );

        Self {
            message,
            parameter,
            value,
        }
    }
}

/// AI SDK `InvalidStreamPartError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InvalidStreamPartError<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// Human-readable error message.
    pub message: String,
    /// Invalid language-model stream chunk.
    pub chunk: LanguageModelStreamPart<NAME, INPUT, OUTPUT>,
}

impl<NAME, INPUT, OUTPUT> InvalidStreamPartError<NAME, INPUT, OUTPUT> {
    /// Create an `InvalidStreamPartError`.
    pub fn new(
        chunk: LanguageModelStreamPart<NAME, INPUT, OUTPUT>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            message: message.into(),
            chunk,
        }
    }
}

/// AI SDK `InvalidToolApprovalError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct InvalidToolApprovalError {
    /// Human-readable error message.
    pub message: String,
    /// Unknown approval id.
    pub approval_id: String,
}

impl InvalidToolApprovalError {
    /// Create an `InvalidToolApprovalError` with the upstream default message.
    pub fn new(approval_id: impl Into<String>) -> Self {
        let approval_id = approval_id.into();
        Self {
            message: format!(
                "Tool approval response references unknown approvalId: \"{approval_id}\". \
                 No matching tool-approval-request found in message history."
            ),
            approval_id,
        }
    }
}

/// AI SDK `ToolCallNotFoundForApprovalError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallNotFoundForApprovalError {
    /// Human-readable error message.
    pub message: String,
    /// Tool call id that was referenced by the approval request.
    pub tool_call_id: String,
    /// Approval id that could not be matched.
    pub approval_id: String,
}

impl ToolCallNotFoundForApprovalError {
    /// Create a `ToolCallNotFoundForApprovalError` with the upstream default message.
    pub fn new(tool_call_id: impl Into<String>, approval_id: impl Into<String>) -> Self {
        let tool_call_id = tool_call_id.into();
        let approval_id = approval_id.into();
        Self {
            message: format!(
                "Tool call \"{tool_call_id}\" not found for approval request \"{approval_id}\"."
            ),
            tool_call_id,
            approval_id,
        }
    }
}

/// AI SDK `NoImageGeneratedError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NoImageGeneratedError {
    /// Human-readable error message.
    pub message: String,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
    /// Response metadata for each call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responses: Option<Vec<ImageModelResponseMetadata>>,
}

impl NoImageGeneratedError {
    /// Create a `NoImageGeneratedError` with the upstream default message.
    pub fn new(
        responses: Option<Vec<ImageModelResponseMetadata>>,
        cause: Option<JSONValue>,
    ) -> Self {
        Self {
            message: "No image generated.".to_string(),
            cause,
            responses,
        }
    }
}

/// AI SDK `NoObjectGeneratedError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NoObjectGeneratedError {
    /// Human-readable error message.
    pub message: String,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
    /// Generated text that failed parsing or validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Response metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<LanguageModelResponseMetadata>,
    /// Language model usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<LanguageModelUsage>,
    /// Finish reason when known.
    #[serde(rename = "finishReason", skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

impl NoObjectGeneratedError {
    /// Create a `NoObjectGeneratedError` with the upstream default message.
    pub fn new(
        text: Option<String>,
        response: Option<LanguageModelResponseMetadata>,
        usage: Option<LanguageModelUsage>,
        finish_reason: Option<FinishReason>,
        cause: Option<JSONValue>,
    ) -> Self {
        Self {
            message: "No object generated.".to_string(),
            cause,
            text,
            response,
            usage,
            finish_reason,
        }
    }
}

/// AI SDK `NoOutputGeneratedError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NoOutputGeneratedError {
    /// Human-readable error message.
    pub message: String,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
}

impl NoOutputGeneratedError {
    /// Create a `NoOutputGeneratedError` with the upstream default message.
    pub fn new(cause: Option<JSONValue>) -> Self {
        Self {
            message: "No output generated.".to_string(),
            cause,
        }
    }
}

/// AI SDK `NoSpeechGeneratedError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NoSpeechGeneratedError {
    /// Human-readable error message.
    pub message: String,
    /// Response metadata for each call.
    pub responses: Vec<SpeechModelResponseMetadata>,
}

impl NoSpeechGeneratedError {
    /// Create a `NoSpeechGeneratedError` with the upstream default message.
    pub fn new(responses: Vec<SpeechModelResponseMetadata>) -> Self {
        Self {
            message: "No speech audio generated.".to_string(),
            responses,
        }
    }
}

/// AI SDK `NoTranscriptGeneratedError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NoTranscriptGeneratedError {
    /// Human-readable error message.
    pub message: String,
    /// Response metadata for each call.
    pub responses: Vec<TranscriptionModelResponseMetadata>,
}

impl NoTranscriptGeneratedError {
    /// Create a `NoTranscriptGeneratedError` with the upstream default message.
    pub fn new(responses: Vec<TranscriptionModelResponseMetadata>) -> Self {
        Self {
            message: "No transcript generated.".to_string(),
            responses,
        }
    }
}

/// AI SDK `NoVideoGeneratedError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NoVideoGeneratedError {
    /// Human-readable error message.
    pub message: String,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
    /// Response metadata for each call.
    pub responses: Vec<VideoModelResponseMetadata>,
}

impl NoVideoGeneratedError {
    /// Create a `NoVideoGeneratedError` with the upstream default message.
    pub fn new(responses: Vec<VideoModelResponseMetadata>, cause: Option<JSONValue>) -> Self {
        Self {
            message: "No video generated.".to_string(),
            cause,
            responses,
        }
    }
}

/// AI SDK `UnsupportedModelVersionError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UnsupportedModelVersionError {
    /// Human-readable error message.
    pub message: String,
    /// Unsupported model specification version.
    pub version: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
}

impl UnsupportedModelVersionError {
    /// Create an `UnsupportedModelVersionError` with the upstream default message.
    pub fn new(
        version: impl Into<String>,
        provider: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Self {
        let version = version.into();
        let provider = provider.into();
        let model_id = model_id.into();
        Self {
            message: format!(
                "Unsupported model version {version} for provider \"{provider}\" and model \
                 \"{model_id}\". AI SDK 5 only supports models that implement specification \
                 version \"v2\"."
            ),
            version,
            provider,
            model_id,
        }
    }
}

/// AI SDK `UIMessageStreamError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UIMessageStreamError {
    /// Human-readable error message.
    pub message: String,
    /// Chunk type that caused the error.
    pub chunk_type: String,
    /// Part id or tool-call id associated with the failing chunk.
    pub chunk_id: String,
}

impl UIMessageStreamError {
    /// Create a `UIMessageStreamError`.
    pub fn new(
        chunk_type: impl Into<String>,
        chunk_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            message: message.into(),
            chunk_type: chunk_type.into(),
            chunk_id: chunk_id.into(),
        }
    }
}

/// AI SDK `InvalidMessageRoleError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct InvalidMessageRoleError {
    /// Human-readable error message.
    pub message: String,
    /// Invalid message role.
    pub role: String,
}

impl InvalidMessageRoleError {
    /// Create an `InvalidMessageRoleError` with the upstream default message.
    pub fn new(role: impl Into<String>) -> Self {
        let role = role.into();
        Self {
            message: format!(
                "Invalid message role: '{role}'. Must be one of: \"system\", \"user\", \
                 \"assistant\", \"tool\"."
            ),
            role,
        }
    }
}

/// AI SDK `Omit<UIMessage, 'id'>` payload used by `MessageConversionError`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiMessageWithoutId {
    /// Message role.
    pub role: UiMessageRole,
    /// Optional UI-only metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JSONValue>,
    /// Renderable UI parts.
    #[serde(default)]
    pub parts: Vec<UiMessagePart>,
}

impl UiMessageWithoutId {
    /// Create a UI message payload without an id.
    pub fn new(role: UiMessageRole, parts: Vec<UiMessagePart>) -> Self {
        Self {
            role,
            metadata: None,
            parts,
        }
    }

    /// Attach UI-only metadata.
    pub fn with_metadata(mut self, metadata: JSONValue) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// AI SDK `MessageConversionError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MessageConversionError {
    /// Human-readable error message.
    pub message: String,
    /// Original UI message without the `id` field.
    pub original_message: UiMessageWithoutId,
}

impl MessageConversionError {
    /// Create a `MessageConversionError`.
    pub fn new(original_message: UiMessageWithoutId, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            original_message,
        }
    }
}

/// AI SDK `RetryErrorReason` union.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetryErrorReason {
    #[serde(rename = "maxRetriesExceeded")]
    MaxRetriesExceeded,
    #[serde(rename = "errorNotRetryable")]
    ErrorNotRetryable,
    #[serde(rename = "abort")]
    Abort,
}

/// AI SDK `RetryError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RetryError {
    /// Human-readable error message.
    pub message: String,
    /// Retry termination reason.
    pub reason: RetryErrorReason,
    /// Last error payload when at least one error exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<JSONValue>,
    /// Error payloads collected across retry attempts.
    pub errors: Vec<JSONValue>,
}

impl RetryError {
    /// Create a `RetryError` and derive `lastError` from the final entry.
    pub fn new(
        message: impl Into<String>,
        reason: RetryErrorReason,
        errors: Vec<JSONValue>,
    ) -> Self {
        Self {
            message: message.into(),
            reason,
            last_error: errors.last().cloned(),
            errors,
        }
    }
}

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

/// AI SDK `DefaultGeneratedFile` export. Rust keeps the same value carrier as `GeneratedFile`.
pub type DefaultGeneratedFile = GeneratedFile;

/// Backwards-compatible AI SDK `Experimental_GeneratedImage` export.
#[allow(non_camel_case_types)]
pub type Experimental_GeneratedImage = GeneratedFile;

fn audio_format_from_media_type(media_type: &str) -> String {
    let normalized = media_type.trim().to_ascii_lowercase();
    if normalized == "audio/mpeg" {
        return "mp3".to_string();
    }

    normalized
        .split_once('/')
        .map(|(_, subtype)| subtype.to_string())
        .filter(|subtype| !subtype.is_empty())
        .unwrap_or_else(|| "mp3".to_string())
}

/// AI SDK-style generated audio file returned by `generateSpeech`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GeneratedAudioFile {
    /// File content and media type.
    #[serde(flatten)]
    pub file: GeneratedFile,
    /// Audio format such as `mp3` or `wav`.
    pub format: String,
}

impl GeneratedAudioFile {
    /// Create generated audio from a generated file and explicit format.
    pub fn new(file: GeneratedFile, format: impl Into<String>) -> Self {
        Self {
            file,
            format: format.into(),
        }
    }

    /// Create generated audio from base64 content, deriving the format from the media type.
    pub fn from_base64(base64: impl Into<String>, media_type: impl Into<String>) -> Self {
        let media_type = media_type.into();
        Self::new(
            GeneratedFile::from_base64(base64, media_type.as_str()),
            audio_format_from_media_type(&media_type),
        )
    }

    /// Create generated audio from bytes, deriving the format from the media type.
    pub fn from_bytes(data: impl AsRef<[u8]>, media_type: impl Into<String>) -> Self {
        let media_type = media_type.into();
        Self::new(
            GeneratedFile::from_bytes(data, media_type.as_str()),
            audio_format_from_media_type(&media_type),
        )
    }

    /// Return base64 content.
    pub fn base64(&self) -> &str {
        self.file.base64()
    }

    /// Return the generated audio media type.
    pub fn media_type(&self) -> &str {
        self.file.media_type.as_str()
    }

    /// Decode the generated audio into bytes.
    pub fn uint8_array(&self) -> Result<Vec<u8>, base64::DecodeError> {
        self.file.uint8_array()
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

/// Static AI SDK tool-call view. Rust keeps the same carrier and uses `dynamic` as data.
pub type StaticToolCall<NAME = String, INPUT = JSONValue> = ToolCall<NAME, INPUT>;

/// Dynamic AI SDK tool-call view. Rust keeps the same carrier and uses `dynamic` as data.
pub type DynamicToolCall<INPUT = JSONValue> = ToolCall<String, INPUT>;

/// Typed AI SDK tool-call view (`StaticToolCall | DynamicToolCall`).
pub type TypedToolCall<NAME = String, INPUT = JSONValue> = ToolCall<NAME, INPUT>;

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

/// Static AI SDK tool-result view. Rust keeps the same carrier and uses `dynamic` as data.
pub type StaticToolResult<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    ToolResult<NAME, INPUT, OUTPUT>;

/// Dynamic AI SDK tool-result view. Rust keeps the same carrier and uses `dynamic` as data.
pub type DynamicToolResult<INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    ToolResult<String, INPUT, OUTPUT>;

/// Typed AI SDK tool-result view (`StaticToolResult | DynamicToolResult`).
pub type TypedToolResult<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    ToolResult<NAME, INPUT, OUTPUT>;

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

/// Static AI SDK tool-error view. Rust keeps the same carrier and uses `dynamic` as data.
pub type StaticToolError<NAME = String, INPUT = JSONValue> = ToolError<NAME, INPUT>;

/// Dynamic AI SDK tool-error view. Rust keeps the same carrier and uses `dynamic` as data.
pub type DynamicToolError<INPUT = JSONValue> = ToolError<String, INPUT>;

/// Typed AI SDK tool-error view (`StaticToolError | DynamicToolError`).
pub type TypedToolError<NAME = String, INPUT = JSONValue> = ToolError<NAME, INPUT>;

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

/// AI SDK `StepResult` export. Rust keeps the same passive carrier as `GenerateTextStepResult`.
pub type StepResult<NAME = String, INPUT = JSONValue, ToolOutput = ToolResultOutput> =
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

/// Output strategy marker used by AI SDK `generateObject` and `streamObject` callbacks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum GenerateObjectOutputStrategy {
    /// Object output strategy.
    Object,
    /// Array output strategy.
    Array,
    /// Enum output strategy.
    Enum,
    /// Schema-less JSON output strategy.
    NoSchema,
}

/// AI SDK-style response metadata envelope for structured object generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectResponseMetadata {
    /// Shared language-model response metadata.
    #[serde(flatten)]
    pub metadata: LanguageModelResponseMetadata,
    /// Raw response body when the provider exposes it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl GenerateObjectResponseMetadata {
    /// Create a structured-output response metadata envelope.
    pub fn new(metadata: LanguageModelResponseMetadata) -> Self {
        Self {
            metadata,
            body: None,
        }
    }

    /// Attach a raw response body.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }
}

/// Event payload for AI SDK `generateObject` / `streamObject` start callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectStartEvent {
    /// Unique generation call id.
    pub call_id: String,
    /// Operation id, normally `ai.generateObject` or `ai.streamObject`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// System prompt input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    /// Prompt input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptInput>,
    /// Message input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<ModelMessage>>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<f64>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Deterministic seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Additional HTTP headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Provider-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
    /// Output strategy.
    pub output: GenerateObjectOutputStrategy,
    /// JSON Schema used for object generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<JSONSchema7>,
    /// Optional schema name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_name: Option<String>,
    /// Optional schema description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_description: Option<String>,
}

/// Event payload for AI SDK structured-output step start callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectStepStartEvent {
    /// Unique generation call id.
    pub call_id: String,
    /// Zero-based step index. Upstream object generation currently uses step `0`.
    pub step_number: u32,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Provider-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
    /// Additional HTTP headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Prompt messages in provider format, approximated by Siumai's shared model-message carrier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_messages: Option<Vec<ModelMessage>>,
}

/// Event payload for AI SDK structured-output step finish callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectStepEndEvent {
    /// Unique generation call id.
    pub call_id: String,
    /// Zero-based step index. Upstream object generation currently uses step `0`.
    pub step_number: u32,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Unified finish reason.
    pub finish_reason: FinishReason,
    /// Token usage.
    pub usage: LanguageModelUsage,
    /// Raw object text before parsing/validation.
    pub object_text: String,
    /// Reasoning text when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Provider warnings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<CallWarning>>,
    /// Request metadata.
    pub request: LanguageModelRequestMetadata,
    /// Response metadata.
    pub response: GenerateObjectResponseMetadata,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Milliseconds from stream start to first chunk on streaming paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms_to_first_chunk: Option<u64>,
}

/// Event payload for AI SDK structured-output finish callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectEndEvent<RESULT = JSONValue> {
    /// Unique generation call id.
    pub call_id: String,
    /// Generated object when parsing and validation succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<RESULT>,
    /// Parse or validation error when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JSONValue>,
    /// Reasoning text when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Unified finish reason.
    pub finish_reason: FinishReason,
    /// Token usage.
    pub usage: LanguageModelUsage,
    /// Provider warnings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<CallWarning>>,
    /// Request metadata.
    pub request: LanguageModelRequestMetadata,
    /// Response metadata.
    pub response: GenerateObjectResponseMetadata,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
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
fixed_text_stream_type_marker!(ObjectStreamObjectPartMarker, "object");
fixed_text_stream_type_marker!(ObjectStreamTextDeltaPartMarker, "text-delta");
fixed_text_stream_type_marker!(ObjectStreamErrorPartMarker, "error");
fixed_text_stream_type_marker!(ObjectStreamFinishPartMarker, "finish");
fixed_text_stream_type_marker!(UiMessageTextStartChunkMarker, "text-start");
fixed_text_stream_type_marker!(UiMessageTextDeltaChunkMarker, "text-delta");
fixed_text_stream_type_marker!(UiMessageTextEndChunkMarker, "text-end");
fixed_text_stream_type_marker!(UiMessageReasoningStartChunkMarker, "reasoning-start");
fixed_text_stream_type_marker!(UiMessageReasoningDeltaChunkMarker, "reasoning-delta");
fixed_text_stream_type_marker!(UiMessageReasoningEndChunkMarker, "reasoning-end");
fixed_text_stream_type_marker!(UiMessageCustomChunkMarker, "custom");
fixed_text_stream_type_marker!(UiMessageErrorChunkMarker, "error");
fixed_text_stream_type_marker!(UiMessageToolInputStartChunkMarker, "tool-input-start");
fixed_text_stream_type_marker!(UiMessageToolInputDeltaChunkMarker, "tool-input-delta");
fixed_text_stream_type_marker!(
    UiMessageToolInputAvailableChunkMarker,
    "tool-input-available"
);
fixed_text_stream_type_marker!(UiMessageToolInputErrorChunkMarker, "tool-input-error");
fixed_text_stream_type_marker!(
    UiMessageToolApprovalRequestChunkMarker,
    "tool-approval-request"
);
fixed_text_stream_type_marker!(
    UiMessageToolApprovalResponseChunkMarker,
    "tool-approval-response"
);
fixed_text_stream_type_marker!(
    UiMessageToolOutputAvailableChunkMarker,
    "tool-output-available"
);
fixed_text_stream_type_marker!(UiMessageToolOutputErrorChunkMarker, "tool-output-error");
fixed_text_stream_type_marker!(UiMessageToolOutputDeniedChunkMarker, "tool-output-denied");
fixed_text_stream_type_marker!(UiMessageSourceUrlChunkMarker, "source-url");
fixed_text_stream_type_marker!(UiMessageSourceDocumentChunkMarker, "source-document");
fixed_text_stream_type_marker!(UiMessageFileChunkMarker, "file");
fixed_text_stream_type_marker!(UiMessageReasoningFileChunkMarker, "reasoning-file");
fixed_text_stream_type_marker!(UiMessageStartStepChunkMarker, "start-step");
fixed_text_stream_type_marker!(UiMessageFinishStepChunkMarker, "finish-step");
fixed_text_stream_type_marker!(UiMessageStartChunkMarker, "start");
fixed_text_stream_type_marker!(UiMessageFinishChunkMarker, "finish");
fixed_text_stream_type_marker!(UiMessageAbortChunkMarker, "abort");
fixed_text_stream_type_marker!(UiMessageMetadataChunkMarker, "message-metadata");

/// Headers used by AI SDK UI message streams.
pub const UI_MESSAGE_STREAM_HEADERS: &[(&str, &str)] = &[
    ("content-type", "text/event-stream"),
    ("cache-control", "no-cache"),
    ("connection", "keep-alive"),
    ("x-vercel-ai-ui-message-stream", "v1"),
    ("x-accel-buffering", "no"),
];

/// AI SDK UI data-part schema map.
pub type UIDataPartSchemas = HashMap<String, FlexibleSchema<JSONValue>>;

/// AI SDK UI data type to schema map. Rust keeps this as the same runtime map.
pub type UIDataTypesToSchemas = UIDataPartSchemas;

/// AI SDK inferred UI data parts. Rust exposes the resolved JSON-value map directly.
pub type InferUIDataParts = HashMap<String, JSONValue>;

/// Passive message input accepted by AI SDK `CreateUIMessage`.
///
/// Upstream models this as `Omit<UIMessage, "id" | "role"> & { id?: ...; role?: ... }`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CreateUIMessage<MessagePart = super::chat::UiMessagePart> {
    /// Optional UI message id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Optional role, defaulted by the caller/runtime when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<super::chat::UiMessageRole>,
    /// Optional UI-only metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JSONValue>,
    /// Renderable UI parts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<MessagePart>,
}

impl<MessagePart> Default for CreateUIMessage<MessagePart> {
    fn default() -> Self {
        Self {
            id: None,
            role: None,
            metadata: None,
            parts: Vec::new(),
        }
    }
}

impl<MessagePart> CreateUIMessage<MessagePart> {
    /// Create an empty create-message payload.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the optional message id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the optional message role.
    pub fn with_role(mut self, role: super::chat::UiMessageRole) -> Self {
        self.role = Some(role);
        self
    }

    /// Set UI-only metadata.
    pub fn with_metadata(mut self, metadata: impl Into<JSONValue>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }

    /// Set renderable UI parts.
    pub fn with_parts(mut self, parts: impl IntoIterator<Item = MessagePart>) -> Self {
        self.parts = parts.into_iter().collect();
        self
    }
}

/// Serializable subset of AI SDK `ChatRequestOptions`.
///
/// Browser `Headers` objects are represented as a plain string map.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatRequestOptions {
    /// Additional headers passed to the API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Additional JSON body properties sent to the API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
    /// Request metadata passed through the UI transport.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JSONValue>,
}

impl ChatRequestOptions {
    /// Create empty chat request options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set request headers.
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers = Some(headers.into_iter().collect());
        self
    }

    /// Set additional request body properties.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Set request metadata.
    pub fn with_metadata(mut self, metadata: impl Into<JSONValue>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }
}

/// AI SDK `ChatStatus` values.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum ChatStatus {
    Submitted,
    Streaming,
    Ready,
    Error,
}

/// Passive snapshot of AI SDK `ChatState`.
///
/// Upstream also includes mutation/snapshot functions. Rust keeps only the serializable state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatState<Message = UiMessage> {
    /// Current chat status.
    pub status: ChatStatus,
    /// Error payload when the chat is in an error state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JSONValue>,
    /// Current UI messages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
}

impl<Message> ChatState<Message> {
    /// Create a ready chat-state snapshot.
    pub fn ready(messages: Vec<Message>) -> Self {
        Self {
            status: ChatStatus::Ready,
            error: None,
            messages,
        }
    }

    /// Attach an error payload and mark the chat as errored.
    pub fn with_error(mut self, error: impl Into<JSONValue>) -> Self {
        self.status = ChatStatus::Error;
        self.error = Some(error.into());
        self
    }
}

/// Serializable subset of AI SDK `ChatInit`.
///
/// Function-valued callbacks, `generateId`, schema validators, and transport objects are
/// intentionally deferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatInit<Message = UiMessage> {
    /// Optional chat id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Initial messages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
}

impl<Message> ChatInit<Message> {
    /// Create empty chat initialization options.
    pub fn new() -> Self {
        Self {
            id: None,
            messages: Vec::new(),
        }
    }

    /// Set the chat id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set initial messages.
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }
}

/// AI SDK chat transport send trigger.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum ChatTransportTrigger {
    SubmitMessage,
    RegenerateMessage,
}

/// Passive options passed to AI SDK `ChatTransport.sendMessages`.
///
/// `AbortSignal` is runtime-only and intentionally omitted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatTransportSendMessagesOptions<Message = UiMessage> {
    /// New submission or regeneration.
    pub trigger: ChatTransportTrigger,
    /// Chat session id.
    pub chat_id: String,
    /// Message id for regeneration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Conversation history.
    pub messages: Vec<Message>,
    /// Additional request options.
    #[serde(flatten)]
    pub request_options: ChatRequestOptions,
}

/// Passive options passed to AI SDK `ChatTransport.reconnectToStream`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatTransportReconnectToStreamOptions {
    /// Chat session id.
    pub chat_id: String,
    /// Additional request options.
    #[serde(flatten)]
    pub request_options: ChatRequestOptions,
}

/// Serializable subset of AI SDK `HttpChatTransportInitOptions`.
///
/// Custom `fetch` and request-preparation callbacks are intentionally deferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HttpChatTransportInitOptions {
    /// Chat API URL. Upstream defaults to `/api/chat` when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// Browser credentials mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// HTTP headers sent with requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Extra body object sent with requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl HttpChatTransportInitOptions {
    /// Create empty HTTP chat transport options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chat API URL.
    pub fn with_api(mut self, api: impl Into<String>) -> Self {
        self.api = Some(api.into());
        self
    }

    /// Set the credentials mode.
    pub fn with_credentials(mut self, credentials: RequestCredentials) -> Self {
        self.credentials = Some(credentials);
        self
    }
}

/// Passive input payload supplied to AI SDK `PrepareSendMessagesRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PrepareSendMessagesRequestOptions<Message = UiMessage> {
    /// Chat id.
    pub id: String,
    /// Conversation history.
    pub messages: Vec<Message>,
    /// Request metadata from `ChatRequestOptions`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_metadata: Option<JSONValue>,
    /// Merged body before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
    /// Credentials mode before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// Headers before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// API URL before final preparation.
    pub api: String,
    /// New submission or regeneration.
    pub trigger: ChatTransportTrigger,
    /// Message id for regeneration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
}

/// Passive return payload from AI SDK `PrepareSendMessagesRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PreparedSendMessagesRequest {
    /// Final request body.
    pub body: JSONValue,
    /// Final request headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Final request credentials.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// Final API URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
}

/// Passive input payload supplied to AI SDK `PrepareReconnectToStreamRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PrepareReconnectToStreamRequestOptions {
    /// Chat id.
    pub id: String,
    /// Request metadata from `ChatRequestOptions`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_metadata: Option<JSONValue>,
    /// Merged body before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
    /// Credentials mode before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// Headers before final preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// API URL before final preparation.
    pub api: String,
}

/// Passive return payload from AI SDK `PrepareReconnectToStreamRequest`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PreparedReconnectToStreamRequest {
    /// Final request headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Final request credentials.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// Final API URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
}

/// Serializable subset of AI SDK `CompletionRequestOptions`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CompletionRequestOptions {
    /// Additional headers passed to the API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Additional JSON body properties sent to the API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl CompletionRequestOptions {
    /// Create empty completion request options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set request headers.
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers = Some(headers.into_iter().collect());
        self
    }

    /// Set additional request body properties.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }
}

/// Browser request credentials mode used by AI SDK UI helpers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum RequestCredentials {
    Omit,
    SameOrigin,
    Include,
}

/// AI SDK `useCompletion` stream protocol.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum CompletionStreamProtocol {
    Data,
    Text,
}

/// Serializable subset of AI SDK `UseCompletionOptions`.
///
/// Function-valued options (`onFinish`, `onError`) and custom `fetch` are intentionally deferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UseCompletionOptions {
    /// Completion API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// Shared completion id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Initial prompt input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_input: Option<String>,
    /// Initial completion text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_completion: Option<String>,
    /// Browser credentials mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<RequestCredentials>,
    /// HTTP headers sent with the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Extra JSON body object sent with the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
    /// Streaming protocol, defaulted by the UI runtime when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_protocol: Option<CompletionStreamProtocol>,
}

impl UseCompletionOptions {
    /// Create empty completion hook options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the completion API endpoint.
    pub fn with_api(mut self, api: impl Into<String>) -> Self {
        self.api = Some(api.into());
        self
    }

    /// Set the shared completion id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the initial prompt input.
    pub fn with_initial_input(mut self, initial_input: impl Into<String>) -> Self {
        self.initial_input = Some(initial_input.into());
        self
    }

    /// Set the initial completion text.
    pub fn with_initial_completion(mut self, initial_completion: impl Into<String>) -> Self {
        self.initial_completion = Some(initial_completion.into());
        self
    }

    /// Set browser credentials mode.
    pub fn with_credentials(mut self, credentials: RequestCredentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Set request headers.
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers = Some(headers.into_iter().collect());
        self
    }

    /// Set extra request body properties.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Set stream protocol.
    pub fn with_stream_protocol(mut self, stream_protocol: CompletionStreamProtocol) -> Self {
        self.stream_protocol = Some(stream_protocol);
        self
    }
}

/// Passive serializable subset of AI SDK `UIMessageStreamOptions`.
///
/// Function-valued options such as `generateMessageId`, `onFinish`, `messageMetadata`, and
/// `onError` are intentionally not represented here because they are runtime callbacks, not data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageStreamOptions<Message = UiMessage> {
    /// Original messages. When present, AI SDK assumes persistence mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_messages: Option<Vec<Message>>,
    /// Whether reasoning parts should be sent to the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_reasoning: Option<bool>,
    /// Whether source parts should be sent to the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_sources: Option<bool>,
    /// Whether the finish event should be sent to the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_finish: Option<bool>,
    /// Whether the message start event should be sent to the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_start: Option<bool>,
}

impl<Message> Default for UiMessageStreamOptions<Message> {
    fn default() -> Self {
        Self {
            original_messages: None,
            send_reasoning: None,
            send_sources: None,
            send_finish: None,
            send_start: None,
        }
    }
}

impl<Message> UiMessageStreamOptions<Message> {
    /// Create empty UI message stream options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the original messages.
    pub fn with_original_messages(
        mut self,
        original_messages: impl IntoIterator<Item = Message>,
    ) -> Self {
        self.original_messages = Some(original_messages.into_iter().collect());
        self
    }

    /// Set whether reasoning parts should be sent.
    pub fn with_send_reasoning(mut self, send_reasoning: bool) -> Self {
        self.send_reasoning = Some(send_reasoning);
        self
    }

    /// Set whether source parts should be sent.
    pub fn with_send_sources(mut self, send_sources: bool) -> Self {
        self.send_sources = Some(send_sources);
        self
    }

    /// Set whether finish events should be sent.
    pub fn with_send_finish(mut self, send_finish: bool) -> Self {
        self.send_finish = Some(send_finish);
        self
    }

    /// Set whether start events should be sent.
    pub fn with_send_start(mut self, send_start: bool) -> Self {
        self.send_start = Some(send_start);
        self
    }
}

/// AI SDK export spelling for `UIMessageStreamOptions`.
pub type UIMessageStreamOptions<Message = UiMessage> = UiMessageStreamOptions<Message>;

/// Partial object event from AI SDK `ObjectStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectStreamObjectPart<PARTIAL = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ObjectStreamObjectPartMarker,
    /// Current partial object snapshot.
    pub object: PARTIAL,
}

impl<PARTIAL> ObjectStreamObjectPart<PARTIAL> {
    /// Create a partial-object stream part.
    pub fn new(object: PARTIAL) -> Self {
        Self {
            marker: ObjectStreamObjectPartMarker::Marker,
            object,
        }
    }

    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub const fn r#type(&self) -> &'static str {
        "object"
    }
}

/// Text-delta event from AI SDK `ObjectStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ObjectStreamTextDeltaPart {
    #[serde(rename = "type", default)]
    marker: ObjectStreamTextDeltaPartMarker,
    /// JSON text delta.
    pub text_delta: String,
}

impl ObjectStreamTextDeltaPart {
    /// Create an object-stream text delta.
    pub fn new(text_delta: impl Into<String>) -> Self {
        Self {
            marker: ObjectStreamTextDeltaPartMarker::Marker,
            text_delta: text_delta.into(),
        }
    }

    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub const fn r#type(&self) -> &'static str {
        "text-delta"
    }
}

/// Error event from AI SDK `ObjectStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectStreamErrorPart {
    #[serde(rename = "type", default)]
    marker: ObjectStreamErrorPartMarker,
    /// Error payload.
    pub error: JSONValue,
}

impl ObjectStreamErrorPart {
    /// Create an object-stream error part.
    pub fn new(error: impl Into<JSONValue>) -> Self {
        Self {
            marker: ObjectStreamErrorPartMarker::Marker,
            error: error.into(),
        }
    }

    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub const fn r#type(&self) -> &'static str {
        "error"
    }
}

/// Finish event from AI SDK `ObjectStreamPart`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ObjectStreamFinishPart {
    #[serde(rename = "type", default)]
    marker: ObjectStreamFinishPartMarker,
    /// Unified finish reason.
    pub finish_reason: FinishReason,
    /// Token usage.
    pub usage: LanguageModelUsage,
    /// Response metadata.
    pub response: LanguageModelResponseMetadata,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl ObjectStreamFinishPart {
    /// Create an object-stream finish part.
    pub fn new(
        finish_reason: FinishReason,
        usage: LanguageModelUsage,
        response: LanguageModelResponseMetadata,
    ) -> Self {
        Self {
            marker: ObjectStreamFinishPartMarker::Marker,
            finish_reason,
            usage,
            response,
            provider_metadata: None,
        }
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub const fn r#type(&self) -> &'static str {
        "finish"
    }
}

/// AI SDK `ObjectStreamPart` union from `generate-object/stream-object-result.ts`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ObjectStreamPart<PARTIAL = JSONValue> {
    /// Partial object snapshot.
    Object(ObjectStreamObjectPart<PARTIAL>),
    /// JSON text delta.
    TextDelta(ObjectStreamTextDeltaPart),
    /// Stream error payload.
    Error(ObjectStreamErrorPart),
    /// Terminal stream metadata.
    Finish(ObjectStreamFinishPart),
}

impl<PARTIAL> ObjectStreamPart<PARTIAL> {
    /// Return the AI SDK `ObjectStreamPart` discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Object(part) => part.r#type(),
            Self::TextDelta(part) => part.r#type(),
            Self::Error(part) => part.r#type(),
            Self::Finish(part) => part.r#type(),
        }
    }
}

impl<PARTIAL> From<ObjectStreamObjectPart<PARTIAL>> for ObjectStreamPart<PARTIAL> {
    fn from(value: ObjectStreamObjectPart<PARTIAL>) -> Self {
        Self::Object(value)
    }
}

impl<PARTIAL> From<ObjectStreamTextDeltaPart> for ObjectStreamPart<PARTIAL> {
    fn from(value: ObjectStreamTextDeltaPart) -> Self {
        Self::TextDelta(value)
    }
}

impl<PARTIAL> From<ObjectStreamErrorPart> for ObjectStreamPart<PARTIAL> {
    fn from(value: ObjectStreamErrorPart) -> Self {
        Self::Error(value)
    }
}

impl<PARTIAL> From<ObjectStreamFinishPart> for ObjectStreamPart<PARTIAL> {
    fn from(value: ObjectStreamFinishPart) -> Self {
        Self::Finish(value)
    }
}

fn deserialize_ui_message_data_chunk_type<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.starts_with("data-") {
        Ok(value)
    } else {
        Err(serde::de::Error::custom(format!(
            "expected UI message data chunk type to start with `data-`, got `{value}`"
        )))
    }
}

macro_rules! ui_message_id_provider_metadata_chunk {
    ($name:ident, $marker:ident, $type:literal) => {
        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        #[serde(rename_all = "camelCase")]
        pub struct $name {
            #[serde(rename = "type")]
            marker: $marker,
            /// Chunk id.
            pub id: String,
            /// Provider-specific metadata.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub provider_metadata: Option<ProviderMetadata>,
        }

        impl $name {
            /// Create a UI message stream chunk.
            pub fn new(id: impl Into<String>) -> Self {
                Self {
                    marker: $marker::Marker,
                    id: id.into(),
                    provider_metadata: None,
                }
            }

            /// Attach provider metadata.
            pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
                self.provider_metadata = Some(provider_metadata);
                self
            }

            /// Return the AI SDK UI message chunk discriminator.
            pub const fn r#type(&self) -> &'static str {
                $type
            }
        }
    };
}

ui_message_id_provider_metadata_chunk!(
    UiMessageTextStartChunk,
    UiMessageTextStartChunkMarker,
    "text-start"
);
ui_message_id_provider_metadata_chunk!(
    UiMessageTextEndChunk,
    UiMessageTextEndChunkMarker,
    "text-end"
);
ui_message_id_provider_metadata_chunk!(
    UiMessageReasoningStartChunk,
    UiMessageReasoningStartChunkMarker,
    "reasoning-start"
);
ui_message_id_provider_metadata_chunk!(
    UiMessageReasoningEndChunk,
    UiMessageReasoningEndChunkMarker,
    "reasoning-end"
);

/// Text delta chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageTextDeltaChunk {
    #[serde(rename = "type")]
    marker: UiMessageTextDeltaChunkMarker,
    /// Text block id.
    pub id: String,
    /// Text delta.
    pub delta: String,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageTextDeltaChunk {
    /// Create a UI text delta chunk.
    pub fn new(id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            marker: UiMessageTextDeltaChunkMarker::Marker,
            id: id.into(),
            delta: delta.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "text-delta"
    }
}

/// Reasoning delta chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageReasoningDeltaChunk {
    #[serde(rename = "type")]
    marker: UiMessageReasoningDeltaChunkMarker,
    /// Reasoning block id.
    pub id: String,
    /// Reasoning delta.
    pub delta: String,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageReasoningDeltaChunk {
    /// Create a UI reasoning delta chunk.
    pub fn new(id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            marker: UiMessageReasoningDeltaChunkMarker::Marker,
            id: id.into(),
            delta: delta.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning-delta"
    }
}

/// Custom chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageCustomChunk {
    #[serde(rename = "type")]
    marker: UiMessageCustomChunkMarker,
    /// Custom kind, normally `{provider}.{kind}`.
    pub kind: String,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageCustomChunk {
    /// Create a custom UI chunk.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            marker: UiMessageCustomChunkMarker::Marker,
            kind: kind.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "custom"
    }
}

/// Error chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageErrorChunk {
    #[serde(rename = "type")]
    marker: UiMessageErrorChunkMarker,
    /// Error text.
    pub error_text: String,
}

impl UiMessageErrorChunk {
    /// Create an error UI chunk.
    pub fn new(error_text: impl Into<String>) -> Self {
        Self {
            marker: UiMessageErrorChunkMarker::Marker,
            error_text: error_text.into(),
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "error"
    }
}

/// Tool input start chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolInputStartChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolInputStartChunkMarker,
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl UiMessageToolInputStartChunk {
    /// Create a tool input start chunk.
    pub fn new(tool_call_id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolInputStartChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            title: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-input-start"
    }
}

/// Tool input delta chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolInputDeltaChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolInputDeltaChunkMarker,
    pub tool_call_id: String,
    pub input_text_delta: String,
}

impl UiMessageToolInputDeltaChunk {
    /// Create a tool input delta chunk.
    pub fn new(tool_call_id: impl Into<String>, input_text_delta: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolInputDeltaChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            input_text_delta: input_text_delta.into(),
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-input-delta"
    }
}

/// Tool input available chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolInputAvailableChunk<INPUT = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageToolInputAvailableChunkMarker,
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: INPUT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl<INPUT> UiMessageToolInputAvailableChunk<INPUT> {
    /// Create a tool input available chunk.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: INPUT,
    ) -> Self {
        Self {
            marker: UiMessageToolInputAvailableChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input,
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            title: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-input-available"
    }
}

/// Tool input error chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolInputErrorChunk<INPUT = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageToolInputErrorChunkMarker,
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: INPUT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    pub error_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl<INPUT> UiMessageToolInputErrorChunk<INPUT> {
    /// Create a tool input error chunk.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: INPUT,
        error_text: impl Into<String>,
    ) -> Self {
        Self {
            marker: UiMessageToolInputErrorChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input,
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            error_text: error_text.into(),
            title: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-input-error"
    }
}

/// Tool approval request chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolApprovalRequestChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolApprovalRequestChunkMarker,
    pub approval_id: String,
    pub tool_call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_automatic: Option<bool>,
}

impl UiMessageToolApprovalRequestChunk {
    /// Create a tool approval request chunk.
    pub fn new(approval_id: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolApprovalRequestChunkMarker::Marker,
            approval_id: approval_id.into(),
            tool_call_id: tool_call_id.into(),
            is_automatic: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-request"
    }
}

/// Tool approval response chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolApprovalResponseChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolApprovalResponseChunkMarker,
    pub approval_id: String,
    pub approved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageToolApprovalResponseChunk {
    /// Create a tool approval response chunk.
    pub fn new(approval_id: impl Into<String>, approved: bool) -> Self {
        Self {
            marker: UiMessageToolApprovalResponseChunkMarker::Marker,
            approval_id: approval_id.into(),
            approved,
            reason: None,
            provider_executed: None,
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-response"
    }
}

/// Tool output available chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolOutputAvailableChunk<OUTPUT = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageToolOutputAvailableChunkMarker,
    pub tool_call_id: String,
    pub output: OUTPUT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
}

impl<OUTPUT> UiMessageToolOutputAvailableChunk<OUTPUT> {
    /// Create a tool output available chunk.
    pub fn new(tool_call_id: impl Into<String>, output: OUTPUT) -> Self {
        Self {
            marker: UiMessageToolOutputAvailableChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            output,
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            preliminary: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-output-available"
    }
}

/// Tool output error chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolOutputErrorChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolOutputErrorChunkMarker,
    pub tool_call_id: String,
    pub error_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
}

impl UiMessageToolOutputErrorChunk {
    /// Create a tool output error chunk.
    pub fn new(tool_call_id: impl Into<String>, error_text: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolOutputErrorChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
            error_text: error_text.into(),
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-output-error"
    }
}

/// Tool output denied chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageToolOutputDeniedChunk {
    #[serde(rename = "type")]
    marker: UiMessageToolOutputDeniedChunkMarker,
    pub tool_call_id: String,
}

impl UiMessageToolOutputDeniedChunk {
    /// Create a tool output denied chunk.
    pub fn new(tool_call_id: impl Into<String>) -> Self {
        Self {
            marker: UiMessageToolOutputDeniedChunkMarker::Marker,
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-output-denied"
    }
}

/// URL source chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageSourceUrlChunk {
    #[serde(rename = "type")]
    marker: UiMessageSourceUrlChunkMarker,
    pub source_id: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageSourceUrlChunk {
    /// Create a URL source chunk.
    pub fn new(source_id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            marker: UiMessageSourceUrlChunkMarker::Marker,
            source_id: source_id.into(),
            url: url.into(),
            title: None,
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "source-url"
    }
}

/// Document source chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageSourceDocumentChunk {
    #[serde(rename = "type")]
    marker: UiMessageSourceDocumentChunkMarker,
    pub source_id: String,
    pub media_type: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageSourceDocumentChunk {
    /// Create a document source chunk.
    pub fn new(
        source_id: impl Into<String>,
        media_type: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        Self {
            marker: UiMessageSourceDocumentChunkMarker::Marker,
            source_id: source_id.into(),
            media_type: media_type.into(),
            title: title.into(),
            filename: None,
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "source-document"
    }
}

/// File chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageFileChunk {
    #[serde(rename = "type")]
    marker: UiMessageFileChunkMarker,
    pub url: String,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageFileChunk {
    /// Create a file chunk.
    pub fn new(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            marker: UiMessageFileChunkMarker::Marker,
            url: url.into(),
            media_type: media_type.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "file"
    }
}

/// Reasoning file chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageReasoningFileChunk {
    #[serde(rename = "type")]
    marker: UiMessageReasoningFileChunkMarker,
    pub url: String,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl UiMessageReasoningFileChunk {
    /// Create a reasoning file chunk.
    pub fn new(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            marker: UiMessageReasoningFileChunkMarker::Marker,
            url: url.into(),
            media_type: media_type.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning-file"
    }
}

/// Data chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageDataChunk<DATA = JSONValue> {
    /// Full data discriminator, e.g. `data-weather`.
    #[serde(
        rename = "type",
        deserialize_with = "deserialize_ui_message_data_chunk_type"
    )]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub data: DATA,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transient: Option<bool>,
}

impl<DATA> UiMessageDataChunk<DATA> {
    /// Create a data UI chunk from the suffix after `data-`.
    pub fn new(data_type: impl AsRef<str>, data: DATA) -> Self {
        Self {
            kind: format!("data-{}", data_type.as_ref().trim_start_matches("data-")),
            id: None,
            data,
            transient: None,
        }
    }

    /// Return the full AI SDK UI message chunk discriminator.
    pub fn r#type(&self) -> &str {
        &self.kind
    }

    /// Return the suffix after `data-`, if the discriminator is valid.
    pub fn data_type(&self) -> Option<&str> {
        self.kind.strip_prefix("data-")
    }
}

/// Start-step chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiMessageStartStepChunk {
    #[serde(rename = "type")]
    marker: UiMessageStartStepChunkMarker,
}

impl UiMessageStartStepChunk {
    /// Create a start-step chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageStartStepChunkMarker::Marker,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "start-step"
    }
}

impl Default for UiMessageStartStepChunk {
    fn default() -> Self {
        Self::new()
    }
}

/// Finish-step chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiMessageFinishStepChunk {
    #[serde(rename = "type")]
    marker: UiMessageFinishStepChunkMarker,
}

impl UiMessageFinishStepChunk {
    /// Create a finish-step chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageFinishStepChunkMarker::Marker,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "finish-step"
    }
}

impl Default for UiMessageFinishStepChunk {
    fn default() -> Self {
        Self::new()
    }
}

/// Start chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageStartChunk<METADATA = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageStartChunkMarker,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_metadata: Option<METADATA>,
}

impl<METADATA> UiMessageStartChunk<METADATA> {
    /// Create a start chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageStartChunkMarker::Marker,
            message_id: None,
            message_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "start"
    }
}

impl<METADATA> Default for UiMessageStartChunk<METADATA> {
    fn default() -> Self {
        Self::new()
    }
}

/// Finish chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageFinishChunk<METADATA = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageFinishChunkMarker,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_metadata: Option<METADATA>,
}

impl<METADATA> UiMessageFinishChunk<METADATA> {
    /// Create a finish chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageFinishChunkMarker::Marker,
            finish_reason: None,
            message_metadata: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "finish"
    }
}

impl<METADATA> Default for UiMessageFinishChunk<METADATA> {
    fn default() -> Self {
        Self::new()
    }
}

/// Abort chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiMessageAbortChunk {
    #[serde(rename = "type")]
    marker: UiMessageAbortChunkMarker,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl UiMessageAbortChunk {
    /// Create an abort chunk.
    pub fn new() -> Self {
        Self {
            marker: UiMessageAbortChunkMarker::Marker,
            reason: None,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "abort"
    }
}

impl Default for UiMessageAbortChunk {
    fn default() -> Self {
        Self::new()
    }
}

/// Message metadata chunk from AI SDK `UIMessageChunk`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UiMessageMetadataChunk<METADATA = JSONValue> {
    #[serde(rename = "type")]
    marker: UiMessageMetadataChunkMarker,
    pub message_metadata: METADATA,
}

impl<METADATA> UiMessageMetadataChunk<METADATA> {
    /// Create a message metadata chunk.
    pub fn new(message_metadata: METADATA) -> Self {
        Self {
            marker: UiMessageMetadataChunkMarker::Marker,
            message_metadata,
        }
    }

    /// Return the AI SDK UI message chunk discriminator.
    pub const fn r#type(&self) -> &'static str {
        "message-metadata"
    }
}

/// AI SDK `UIMessageChunk` union from `ui-message-stream/ui-message-chunks.ts`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UiMessageChunk<METADATA = JSONValue, DATA = JSONValue> {
    TextStart(UiMessageTextStartChunk),
    TextDelta(UiMessageTextDeltaChunk),
    TextEnd(UiMessageTextEndChunk),
    ReasoningStart(UiMessageReasoningStartChunk),
    ReasoningDelta(UiMessageReasoningDeltaChunk),
    ReasoningEnd(UiMessageReasoningEndChunk),
    Custom(UiMessageCustomChunk),
    Error(UiMessageErrorChunk),
    ToolInputStart(UiMessageToolInputStartChunk),
    ToolInputDelta(UiMessageToolInputDeltaChunk),
    ToolInputAvailable(UiMessageToolInputAvailableChunk<DATA>),
    ToolInputError(UiMessageToolInputErrorChunk<DATA>),
    ToolApprovalRequest(UiMessageToolApprovalRequestChunk),
    ToolApprovalResponse(UiMessageToolApprovalResponseChunk),
    ToolOutputAvailable(UiMessageToolOutputAvailableChunk<DATA>),
    ToolOutputError(UiMessageToolOutputErrorChunk),
    ToolOutputDenied(UiMessageToolOutputDeniedChunk),
    SourceUrl(UiMessageSourceUrlChunk),
    SourceDocument(UiMessageSourceDocumentChunk),
    File(UiMessageFileChunk),
    ReasoningFile(UiMessageReasoningFileChunk),
    Data(UiMessageDataChunk<DATA>),
    StartStep(UiMessageStartStepChunk),
    FinishStep(UiMessageFinishStepChunk),
    Start(UiMessageStartChunk<METADATA>),
    Finish(UiMessageFinishChunk<METADATA>),
    Abort(UiMessageAbortChunk),
    MessageMetadata(UiMessageMetadataChunk<METADATA>),
}

impl<METADATA, DATA> UiMessageChunk<METADATA, DATA> {
    /// Return the AI SDK UI message chunk discriminator.
    pub fn r#type(&self) -> &str {
        match self {
            Self::TextStart(part) => part.r#type(),
            Self::TextDelta(part) => part.r#type(),
            Self::TextEnd(part) => part.r#type(),
            Self::ReasoningStart(part) => part.r#type(),
            Self::ReasoningDelta(part) => part.r#type(),
            Self::ReasoningEnd(part) => part.r#type(),
            Self::Custom(part) => part.r#type(),
            Self::Error(part) => part.r#type(),
            Self::ToolInputStart(part) => part.r#type(),
            Self::ToolInputDelta(part) => part.r#type(),
            Self::ToolInputAvailable(part) => part.r#type(),
            Self::ToolInputError(part) => part.r#type(),
            Self::ToolApprovalRequest(part) => part.r#type(),
            Self::ToolApprovalResponse(part) => part.r#type(),
            Self::ToolOutputAvailable(part) => part.r#type(),
            Self::ToolOutputError(part) => part.r#type(),
            Self::ToolOutputDenied(part) => part.r#type(),
            Self::SourceUrl(part) => part.r#type(),
            Self::SourceDocument(part) => part.r#type(),
            Self::File(part) => part.r#type(),
            Self::ReasoningFile(part) => part.r#type(),
            Self::Data(part) => part.r#type(),
            Self::StartStep(part) => part.r#type(),
            Self::FinishStep(part) => part.r#type(),
            Self::Start(part) => part.r#type(),
            Self::Finish(part) => part.r#type(),
            Self::Abort(part) => part.r#type(),
            Self::MessageMetadata(part) => part.r#type(),
        }
    }
}

impl<METADATA, DATA> From<UiMessageTextStartChunk> for UiMessageChunk<METADATA, DATA> {
    fn from(value: UiMessageTextStartChunk) -> Self {
        Self::TextStart(value)
    }
}

impl<METADATA, DATA> From<UiMessageTextDeltaChunk> for UiMessageChunk<METADATA, DATA> {
    fn from(value: UiMessageTextDeltaChunk) -> Self {
        Self::TextDelta(value)
    }
}

impl<METADATA, DATA> From<UiMessageDataChunk<DATA>> for UiMessageChunk<METADATA, DATA> {
    fn from(value: UiMessageDataChunk<DATA>) -> Self {
        Self::Data(value)
    }
}

impl<METADATA, DATA> From<UiMessageFinishChunk<METADATA>> for UiMessageChunk<METADATA, DATA> {
    fn from(value: UiMessageFinishChunk<METADATA>) -> Self {
        Self::Finish(value)
    }
}

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

/// Passive representation of AI SDK `StopCondition`.
///
/// The TypeScript surface accepts predicates/functions. Rust exposes the built-in
/// conditions as symbolic data and evaluates only those built-ins. `Custom` is a
/// transport lane for application-owned metadata and never evaluates to `true`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum StopCondition {
    /// Equivalent to AI SDK `isStepCount(stepCount)`.
    StepCount {
        /// Number of completed steps required for the condition to match.
        #[serde(rename = "stepCount", alias = "step_count", alias = "maxSteps")]
        step_count: usize,
    },
    /// Equivalent to AI SDK `isLoopFinished()`, which never stops by itself.
    LoopFinished,
    /// Equivalent to AI SDK `hasToolCall(...toolNames)`.
    ToolCall {
        /// Tool names that should stop the loop when present in the latest step.
        #[serde(rename = "toolNames", alias = "tool_names")]
        tool_names: Vec<String>,
    },
    /// Application-owned symbolic condition metadata.
    Custom {
        /// Opaque condition payload.
        value: JSONValue,
    },
}

impl StopCondition {
    /// Create a step-count stop condition.
    pub const fn is_step_count(step_count: usize) -> Self {
        Self::StepCount { step_count }
    }

    /// Create a condition that never stops the loop by itself.
    pub const fn is_loop_finished() -> Self {
        Self::LoopFinished
    }

    /// Create a condition that matches tool calls in the latest step.
    pub fn has_tool_call(tool_names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::ToolCall {
            tool_names: tool_names.into_iter().map(Into::into).collect(),
        }
    }

    /// Create an application-owned custom condition.
    pub fn custom(value: impl Into<JSONValue>) -> Self {
        Self::Custom {
            value: value.into(),
        }
    }

    /// Evaluate the built-in stop condition against completed steps.
    pub fn is_met<NAME, INPUT, OUTPUT>(
        &self,
        steps: &[GenerateTextStepResult<NAME, INPUT, OUTPUT>],
    ) -> bool
    where
        NAME: AsRef<str>,
    {
        match self {
            Self::StepCount { step_count } => steps.len() == *step_count,
            Self::LoopFinished | Self::Custom { .. } => false,
            Self::ToolCall { tool_names } => {
                let Some(step) = steps.last() else {
                    return false;
                };
                step.tool_calls.iter().any(|tool_call| {
                    tool_names
                        .iter()
                        .any(|tool_name| tool_name == tool_call.tool_name.as_ref())
                })
            }
        }
    }
}

/// Create a step-count stop condition.
pub const fn is_step_count(step_count: usize) -> StopCondition {
    StopCondition::is_step_count(step_count)
}

/// Deprecated AI SDK `stepCountIs` helper alias.
#[deprecated(note = "Use is_step_count instead.")]
pub const fn step_count_is(step_count: usize) -> StopCondition {
    is_step_count(step_count)
}

/// Create a condition that never stops the loop by itself.
pub const fn is_loop_finished() -> StopCondition {
    StopCondition::is_loop_finished()
}

/// Create a condition that matches tool calls in the latest step.
pub fn has_tool_call(tool_names: impl IntoIterator<Item = impl Into<String>>) -> StopCondition {
    StopCondition::has_tool_call(tool_names)
}

/// Evaluate built-in stop conditions, returning true when any condition matches.
pub fn is_stop_condition_met<NAME, INPUT, OUTPUT>(
    stop_conditions: &[StopCondition],
    steps: &[GenerateTextStepResult<NAME, INPUT, OUTPUT>],
) -> bool
where
    NAME: AsRef<str>,
{
    stop_conditions
        .iter()
        .any(|condition| condition.is_met(steps))
}

fn tool_name(tool: &Tool) -> &str {
    match tool {
        Tool::Function { function } => function.name.as_str(),
        Tool::ProviderDefined(tool) => tool.name.as_str(),
    }
}

/// Filter tools to the active tool names, matching AI SDK `filterActiveTools`.
pub fn filter_active_tools<N>(
    tools: Option<&[Tool]>,
    active_tools: Option<&[N]>,
) -> Option<Vec<Tool>>
where
    N: AsRef<str>,
{
    let tools = tools?;
    let Some(active_tools) = active_tools else {
        return Some(tools.to_vec());
    };

    Some(
        tools
            .iter()
            .filter(|tool| {
                active_tools
                    .iter()
                    .any(|active_tool| active_tool.as_ref() == tool_name(tool))
            })
            .cloned()
            .collect(),
    )
}

/// AI SDK `experimental_filterActiveTools` helper alias using Rust naming.
pub fn experimental_filter_active_tools<N>(
    tools: Option<&[Tool]>,
    active_tools: Option<&[N]>,
) -> Option<Vec<Tool>>
where
    N: AsRef<str>,
{
    filter_active_tools(tools, active_tools)
}

/// Reasoning pruning strategy for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PruneReasoningMode {
    /// Remove reasoning from all assistant messages.
    All,
    /// Remove reasoning from all assistant messages except the last message.
    BeforeLastMessage,
    /// Keep reasoning parts.
    #[default]
    None,
}

/// Empty-message handling for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PruneEmptyMessagesMode {
    /// Keep messages even when their content is empty after pruning.
    Keep,
    /// Remove messages whose content is empty after pruning.
    #[default]
    Remove,
}

/// Tool pruning scope for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PruneToolCallMode {
    /// Prune matching tool parts from all messages.
    All,
    /// Prune matching tool parts from all messages except the last message.
    BeforeLastMessage,
    /// Prune matching tool parts before the last `count` messages.
    BeforeLastMessages {
        /// Number of trailing messages to keep.
        count: usize,
    },
}

/// One tool-call pruning rule for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PruneToolCallRule {
    /// Scope that decides which messages are eligible for pruning.
    pub mode: PruneToolCallMode,
    /// Optional tool-name allowlist. Parts for tools outside this list are kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
}

impl PruneToolCallRule {
    /// Prune matching tool parts from all messages.
    pub const fn all() -> Self {
        Self {
            mode: PruneToolCallMode::All,
            tools: None,
        }
    }

    /// Prune matching tool parts before the last message.
    pub const fn before_last_message() -> Self {
        Self {
            mode: PruneToolCallMode::BeforeLastMessage,
            tools: None,
        }
    }

    /// Prune matching tool parts before the last `count` messages.
    pub const fn before_last_messages(count: usize) -> Self {
        Self {
            mode: PruneToolCallMode::BeforeLastMessages { count },
            tools: None,
        }
    }

    /// Limit pruning to specific tool names.
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    fn keep_last_messages_count(&self) -> Option<usize> {
        match self.mode {
            PruneToolCallMode::All => None,
            PruneToolCallMode::BeforeLastMessage => Some(1),
            PruneToolCallMode::BeforeLastMessages { count } => Some(count),
        }
    }
}

/// Options for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PruneMessagesOptions {
    /// How to remove reasoning content from assistant messages.
    #[serde(default)]
    pub reasoning: PruneReasoningMode,
    /// Tool-call/result/approval pruning rules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<PruneToolCallRule>,
    /// Whether to keep or remove messages that become empty.
    #[serde(default)]
    pub empty_messages: PruneEmptyMessagesMode,
}

impl PruneMessagesOptions {
    /// Create options with AI SDK defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set reasoning pruning mode.
    pub const fn with_reasoning(mut self, reasoning: PruneReasoningMode) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Set tool pruning rules.
    pub fn with_tool_calls(mut self, tool_calls: Vec<PruneToolCallRule>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Set empty-message handling.
    pub const fn with_empty_messages(mut self, empty_messages: PruneEmptyMessagesMode) -> Self {
        self.empty_messages = empty_messages;
        self
    }
}

fn is_tool_name_outside_rule(tools: Option<&Vec<String>>, tool_name: Option<&str>) -> bool {
    match tools {
        Some(tools) => match tool_name {
            Some(tool_name) => !tools.iter().any(|tool| tool == tool_name),
            None => true,
        },
        None => false,
    }
}

fn should_keep_tool_part(
    rule: &PruneToolCallRule,
    id: &str,
    tool_name: Option<&str>,
    kept_ids: &std::collections::HashSet<String>,
) -> bool {
    kept_ids.contains(id) || is_tool_name_outside_rule(rule.tools.as_ref(), tool_name)
}

fn collect_kept_tool_part_ids(
    message: &ModelMessage,
    kept_tool_call_ids: &mut std::collections::HashSet<String>,
    kept_approval_ids: &mut std::collections::HashSet<String>,
) {
    match message {
        ModelMessage::Assistant(message) => {
            let AssistantContent::Parts(parts) = &message.content else {
                return;
            };
            for part in parts {
                match part {
                    AssistantContentPart::ToolCall(part) => {
                        kept_tool_call_ids.insert(part.tool_call_id.clone());
                    }
                    AssistantContentPart::ToolResult(part) => {
                        kept_tool_call_ids.insert(part.tool_call_id.clone());
                    }
                    AssistantContentPart::ToolApprovalRequest(part) => {
                        kept_approval_ids.insert(part.approval_id.clone());
                    }
                    _ => {}
                }
            }
        }
        ModelMessage::Tool(message) => {
            for part in &message.content {
                match part {
                    ToolContentPart::ToolResult(part) => {
                        kept_tool_call_ids.insert(part.tool_call_id.clone());
                    }
                    ToolContentPart::ToolApprovalResponse(part) => {
                        kept_approval_ids.insert(part.approval_id.clone());
                    }
                }
            }
        }
        _ => {}
    }
}

fn prune_assistant_tool_parts(
    parts: Vec<AssistantContentPart>,
    rule: &PruneToolCallRule,
    kept_tool_call_ids: &std::collections::HashSet<String>,
    kept_approval_ids: &std::collections::HashSet<String>,
) -> Vec<AssistantContentPart> {
    let mut tool_call_id_to_tool_name = HashMap::<String, String>::new();
    let mut approval_id_to_tool_name = HashMap::<String, String>::new();

    parts
        .into_iter()
        .filter(|part| match part {
            AssistantContentPart::ToolCall(part) => {
                tool_call_id_to_tool_name.insert(part.tool_call_id.clone(), part.tool_name.clone());
                should_keep_tool_part(
                    rule,
                    &part.tool_call_id,
                    Some(part.tool_name.as_str()),
                    kept_tool_call_ids,
                )
            }
            AssistantContentPart::ToolResult(part) => should_keep_tool_part(
                rule,
                &part.tool_call_id,
                Some(part.tool_name.as_str()),
                kept_tool_call_ids,
            ),
            AssistantContentPart::ToolApprovalRequest(part) => {
                let tool_name = tool_call_id_to_tool_name
                    .get(&part.tool_call_id)
                    .map(String::as_str);
                if let Some(tool_name) = tool_name {
                    approval_id_to_tool_name
                        .insert(part.approval_id.clone(), tool_name.to_string());
                }
                should_keep_tool_part(rule, &part.approval_id, tool_name, kept_approval_ids)
            }
            _ => true,
        })
        .collect()
}

fn prune_tool_message_parts(
    parts: Vec<ToolContentPart>,
    rule: &PruneToolCallRule,
    kept_tool_call_ids: &std::collections::HashSet<String>,
    kept_approval_ids: &std::collections::HashSet<String>,
) -> Vec<ToolContentPart> {
    let approval_id_to_tool_name = HashMap::<String, String>::new();

    parts
        .into_iter()
        .filter(|part| match part {
            ToolContentPart::ToolResult(part) => should_keep_tool_part(
                rule,
                &part.tool_call_id,
                Some(part.tool_name.as_str()),
                kept_tool_call_ids,
            ),
            ToolContentPart::ToolApprovalResponse(part) => {
                let tool_name = approval_id_to_tool_name
                    .get(&part.approval_id)
                    .map(String::as_str);
                should_keep_tool_part(rule, &part.approval_id, tool_name, kept_approval_ids)
            }
        })
        .collect()
}

fn model_message_is_empty(message: &ModelMessage) -> bool {
    match message {
        ModelMessage::System(message) => message.content.is_empty(),
        ModelMessage::User(message) => match &message.content {
            super::prompt::UserContent::Text(text) => text.is_empty(),
            super::prompt::UserContent::Parts(parts) => parts.is_empty(),
        },
        ModelMessage::Assistant(message) => match &message.content {
            AssistantContent::Text(text) => text.is_empty(),
            AssistantContent::Parts(parts) => parts.is_empty(),
        },
        ModelMessage::Tool(message) => message.content.is_empty(),
    }
}

/// Prune AI SDK-style model messages.
///
/// This mirrors the pure data behavior of `generate-text/prune-messages.ts`: reasoning parts can
/// be removed from assistant messages, tool call/result/approval parts can be pruned by recency
/// and tool name, and messages that become empty are removed by default.
pub fn prune_messages(
    mut messages: Vec<ModelMessage>,
    options: PruneMessagesOptions,
) -> Vec<ModelMessage> {
    if matches!(
        options.reasoning,
        PruneReasoningMode::All | PruneReasoningMode::BeforeLastMessage
    ) {
        let last_index = messages.len().saturating_sub(1);
        messages = messages
            .into_iter()
            .enumerate()
            .map(|(index, message)| {
                let ModelMessage::Assistant(mut assistant) = message else {
                    return message;
                };
                if options.reasoning == PruneReasoningMode::BeforeLastMessage && index == last_index
                {
                    return ModelMessage::Assistant(assistant);
                }
                let AssistantContent::Parts(parts) = assistant.content else {
                    return ModelMessage::Assistant(assistant);
                };

                assistant.content = AssistantContent::Parts(
                    parts
                        .into_iter()
                        .filter(|part| !matches!(part, AssistantContentPart::Reasoning(_)))
                        .collect(),
                );
                ModelMessage::Assistant(assistant)
            })
            .collect();
    }

    for rule in &options.tool_calls {
        let keep_last_messages_count = rule.keep_last_messages_count();
        let mut kept_tool_call_ids = std::collections::HashSet::new();
        let mut kept_approval_ids = std::collections::HashSet::new();

        if let Some(count) = keep_last_messages_count {
            for message in messages.iter().rev().take(count) {
                collect_kept_tool_part_ids(
                    message,
                    &mut kept_tool_call_ids,
                    &mut kept_approval_ids,
                );
            }
        }

        let prune_before_index = keep_last_messages_count
            .map(|count| messages.len().saturating_sub(count))
            .unwrap_or(messages.len());

        messages = messages
            .into_iter()
            .enumerate()
            .map(|(index, message)| {
                if index >= prune_before_index {
                    return message;
                }

                match message {
                    ModelMessage::Assistant(mut assistant) => {
                        let AssistantContent::Parts(parts) = assistant.content else {
                            return ModelMessage::Assistant(assistant);
                        };
                        assistant.content = AssistantContent::Parts(prune_assistant_tool_parts(
                            parts,
                            rule,
                            &kept_tool_call_ids,
                            &kept_approval_ids,
                        ));
                        ModelMessage::Assistant(assistant)
                    }
                    ModelMessage::Tool(mut tool) => {
                        tool.content = prune_tool_message_parts(
                            tool.content,
                            rule,
                            &kept_tool_call_ids,
                            &kept_approval_ids,
                        );
                        ModelMessage::Tool(tool)
                    }
                    other => other,
                }
            })
            .collect();
    }

    if options.empty_messages == PruneEmptyMessagesMode::Remove {
        messages.retain(|message| !model_message_is_empty(message));
    }

    messages
}

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

/// AI SDK tool approval status discriminator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ToolApprovalStatusType {
    /// The tool does not require approval.
    NotApplicable,
    /// The tool is automatically approved.
    Approved,
    /// The tool is automatically denied.
    Denied,
    /// The tool requires user approval.
    UserApproval,
}

/// Object-form AI SDK tool approval status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolApprovalStatusDetails {
    /// Approval status discriminator.
    #[serde(rename = "type")]
    pub status_type: ToolApprovalStatusType,
    /// Optional approval/denial reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl ToolApprovalStatusDetails {
    /// Create a detailed approval status.
    pub const fn new(status_type: ToolApprovalStatusType) -> Self {
        Self {
            status_type,
            reason: None,
        }
    }

    /// Attach an approval/denial reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// AI SDK tool approval status.
///
/// Upstream also treats `undefined` as not-applicable; Rust represents that with
/// `Option<ToolApprovalStatus>`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ToolApprovalStatus {
    /// String status form.
    Simple(ToolApprovalStatusType),
    /// Object status form.
    Detailed(ToolApprovalStatusDetails),
}

impl ToolApprovalStatus {
    /// Create a string-form not-applicable status.
    pub const fn not_applicable() -> Self {
        Self::Simple(ToolApprovalStatusType::NotApplicable)
    }

    /// Create a string-form approved status.
    pub const fn approved() -> Self {
        Self::Simple(ToolApprovalStatusType::Approved)
    }

    /// Create a string-form denied status.
    pub const fn denied() -> Self {
        Self::Simple(ToolApprovalStatusType::Denied)
    }

    /// Create a string-form user-approval status.
    pub const fn user_approval() -> Self {
        Self::Simple(ToolApprovalStatusType::UserApproval)
    }

    /// Create an object-form status with an optional reason.
    pub fn detailed(status_type: ToolApprovalStatusType, reason: Option<String>) -> Self {
        Self::Detailed(ToolApprovalStatusDetails {
            status_type,
            reason,
        })
    }
}

/// Static per-tool approval configuration.
///
/// AI SDK also accepts approval functions. Rust keeps executable callbacks out of
/// the spec layer and exposes the serializable per-tool status map honestly.
pub type ToolApprovalConfiguration = HashMap<String, ToolApprovalStatus>;

/// Passive options passed to a generic AI SDK tool approval function.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolApprovalDecisionContext<NAME = String, INPUT = JSONValue> {
    /// Tool call that needs approval.
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Tools available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool context snapshot.
    pub tools_context: Context,
    /// Runtime context snapshot.
    pub runtime_context: Context,
    /// Messages sent to the model before the assistant tool-call response.
    pub messages: Vec<ModelMessage>,
}

/// AI SDK `NoSuchToolError` data carried into tool-call repair callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NoSuchToolError {
    /// Tool name from the failed call.
    pub tool_name: String,
    /// Available tool names when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_tools: Option<Vec<String>>,
    /// Human-readable error message.
    pub message: String,
}

impl NoSuchToolError {
    /// Create a `NoSuchToolError` with the upstream default message shape.
    pub fn new(tool_name: impl Into<String>, available_tools: Option<Vec<String>>) -> Self {
        let tool_name = tool_name.into();
        let message = match available_tools.as_ref() {
            Some(available_tools) => format!(
                "Model tried to call unavailable tool '{tool_name}'. Available tools: {}.",
                available_tools.join(", ")
            ),
            None => format!(
                "Model tried to call unavailable tool '{tool_name}'. No tools are available."
            ),
        };

        Self {
            tool_name,
            available_tools,
            message,
        }
    }
}

fn ai_sdk_error_message(cause: Option<&JSONValue>) -> String {
    match cause {
        None | Some(JSONValue::Null) => "unknown error".to_string(),
        Some(JSONValue::String(message)) => message.clone(),
        Some(cause) => cause.to_string(),
    }
}

/// AI SDK `InvalidToolInputError` data carried into tool-call repair callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InvalidToolInputError {
    /// Tool name from the failed call.
    pub tool_name: String,
    /// Raw tool input text that failed parsing or validation.
    pub tool_input: String,
    /// Human-readable error message.
    pub message: String,
    /// Provider/application error payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
}

impl InvalidToolInputError {
    /// Create an invalid-tool-input error.
    pub fn new(
        tool_name: impl Into<String>,
        tool_input: impl Into<String>,
        cause: Option<JSONValue>,
    ) -> Self {
        let tool_name = tool_name.into();
        let message = format!(
            "Invalid input for tool {tool_name}: {}",
            ai_sdk_error_message(cause.as_ref())
        );

        Self {
            tool_name,
            tool_input: tool_input.into(),
            message,
            cause,
        }
    }
}

/// Passive repair error union accepted by AI SDK `ToolCallRepairFunction`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolCallRepairFunctionError {
    /// The requested tool name is not available.
    NoSuchTool(NoSuchToolError),
    /// The tool input failed schema validation or parsing.
    InvalidToolInput(InvalidToolInputError),
}

impl From<NoSuchToolError> for ToolCallRepairFunctionError {
    fn from(error: NoSuchToolError) -> Self {
        Self::NoSuchTool(error)
    }
}

impl From<InvalidToolInputError> for ToolCallRepairFunctionError {
    fn from(error: InvalidToolInputError) -> Self {
        Self::InvalidToolInput(error)
    }
}

/// AI SDK `ToolCallRepairError` wrapper thrown when repair itself fails.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallRepairError {
    /// Human-readable error message.
    pub message: String,
    /// Original parse/availability error that triggered repair.
    pub original_error: ToolCallRepairFunctionError,
    /// Provider/application repair failure payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<JSONValue>,
}

impl ToolCallRepairError {
    /// Create a repair error wrapper.
    pub fn new(original_error: ToolCallRepairFunctionError, cause: Option<JSONValue>) -> Self {
        let message = format!(
            "Error repairing tool call: {}",
            ai_sdk_error_message(cause.as_ref())
        );

        Self {
            message,
            original_error,
            cause,
        }
    }
}

/// Passive options passed to AI SDK `ToolCallRepairFunction`.
///
/// Upstream receives an `inputSchema(toolName)` function. Rust stores the known
/// input schemas by tool name instead of pretending to serialize that callback.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallRepairContext<NAME = String, INPUT = JSONValue> {
    /// Optional system prompt override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    /// Messages in the current generation step.
    pub messages: Vec<ModelMessage>,
    /// Tool call that failed to parse or validate.
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Tools available to the model.
    pub tools: Vec<Tool>,
    /// Input schemas keyed by tool name.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub input_schemas: HashMap<String, JSONSchema7>,
    /// Error that caused repair to be attempted.
    pub error: ToolCallRepairFunctionError,
}

/// Passive repair result returned by an AI SDK `ToolCallRepairFunction`.
pub type ToolCallRepairResult<NAME = String, INPUT = JSONValue> = Option<ToolCall<NAME, INPUT>>;

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
    /// Create a normalized usage payload from input and output token counts.
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens: Some(input_tokens),
            input_token_details: LanguageModelInputTokenDetails::default(),
            output_tokens: Some(output_tokens),
            output_token_details: LanguageModelOutputTokenDetails::default(),
            total_tokens: Some(input_tokens.saturating_add(output_tokens)),
            reasoning_tokens: None,
            cached_input_tokens: None,
            raw: None,
        }
    }

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

/// Optional raw response data returned by embedding and reranking helpers.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelCallResponseData {
    /// Response headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Raw response body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl ModelCallResponseData {
    /// Create an empty response data envelope.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach response headers.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }

    /// Attach a raw response body.
    pub fn with_body(mut self, body: JSONValue) -> Self {
        self.body = Some(body);
        self
    }
}

/// Value input accepted by AI SDK embed callback payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbedValue {
    /// Single embedded value.
    Single(String),
    /// Multiple embedded values.
    Many(Vec<String>),
}

impl From<String> for EmbedValue {
    fn from(value: String) -> Self {
        Self::Single(value)
    }
}

impl From<&str> for EmbedValue {
    fn from(value: &str) -> Self {
        Self::Single(value.to_string())
    }
}

impl From<Vec<String>> for EmbedValue {
    fn from(value: Vec<String>) -> Self {
        Self::Many(value)
    }
}

/// Embedding output accepted by AI SDK embed callback payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbedOutput {
    /// Single embedding vector.
    Single(Embedding),
    /// Multiple embedding vectors.
    Many(Vec<Embedding>),
}

impl From<Embedding> for EmbedOutput {
    fn from(value: Embedding) -> Self {
        Self::Single(value)
    }
}

impl From<Vec<Embedding>> for EmbedOutput {
    fn from(value: Vec<Embedding>) -> Self {
        Self::Many(value)
    }
}

/// Response data accepted by AI SDK embed callback payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbedResponseData {
    /// Single model-call response envelope.
    Single(ModelCallResponseData),
    /// Multiple model-call response envelopes.
    Many(Vec<Option<ModelCallResponseData>>),
}

impl From<ModelCallResponseData> for EmbedResponseData {
    fn from(value: ModelCallResponseData) -> Self {
        Self::Single(value)
    }
}

impl From<Vec<Option<ModelCallResponseData>>> for EmbedResponseData {
    fn from(value: Vec<Option<ModelCallResponseData>>) -> Self {
        Self::Many(value)
    }
}

/// Passive AI SDK-style result envelope for an `embed` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbedResult {
    /// Value that was embedded.
    pub value: String,
    /// Embedding vector for the value.
    pub embedding: Embedding,
    /// Embedding token usage.
    pub usage: EmbeddingModelUsage,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Optional raw response data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<ModelCallResponseData>,
}

impl EmbedResult {
    /// Create an embedding result.
    pub fn new(value: impl Into<String>, embedding: Embedding, usage: EmbeddingModelUsage) -> Self {
        Self {
            value: value.into(),
            embedding,
            usage,
            warnings: Vec::new(),
            provider_metadata: None,
            response: None,
        }
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Attach raw response data.
    pub fn with_response(mut self, response: ModelCallResponseData) -> Self {
        self.response = Some(response);
        self
    }
}

/// Passive AI SDK-style result envelope for an `embedMany` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbedManyResult {
    /// Values that were embedded.
    pub values: Vec<String>,
    /// Embeddings in the same order as `values`.
    pub embeddings: Vec<Embedding>,
    /// Embedding token usage.
    pub usage: EmbeddingModelUsage,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Optional raw response data for each underlying model call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responses: Option<Vec<Option<ModelCallResponseData>>>,
}

impl EmbedManyResult {
    /// Create an embed-many result.
    pub fn new(
        values: Vec<String>,
        embeddings: Vec<Embedding>,
        usage: EmbeddingModelUsage,
    ) -> Self {
        Self {
            values,
            embeddings,
            usage,
            warnings: Vec::new(),
            provider_metadata: None,
            responses: None,
        }
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Attach raw response data for underlying model calls.
    pub fn with_responses(mut self, responses: Vec<Option<ModelCallResponseData>>) -> Self {
        self.responses = Some(responses);
        self
    }
}

/// Event payload for AI SDK embed/embedMany `onStart` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbedStartEvent {
    /// Unique call id.
    pub call_id: String,
    /// Operation id, e.g. `ai.embed` or `ai.embedMany`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Value or values being embedded.
    pub value: EmbedValue,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Additional HTTP headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Provider-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// Event payload for AI SDK embed/embedMany `onFinish` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbedEndEvent {
    /// Unique call id.
    pub call_id: String,
    /// Operation id, e.g. `ai.embed` or `ai.embedMany`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Value or values that were embedded.
    pub value: EmbedValue,
    /// Resulting embedding or embeddings.
    pub embedding: EmbedOutput,
    /// Embedding token usage.
    pub usage: EmbeddingModelUsage,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Optional raw response data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<EmbedResponseData>,
}

/// Event payload for the start of an underlying embedding-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingModelCallStartEvent {
    /// Unique outer embed call id.
    pub call_id: String,
    /// Unique id for this underlying model invocation.
    pub embed_call_id: String,
    /// Operation id, e.g. `ai.embed.doEmbed`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Values being embedded in this model call.
    pub values: Vec<String>,
}

/// Event payload for the end of an underlying embedding-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingModelCallEndEvent {
    /// Unique outer embed call id.
    pub call_id: String,
    /// Unique id for this underlying model invocation.
    pub embed_call_id: String,
    /// Operation id, e.g. `ai.embed.doEmbed`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Values embedded in this model call.
    pub values: Vec<String>,
    /// Resulting embeddings from this model call.
    pub embeddings: Vec<Embedding>,
    /// Token usage for this model call.
    pub usage: EmbeddingModelUsage,
}

/// AI SDK-style response data returned by reranking helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankResponseMetadata {
    /// Response id when the provider sends one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Timestamp for the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    pub model_id: String,
    /// Response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Raw response body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl RerankResponseMetadata {
    /// Create rerank response metadata.
    pub fn new(timestamp: DateTime<Utc>, model_id: impl Into<String>) -> Self {
        Self {
            id: None,
            timestamp,
            model_id: model_id.into(),
            headers: None,
            body: None,
        }
    }

    /// Attach a response id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Attach response headers.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }

    /// Attach a raw response body.
    pub fn with_body(mut self, body: JSONValue) -> Self {
        self.body = Some(body);
        self
    }
}

/// Single ranking entry in an AI SDK-style rerank result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankRanking<VALUE = JSONValue> {
    /// Original input index.
    pub original_index: u32,
    /// Relevance score.
    pub score: f64,
    /// Reranked document value.
    pub document: VALUE,
}

impl<VALUE> RerankRanking<VALUE> {
    /// Create a rerank ranking entry.
    pub fn new(original_index: u32, score: f64, document: VALUE) -> Self {
        Self {
            original_index,
            score,
            document,
        }
    }
}

/// Passive AI SDK-style result envelope for a `rerank` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankResult<VALUE = JSONValue> {
    /// Original documents that were reranked.
    pub original_documents: Vec<VALUE>,
    /// Reranked documents sorted by descending relevance.
    pub reranked_documents: Vec<VALUE>,
    /// Ranking entries with original indices, scores, and documents.
    pub ranking: Vec<RerankRanking<VALUE>>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Response metadata.
    pub response: RerankResponseMetadata,
}

impl<VALUE: Clone> RerankResult<VALUE> {
    /// Create a rerank result from original documents, ranking entries, and response metadata.
    pub fn new(
        original_documents: Vec<VALUE>,
        ranking: Vec<RerankRanking<VALUE>>,
        response: RerankResponseMetadata,
    ) -> Self {
        let reranked_documents = ranking.iter().map(|entry| entry.document.clone()).collect();

        Self {
            original_documents,
            reranked_documents,
            ranking,
            provider_metadata: None,
            response,
        }
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

/// Event payload for AI SDK rerank `onStart` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankStartEvent {
    /// Unique call id.
    pub call_id: String,
    /// Operation id, normally `ai.rerank`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Documents being reranked.
    pub documents: Vec<JSONValue>,
    /// Query used for reranking.
    pub query: String,
    /// Number of top documents to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Additional HTTP headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Provider-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// Event payload for AI SDK rerank `onFinish` callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankEndEvent {
    /// Unique call id.
    pub call_id: String,
    /// Operation id, normally `ai.rerank`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Documents that were reranked.
    pub documents: Vec<JSONValue>,
    /// Query used for reranking.
    pub query: String,
    /// Ranking entries.
    pub ranking: Vec<RerankRanking<JSONValue>>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Optional provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Response metadata.
    pub response: RerankResponseMetadata,
}

/// Event payload for the start of an underlying reranking-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankingModelCallStartEvent {
    /// Unique outer rerank call id.
    pub call_id: String,
    /// Operation id, normally `ai.rerank.doRerank`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Documents being reranked.
    pub documents: Vec<JSONValue>,
    /// Document family, usually `text` or `object`.
    pub documents_type: String,
    /// Query used for reranking.
    pub query: String,
    /// Number of top documents to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
}

/// Ranking summary returned by an underlying reranking-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankingModelCallRanking {
    /// Original document index.
    pub index: u32,
    /// Provider relevance score.
    pub relevance_score: f64,
}

impl RerankingModelCallRanking {
    /// Create a reranking model-call ranking entry.
    pub fn new(index: u32, relevance_score: f64) -> Self {
        Self {
            index,
            relevance_score,
        }
    }
}

/// Event payload for the end of an underlying reranking-model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RerankingModelCallEndEvent {
    /// Unique outer rerank call id.
    pub call_id: String,
    /// Operation id, normally `ai.rerank.doRerank`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Document family, usually `text` or `object`.
    pub documents_type: String,
    /// Ranking summaries from the model call.
    pub ranking: Vec<RerankingModelCallRanking>,
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

/// Passive AI SDK-style result envelope for a `generateImage` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateImageResult {
    /// First generated image.
    pub image: GeneratedFile,
    /// Generated images.
    pub images: Vec<GeneratedFile>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Response metadata from each underlying provider call.
    pub responses: Vec<ImageModelResponseMetadata>,
    /// Provider-specific metadata keyed by provider id.
    pub provider_metadata: ImageModelProviderMetadata,
    /// Combined image-model usage across all underlying provider calls.
    pub usage: ImageModelUsage,
}

impl GenerateImageResult {
    /// Create a result from a required first image and the full image list.
    pub fn new(image: GeneratedFile, mut images: Vec<GeneratedFile>) -> Self {
        if images.is_empty() {
            images.push(image.clone());
        }

        Self {
            image,
            images,
            warnings: Vec::new(),
            responses: Vec::new(),
            provider_metadata: ImageModelProviderMetadata::default(),
            usage: ImageModelUsage::default(),
        }
    }

    /// Try to create a result from the full image list.
    pub fn from_images(images: Vec<GeneratedFile>) -> Option<Self> {
        let image = images.first()?.clone();
        Some(Self::new(image, images))
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach response metadata envelopes.
    pub fn with_responses(mut self, responses: Vec<ImageModelResponseMetadata>) -> Self {
        self.responses = responses;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ImageModelProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }

    /// Attach image-model usage.
    pub fn with_usage(mut self, usage: ImageModelUsage) -> Self {
        self.usage = usage;
        self
    }
}

/// Backwards-compatible AI SDK `Experimental_GenerateImageResult` export.
#[allow(non_camel_case_types)]
pub type Experimental_GenerateImageResult = GenerateImageResult;

/// Passive AI SDK-style result envelope for an `experimental_generateVideo` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateVideoResult {
    /// First generated video.
    pub video: GeneratedFile,
    /// Generated videos.
    pub videos: Vec<GeneratedFile>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Response metadata from each underlying provider call.
    pub responses: Vec<VideoModelResponseMetadata>,
    /// Provider-specific metadata keyed by provider id.
    pub provider_metadata: VideoModelProviderMetadata,
}

impl GenerateVideoResult {
    /// Create a result from a required first video and the full video list.
    pub fn new(video: GeneratedFile, mut videos: Vec<GeneratedFile>) -> Self {
        if videos.is_empty() {
            videos.push(video.clone());
        }

        Self {
            video,
            videos,
            warnings: Vec::new(),
            responses: Vec::new(),
            provider_metadata: VideoModelProviderMetadata::default(),
        }
    }

    /// Try to create a result from the full video list.
    pub fn from_videos(videos: Vec<GeneratedFile>) -> Option<Self> {
        let video = videos.first()?.clone();
        Some(Self::new(video, videos))
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach response metadata envelopes.
    pub fn with_responses(mut self, responses: Vec<VideoModelResponseMetadata>) -> Self {
        self.responses = responses;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: VideoModelProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }
}

/// Passive AI SDK-style result envelope for a `generateSpeech` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechResult {
    /// Generated audio file.
    pub audio: GeneratedAudioFile,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Response metadata from each underlying provider call.
    pub responses: Vec<SpeechModelResponseMetadata>,
    /// Provider-specific metadata keyed by provider id.
    pub provider_metadata: ProviderMetadata,
}

impl SpeechResult {
    /// Create a speech result from generated audio.
    pub fn new(audio: GeneratedAudioFile) -> Self {
        Self {
            audio,
            warnings: Vec::new(),
            responses: Vec::new(),
            provider_metadata: ProviderMetadata::default(),
        }
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach response metadata envelopes.
    pub fn with_responses(mut self, responses: Vec<SpeechModelResponseMetadata>) -> Self {
        self.responses = responses;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }
}

/// Backwards-compatible AI SDK `Experimental_SpeechResult` export.
#[allow(non_camel_case_types)]
pub type Experimental_SpeechResult = SpeechResult;

/// Transcript segment used by AI SDK-style `TranscriptionResult`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionSegment {
    /// Segment text.
    pub text: String,
    /// Segment start time in seconds.
    pub start_second: f64,
    /// Segment end time in seconds.
    pub end_second: f64,
}

impl TranscriptionSegment {
    /// Create a transcript segment.
    pub fn new(text: impl Into<String>, start_second: f64, end_second: f64) -> Self {
        Self {
            text: text.into(),
            start_second,
            end_second,
        }
    }
}

/// Passive AI SDK-style result envelope for a `transcribe` call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionResult {
    /// Complete transcript text.
    pub text: String,
    /// Segment-level transcript timing information.
    pub segments: Vec<TranscriptionSegment>,
    /// Detected language, usually as an ISO-639-1 code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Total audio duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_in_seconds: Option<f64>,
    /// Non-fatal provider warnings.
    pub warnings: Vec<Warning>,
    /// Response metadata from each underlying provider call.
    pub responses: Vec<TranscriptionModelResponseMetadata>,
    /// Provider-specific metadata keyed by provider id.
    pub provider_metadata: ProviderMetadata,
}

impl TranscriptionResult {
    /// Create a transcription result from final text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            segments: Vec::new(),
            language: None,
            duration_in_seconds: None,
            warnings: Vec::new(),
            responses: Vec::new(),
            provider_metadata: ProviderMetadata::default(),
        }
    }

    /// Attach transcript segments.
    pub fn with_segments(mut self, segments: Vec<TranscriptionSegment>) -> Self {
        self.segments = segments;
        self
    }

    /// Attach detected language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Attach total audio duration in seconds.
    pub fn with_duration_in_seconds(mut self, duration_in_seconds: f64) -> Self {
        self.duration_in_seconds = Some(duration_in_seconds);
        self
    }

    /// Attach non-fatal provider warnings.
    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Attach response metadata envelopes.
    pub fn with_responses(mut self, responses: Vec<TranscriptionModelResponseMetadata>) -> Self {
        self.responses = responses;
        self
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }
}

/// Backwards-compatible AI SDK `Experimental_TranscriptionResult` export.
#[allow(non_camel_case_types)]
pub type Experimental_TranscriptionResult = TranscriptionResult;

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
    use super::super::{
        AssistantContent, ReasoningPart, TextPart, ToolApprovalRequest, ToolApprovalResponse,
        ToolCallPart, ToolResultPart, UiMessagePart, UiMessageRole,
    };
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
    fn generate_object_events_and_stream_parts_match_ai_sdk_shape() {
        let response_metadata = LanguageModelResponseMetadata {
            id: "resp_1".to_string(),
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "gpt-4o-mini".to_string(),
            headers: None,
        };
        let response = GenerateObjectResponseMetadata::new(response_metadata.clone())
            .with_body(serde_json::json!({ "raw": true }));
        let request = LanguageModelRequestMetadata {
            body: Some(serde_json::json!({ "messages": [] })),
        };

        let mut provider_options = ProviderOptionsMap::new();
        provider_options.insert("openai", serde_json::json!({ "strictJsonSchema": true }));

        let start = GenerateObjectStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.generateObject".to_string(),
            provider: "openai".to_string(),
            model_id: "gpt-4o-mini".to_string(),
            system: Some(SystemPrompt::Text("return JSON".to_string())),
            prompt: Some(PromptInput::Text("extract".to_string())),
            messages: None,
            max_output_tokens: Some(128),
            temperature: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: Some(7),
            max_retries: 2,
            headers: Some(HashMap::from([(
                "x-test".to_string(),
                Some("1".to_string()),
            )])),
            provider_options: Some(provider_options),
            output: GenerateObjectOutputStrategy::NoSchema,
            schema: None,
            schema_name: None,
            schema_description: None,
        };
        let start_json = serde_json::to_value(&start).expect("serialize start event");
        assert_eq!(start_json["callId"], serde_json::json!("call_1"));
        assert_eq!(start_json["output"], serde_json::json!("no-schema"));
        assert_eq!(
            start_json["providerOptions"]["openai"]["strictJsonSchema"],
            serde_json::json!(true)
        );

        let step_end = GenerateObjectStepEndEvent {
            call_id: "call_1".to_string(),
            step_number: 0,
            provider: "openai".to_string(),
            model_id: "gpt-4o-mini".to_string(),
            finish_reason: FinishReason::Stop,
            usage: LanguageModelUsage::new(5, 7),
            object_text: "{\"answer\":42}".to_string(),
            reasoning: Some("parsed as JSON".to_string()),
            warnings: None,
            request: request.clone(),
            response: response.clone(),
            provider_metadata: None,
            ms_to_first_chunk: Some(12),
        };
        let step_end_json = serde_json::to_value(&step_end).expect("serialize step end event");
        assert_eq!(
            step_end_json["objectText"],
            serde_json::json!("{\"answer\":42}")
        );
        assert_eq!(
            step_end_json["response"]["body"]["raw"],
            serde_json::json!(true)
        );
        assert_eq!(step_end_json["msToFirstChunk"], serde_json::json!(12));

        let end: GenerateObjectEndEvent = GenerateObjectEndEvent {
            call_id: "call_1".to_string(),
            object: Some(serde_json::json!({ "answer": 42 })),
            error: None,
            reasoning: Some("parsed as JSON".to_string()),
            finish_reason: FinishReason::Stop,
            usage: LanguageModelUsage::new(5, 7),
            warnings: None,
            request,
            response,
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "responseId": "resp_1" }),
            )])),
        };
        let end_json = serde_json::to_value(&end).expect("serialize end event");
        assert_eq!(end_json["object"]["answer"], serde_json::json!(42));
        assert_eq!(
            end_json["providerMetadata"]["openai"]["responseId"],
            serde_json::json!("resp_1")
        );

        let stream_parts: Vec<ObjectStreamPart> = vec![
            ObjectStreamObjectPart::new(serde_json::json!({ "answer": 4 })).into(),
            ObjectStreamTextDeltaPart::new("{\"answer\"").into(),
            ObjectStreamErrorPart::new(serde_json::json!({ "message": "transient" })).into(),
            ObjectStreamFinishPart::new(
                FinishReason::Stop,
                LanguageModelUsage::new(5, 7),
                response_metadata,
            )
            .into(),
        ];
        let stream_json = serde_json::to_value(&stream_parts).expect("serialize object parts");
        assert_eq!(stream_json[0]["type"], serde_json::json!("object"));
        assert_eq!(stream_json[0]["object"]["answer"], serde_json::json!(4));
        assert_eq!(
            stream_json[1]["textDelta"],
            serde_json::json!("{\"answer\"")
        );
        assert_eq!(
            stream_json[2]["error"]["message"],
            serde_json::json!("transient")
        );
        assert_eq!(stream_json[3]["finishReason"], serde_json::json!("stop"));

        let roundtrip: Vec<ObjectStreamPart> =
            serde_json::from_value(stream_json).expect("deserialize object parts");
        assert_eq!(roundtrip[0].r#type(), "object");
        assert_eq!(roundtrip[1].r#type(), "text-delta");
        assert_eq!(roundtrip[2].r#type(), "error");
        assert_eq!(roundtrip[3].r#type(), "finish");
    }

    #[test]
    fn ui_message_chunks_match_ai_sdk_stream_shape() {
        assert_eq!(
            UI_MESSAGE_STREAM_HEADERS
                .iter()
                .find(|(key, _)| *key == "x-vercel-ai-ui-message-stream")
                .map(|(_, value)| *value),
            Some("v1")
        );

        let create_message = CreateUIMessage::new()
            .with_id("msg_new")
            .with_role(UiMessageRole::User)
            .with_metadata(serde_json::json!({ "draft": true }))
            .with_parts(vec![UiMessagePart::text("hello")]);
        let create_message_json =
            serde_json::to_value(&create_message).expect("serialize create UI message");
        assert_eq!(create_message_json["id"], serde_json::json!("msg_new"));
        assert_eq!(create_message_json["role"], serde_json::json!("user"));
        assert_eq!(
            create_message_json["metadata"]["draft"],
            serde_json::json!(true)
        );
        assert_eq!(
            create_message_json["parts"][0]["type"],
            serde_json::json!("text")
        );

        let chat_request_options = ChatRequestOptions::new()
            .with_headers(HashMap::from([(
                "x-trace-id".to_string(),
                "trace_1".to_string(),
            )]))
            .with_body(serde_json::json!({ "sessionId": "sess_1" }))
            .with_metadata(serde_json::json!({ "source": "ui" }));
        let chat_request_json =
            serde_json::to_value(&chat_request_options).expect("serialize chat request options");
        assert_eq!(
            chat_request_json["headers"]["x-trace-id"],
            serde_json::json!("trace_1")
        );
        assert_eq!(
            chat_request_json["body"]["sessionId"],
            serde_json::json!("sess_1")
        );
        assert_eq!(
            chat_request_json["metadata"]["source"],
            serde_json::json!("ui")
        );
        assert_eq!(
            serde_json::to_value(ChatStatus::Streaming).expect("serialize chat status"),
            serde_json::json!("streaming")
        );

        let chat_state = ChatState::ready(vec![UiMessage::user(
            "msg_user",
            vec![UiMessagePart::text("hello")],
        )])
        .with_error(serde_json::json!({ "message": "network" }));
        let chat_state_json = serde_json::to_value(&chat_state).expect("serialize chat state");
        assert_eq!(chat_state_json["status"], serde_json::json!("error"));
        assert_eq!(
            chat_state_json["error"]["message"],
            serde_json::json!("network")
        );
        assert!(chat_state_json.get("pushMessage").is_none());

        let chat_init =
            ChatInit::new()
                .with_id("chat_1")
                .with_messages(vec![UiMessage::assistant(
                    "msg_assistant",
                    vec![UiMessagePart::text("ready")],
                )]);
        let chat_init_json = serde_json::to_value(&chat_init).expect("serialize chat init");
        assert_eq!(chat_init_json["id"], serde_json::json!("chat_1"));
        assert_eq!(
            chat_init_json["messages"][0]["role"],
            serde_json::json!("assistant")
        );
        assert!(chat_init_json.get("transport").is_none());
        assert!(chat_init_json.get("onFinish").is_none());

        let send_options = ChatTransportSendMessagesOptions {
            trigger: ChatTransportTrigger::SubmitMessage,
            chat_id: "chat_1".to_string(),
            message_id: None,
            messages: vec![UiMessage::user(
                "msg_user",
                vec![UiMessagePart::text("hello")],
            )],
            request_options: chat_request_options.clone(),
        };
        let send_options_json =
            serde_json::to_value(&send_options).expect("serialize send messages options");
        assert_eq!(
            send_options_json["trigger"],
            serde_json::json!("submit-message")
        );
        assert_eq!(send_options_json["chatId"], serde_json::json!("chat_1"));
        assert_eq!(
            send_options_json["headers"]["x-trace-id"],
            serde_json::json!("trace_1")
        );

        let http_transport_options = HttpChatTransportInitOptions::new()
            .with_api("/api/chat")
            .with_credentials(RequestCredentials::Include);
        let http_transport_json = serde_json::to_value(&http_transport_options)
            .expect("serialize HTTP chat transport options");
        assert_eq!(http_transport_json["api"], serde_json::json!("/api/chat"));
        assert_eq!(
            http_transport_json["credentials"],
            serde_json::json!("include")
        );
        assert!(http_transport_json.get("fetch").is_none());
        assert!(
            http_transport_json
                .get("prepareSendMessagesRequest")
                .is_none()
        );

        let prepare_send = PrepareSendMessagesRequestOptions {
            id: "chat_1".to_string(),
            messages: vec![UiMessage::user(
                "msg_user",
                vec![UiMessagePart::text("hello")],
            )],
            request_metadata: Some(serde_json::json!({ "source": "ui" })),
            body: Some(serde_json::json!({ "sessionId": "sess_1" })),
            credentials: Some(RequestCredentials::SameOrigin),
            headers: Some(HashMap::from([(
                "x-trace-id".to_string(),
                "trace_1".to_string(),
            )])),
            api: "/api/chat".to_string(),
            trigger: ChatTransportTrigger::RegenerateMessage,
            message_id: Some("msg_assistant".to_string()),
        };
        let prepare_send_json =
            serde_json::to_value(&prepare_send).expect("serialize prepare send options");
        assert_eq!(
            prepare_send_json["requestMetadata"]["source"],
            serde_json::json!("ui")
        );
        assert_eq!(
            prepare_send_json["trigger"],
            serde_json::json!("regenerate-message")
        );

        let prepared_send = PreparedSendMessagesRequest {
            body: serde_json::json!({ "id": "chat_1" }),
            headers: Some(HashMap::from([("x-prepared".to_string(), "1".to_string())])),
            credentials: Some(RequestCredentials::Omit),
            api: Some("/api/custom-chat".to_string()),
        };
        let prepared_send_json =
            serde_json::to_value(&prepared_send).expect("serialize prepared send request");
        assert_eq!(prepared_send_json["credentials"], serde_json::json!("omit"));
        assert_eq!(
            prepared_send_json["api"],
            serde_json::json!("/api/custom-chat")
        );

        let reconnect_options = ChatTransportReconnectToStreamOptions {
            chat_id: "chat_1".to_string(),
            request_options: ChatRequestOptions::new().with_metadata(serde_json::json!({
                "resume": true
            })),
        };
        let reconnect_json =
            serde_json::to_value(&reconnect_options).expect("serialize reconnect options");
        assert_eq!(
            reconnect_json["metadata"]["resume"],
            serde_json::json!(true)
        );

        let prepare_reconnect = PrepareReconnectToStreamRequestOptions {
            id: "chat_1".to_string(),
            request_metadata: Some(serde_json::json!({ "resume": true })),
            body: None,
            credentials: Some(RequestCredentials::SameOrigin),
            headers: None,
            api: "/api/chat".to_string(),
        };
        let prepare_reconnect_json =
            serde_json::to_value(&prepare_reconnect).expect("serialize prepare reconnect options");
        assert_eq!(
            prepare_reconnect_json["credentials"],
            serde_json::json!("same-origin")
        );
        assert_eq!(
            serde_json::to_value(PreparedReconnectToStreamRequest {
                api: Some("/api/chat/chat_1/stream".to_string()),
                ..PreparedReconnectToStreamRequest::default()
            })
            .expect("serialize prepared reconnect request")["api"],
            serde_json::json!("/api/chat/chat_1/stream")
        );

        let completion_request_options = CompletionRequestOptions::new()
            .with_headers(HashMap::from([("x-mode".to_string(), "test".to_string())]))
            .with_body(serde_json::json!({ "tenant": "acme" }));
        let completion_request_json = serde_json::to_value(&completion_request_options)
            .expect("serialize completion request options");
        assert_eq!(
            completion_request_json["headers"]["x-mode"],
            serde_json::json!("test")
        );
        assert_eq!(
            completion_request_json["body"]["tenant"],
            serde_json::json!("acme")
        );

        let use_completion_options = UseCompletionOptions::new()
            .with_api("/api/completion")
            .with_id("completion_1")
            .with_initial_input("question")
            .with_initial_completion("answer")
            .with_credentials(RequestCredentials::SameOrigin)
            .with_headers(HashMap::from([("x-ui".to_string(), "1".to_string())]))
            .with_body(serde_json::json!({ "sessionId": "sess_1" }))
            .with_stream_protocol(CompletionStreamProtocol::Text);
        let use_completion_json = serde_json::to_value(&use_completion_options)
            .expect("serialize use completion options");
        assert_eq!(
            use_completion_json["initialInput"],
            serde_json::json!("question")
        );
        assert_eq!(
            use_completion_json["initialCompletion"],
            serde_json::json!("answer")
        );
        assert_eq!(
            use_completion_json["credentials"],
            serde_json::json!("same-origin")
        );
        assert_eq!(
            use_completion_json["streamProtocol"],
            serde_json::json!("text")
        );
        assert!(use_completion_json.get("onFinish").is_none());
        assert!(use_completion_json.get("onError").is_none());
        assert!(use_completion_json.get("fetch").is_none());

        let options = UiMessageStreamOptions::new()
            .with_original_messages(vec![UiMessage::user(
                "msg_user",
                vec![UiMessagePart::text("hello")],
            )])
            .with_send_reasoning(false)
            .with_send_sources(true)
            .with_send_finish(false)
            .with_send_start(true);
        let options_json =
            serde_json::to_value(&options).expect("serialize UI message stream options");
        assert_eq!(
            options_json["originalMessages"][0]["id"],
            serde_json::json!("msg_user")
        );
        assert_eq!(options_json["sendReasoning"], serde_json::json!(false));
        assert_eq!(options_json["sendSources"], serde_json::json!(true));
        assert_eq!(options_json["sendFinish"], serde_json::json!(false));
        assert_eq!(options_json["sendStart"], serde_json::json!(true));
        assert!(options_json.get("generateMessageId").is_none());
        assert!(options_json.get("onFinish").is_none());
        assert!(options_json.get("messageMetadata").is_none());
        assert!(options_json.get("onError").is_none());

        let options_roundtrip: UIMessageStreamOptions =
            serde_json::from_value(options_json).expect("deserialize UI message stream options");
        assert_eq!(
            options_roundtrip
                .original_messages
                .as_ref()
                .expect("original messages")[0]
                .id,
            "msg_user"
        );
        let empty_options_json = serde_json::to_value(UiMessageStreamOptions::<UiMessage>::new())
            .expect("serialize empty UI message stream options");
        assert_eq!(empty_options_json, serde_json::json!({}));

        let mut start = UiMessageStartChunk::new();
        start.message_id = Some("msg_1".to_string());
        start.message_metadata = Some(serde_json::json!({ "turn": 1 }));

        let mut finish = UiMessageFinishChunk::new();
        finish.finish_reason = Some(FinishReason::Stop);
        finish.message_metadata = Some(serde_json::json!({ "done": true }));

        let mut data = UiMessageDataChunk::new("weather", serde_json::json!({ "city": "Paris" }));
        data.id = Some("data_1".to_string());
        data.transient = Some(true);

        let chunks: Vec<UiMessageChunk> = vec![
            UiMessageChunk::Start(start),
            UiMessageTextStartChunk::new("text_1").into(),
            UiMessageTextDeltaChunk::new("text_1", "hello").into(),
            UiMessageChunk::ToolInputAvailable(UiMessageToolInputAvailableChunk::new(
                "call_1",
                "weather",
                serde_json::json!({ "city": "Paris" }),
            )),
            UiMessageChunk::Data(data),
            UiMessageChunk::SourceDocument(UiMessageSourceDocumentChunk::new(
                "src_1",
                "text/plain",
                "Notes",
            )),
            UiMessageChunk::Finish(finish),
        ];

        let json = serde_json::to_value(&chunks).expect("serialize UI message chunks");
        assert_eq!(json[0]["type"], serde_json::json!("start"));
        assert_eq!(json[0]["messageId"], serde_json::json!("msg_1"));
        assert_eq!(json[2]["type"], serde_json::json!("text-delta"));
        assert_eq!(json[2]["delta"], serde_json::json!("hello"));
        assert_eq!(json[3]["toolCallId"], serde_json::json!("call_1"));
        assert_eq!(json[4]["type"], serde_json::json!("data-weather"));
        assert_eq!(json[4]["transient"], serde_json::json!(true));
        assert_eq!(json[5]["sourceId"], serde_json::json!("src_1"));
        assert_eq!(json[6]["finishReason"], serde_json::json!("stop"));

        let roundtrip: Vec<UiMessageChunk> =
            serde_json::from_value(json).expect("deserialize UI message chunks");
        assert_eq!(roundtrip[0].r#type(), "start");
        assert_eq!(roundtrip[2].r#type(), "text-delta");
        assert_eq!(roundtrip[4].r#type(), "data-weather");
        assert_eq!(roundtrip[6].r#type(), "finish");

        let invalid_data_chunk = serde_json::json!({
            "type": "invalid-data",
            "data": { "city": "Paris" }
        });
        assert!(serde_json::from_value::<UiMessageDataChunk>(invalid_data_chunk.clone()).is_err());
        assert!(serde_json::from_value::<UiMessageChunk>(invalid_data_chunk).is_err());

        let missing_type_chunk = serde_json::json!({ "messageId": "msg_1" });
        assert!(serde_json::from_value::<UiMessageStartChunk>(missing_type_chunk.clone()).is_err());
        assert!(serde_json::from_value::<UiMessageChunk>(missing_type_chunk).is_err());
        assert!(serde_json::from_value::<UiMessageChunk>(serde_json::json!({})).is_err());
    }

    #[test]
    fn image_speech_and_transcription_results_match_ai_sdk_shape() {
        let image_file = GeneratedFile::from_bytes(b"image", "image/png");
        let image_response = ImageModelResponseMetadata {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "image-model".to_string(),
            headers: None,
        };
        let image_result = GenerateImageResult::new(image_file.clone(), vec![image_file])
            .with_responses(vec![image_response])
            .with_provider_metadata(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "images": [{ "id": "img_1" }] }),
            )]))
            .with_usage(ImageModelUsage::new(Some(1), Some(2), Some(3)));
        let image_json = serde_json::to_value(&image_result).expect("serialize image result");

        assert_eq!(
            image_json["image"]["mediaType"],
            serde_json::json!("image/png")
        );
        assert_eq!(
            image_json["images"][0]["base64"],
            serde_json::json!("aW1hZ2U=")
        );
        assert_eq!(image_json["usage"]["totalTokens"], serde_json::json!(3));
        assert_eq!(
            image_json["providerMetadata"]["openai"]["images"][0]["id"],
            serde_json::json!("img_1")
        );
        let _: Experimental_GenerateImageResult =
            serde_json::from_value(image_json).expect("deserialize image result");

        let video_file = GeneratedFile::from_bytes(b"video", "video/mp4");
        let video_response = VideoModelResponseMetadata {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:30Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "video-model".to_string(),
            headers: None,
            provider_metadata: Some(HashMap::from([(
                "xai".to_string(),
                serde_json::json!({ "requestId": "vid_1" }),
            )])),
        };
        let video_result = GenerateVideoResult::new(video_file.clone(), vec![video_file])
            .with_responses(vec![video_response])
            .with_provider_metadata(HashMap::from([(
                "xai".to_string(),
                serde_json::json!({ "videos": [{ "id": "vid_1" }] }),
            )]));
        let video_json = serde_json::to_value(&video_result).expect("serialize video result");

        assert_eq!(
            video_json["video"]["mediaType"],
            serde_json::json!("video/mp4")
        );
        assert_eq!(
            video_json["videos"][0]["base64"],
            serde_json::json!("dmlkZW8=")
        );
        assert_eq!(
            video_json["responses"][0]["providerMetadata"]["xai"]["requestId"],
            serde_json::json!("vid_1")
        );
        assert_eq!(
            video_json["providerMetadata"]["xai"]["videos"][0]["id"],
            serde_json::json!("vid_1")
        );
        let _: GenerateVideoResult =
            serde_json::from_value(video_json).expect("deserialize video result");

        let audio = GeneratedAudioFile::from_bytes(b"audio", "audio/mpeg");
        assert_eq!(audio.format, "mp3");
        assert_eq!(audio.base64(), "YXVkaW8=");
        let speech_response = SpeechModelResponseMetadata {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:01:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "speech-model".to_string(),
            headers: None,
            body: Some(serde_json::json!({ "ok": true })),
        };
        let speech_result = SpeechResult::new(audio)
            .with_responses(vec![speech_response])
            .with_provider_metadata(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "voice": "alloy" }),
            )]));
        let speech_json = serde_json::to_value(&speech_result).expect("serialize speech result");

        assert_eq!(speech_json["audio"]["format"], serde_json::json!("mp3"));
        assert_eq!(
            speech_json["audio"]["mediaType"],
            serde_json::json!("audio/mpeg")
        );
        assert_eq!(
            speech_json["providerMetadata"]["openai"]["voice"],
            serde_json::json!("alloy")
        );
        let _: Experimental_SpeechResult =
            serde_json::from_value(speech_json).expect("deserialize speech result");

        let transcription_response = TranscriptionModelResponseMetadata {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:02:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: "stt-model".to_string(),
            headers: None,
        };
        let transcription_result = TranscriptionResult::new("hello")
            .with_segments(vec![TranscriptionSegment::new("hello", 0.0, 0.5)])
            .with_language("en")
            .with_duration_in_seconds(0.5)
            .with_responses(vec![transcription_response])
            .with_provider_metadata(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "transcriptId": "tr_1" }),
            )]));
        let transcription_json =
            serde_json::to_value(&transcription_result).expect("serialize transcription result");

        assert_eq!(transcription_json["text"], serde_json::json!("hello"));
        assert_eq!(
            transcription_json["segments"][0]["startSecond"],
            serde_json::json!(0.0)
        );
        assert_eq!(
            transcription_json["durationInSeconds"],
            serde_json::json!(0.5)
        );
        assert_eq!(
            transcription_json["providerMetadata"]["openai"]["transcriptId"],
            serde_json::json!("tr_1")
        );
        let _: Experimental_TranscriptionResult =
            serde_json::from_value(transcription_json).expect("deserialize transcription result");
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
    fn embedding_result_and_event_payloads_match_ai_sdk_shape() {
        let response_data = ModelCallResponseData::new()
            .with_headers(HashMap::from([(
                "x-request-id".to_string(),
                "req_1".to_string(),
            )]))
            .with_body(serde_json::json!({ "ok": true }));
        let provider_metadata = HashMap::from([(
            "openai".to_string(),
            serde_json::json!({ "embeddingId": "emb_1" }),
        )]);

        let result = EmbedResult::new("hello", vec![1.0, 2.0], EmbeddingModelUsage::new(2))
            .with_provider_metadata(provider_metadata.clone())
            .with_response(response_data.clone());
        let result_json = serde_json::to_value(&result).expect("serialize embed result");

        assert_eq!(result_json["value"], serde_json::json!("hello"));
        assert_eq!(result_json["embedding"][0], serde_json::json!(1.0));
        assert_eq!(result_json["usage"]["tokens"], serde_json::json!(2));
        assert_eq!(
            result_json["providerMetadata"]["openai"]["embeddingId"],
            serde_json::json!("emb_1")
        );
        assert_eq!(
            result_json["response"]["headers"]["x-request-id"],
            serde_json::json!("req_1")
        );
        let _: EmbedResult = serde_json::from_value(result_json).expect("deserialize embed result");

        let many = EmbedManyResult::new(
            vec!["a".to_string(), "b".to_string()],
            vec![vec![1.0], vec![2.0]],
            EmbeddingModelUsage::new(4),
        )
        .with_provider_metadata(provider_metadata.clone())
        .with_responses(vec![Some(response_data.clone()), None]);
        let many_json = serde_json::to_value(&many).expect("serialize embedMany result");

        assert_eq!(many_json["values"][1], serde_json::json!("b"));
        assert_eq!(many_json["embeddings"][1][0], serde_json::json!(2.0));
        assert!(many_json["responses"][1].is_null());
        let _: EmbedManyResult =
            serde_json::from_value(many_json).expect("deserialize embedMany result");

        let start = EmbedStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.embedMany".to_string(),
            provider: "openai".to_string(),
            model_id: "text-embedding-3-small".to_string(),
            value: EmbedValue::from(vec!["a".to_string(), "b".to_string()]),
            max_retries: 2,
            headers: Some(HashMap::from([(
                "x-test".to_string(),
                Some("1".to_string()),
            )])),
            provider_options: None,
        };
        let start_json = serde_json::to_value(&start).expect("serialize embed start event");
        assert_eq!(start_json["callId"], serde_json::json!("call_1"));
        assert_eq!(start_json["value"][0], serde_json::json!("a"));

        let end = EmbedEndEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.embedMany".to_string(),
            provider: "openai".to_string(),
            model_id: "text-embedding-3-small".to_string(),
            value: EmbedValue::from(vec!["a".to_string(), "b".to_string()]),
            embedding: EmbedOutput::from(vec![vec![1.0], vec![2.0]]),
            usage: EmbeddingModelUsage::new(4),
            warnings: Vec::new(),
            provider_metadata: Some(provider_metadata),
            response: Some(EmbedResponseData::from(vec![Some(response_data), None])),
        };
        let end_json = serde_json::to_value(&end).expect("serialize embed end event");
        assert_eq!(end_json["embedding"][1][0], serde_json::json!(2.0));
        assert!(end_json["response"][1].is_null());

        let model_call_start = EmbeddingModelCallStartEvent {
            call_id: "call_1".to_string(),
            embed_call_id: "embed_1".to_string(),
            operation_id: "ai.embedMany.doEmbed".to_string(),
            provider: "openai".to_string(),
            model_id: "text-embedding-3-small".to_string(),
            values: vec!["a".to_string()],
        };
        let model_call_end = EmbeddingModelCallEndEvent {
            call_id: "call_1".to_string(),
            embed_call_id: "embed_1".to_string(),
            operation_id: "ai.embedMany.doEmbed".to_string(),
            provider: "openai".to_string(),
            model_id: "text-embedding-3-small".to_string(),
            values: vec!["a".to_string()],
            embeddings: vec![vec![1.0]],
            usage: EmbeddingModelUsage::new(1),
        };
        let model_start_json =
            serde_json::to_value(&model_call_start).expect("serialize embedding model start");
        let model_end_json =
            serde_json::to_value(&model_call_end).expect("serialize embedding model end");
        assert_eq!(
            model_start_json["embedCallId"],
            serde_json::json!("embed_1")
        );
        assert_eq!(model_end_json["embeddings"][0][0], serde_json::json!(1.0));
    }

    #[test]
    fn rerank_result_and_event_payloads_match_ai_sdk_shape() {
        let timestamp = DateTime::parse_from_rfc3339("2026-04-21T09:03:00Z")
            .expect("valid timestamp")
            .with_timezone(&Utc);
        let response = RerankResponseMetadata::new(timestamp, "rerank-model")
            .with_id("rr_1")
            .with_headers(HashMap::from([(
                "x-request-id".to_string(),
                "req_1".to_string(),
            )]))
            .with_body(serde_json::json!({ "ok": true }));
        let original_documents = vec!["apple".to_string(), "banana".to_string()];
        let ranking = vec![
            RerankRanking::new(1, 0.9, "banana".to_string()),
            RerankRanking::new(0, 0.7, "apple".to_string()),
        ];
        let result = RerankResult::new(
            original_documents.clone(),
            ranking.clone(),
            response.clone(),
        )
        .with_provider_metadata(HashMap::from([(
            "cohere".to_string(),
            serde_json::json!({ "searchUnits": 1 }),
        )]));
        let result_json = serde_json::to_value(&result).expect("serialize rerank result");

        assert_eq!(
            result_json["originalDocuments"][0],
            serde_json::json!("apple")
        );
        assert_eq!(
            result_json["rerankedDocuments"][0],
            serde_json::json!("banana")
        );
        assert_eq!(
            result_json["ranking"][0]["originalIndex"],
            serde_json::json!(1)
        );
        assert_eq!(result_json["ranking"][0]["score"], serde_json::json!(0.9));
        assert_eq!(result_json["response"]["id"], serde_json::json!("rr_1"));
        assert_eq!(
            result_json["providerMetadata"]["cohere"]["searchUnits"],
            serde_json::json!(1)
        );
        let _: RerankResult<String> =
            serde_json::from_value(result_json).expect("deserialize rerank result");

        let documents = vec![serde_json::json!("apple"), serde_json::json!("banana")];
        let start = RerankStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.rerank".to_string(),
            provider: "cohere".to_string(),
            model_id: "rerank-model".to_string(),
            documents: documents.clone(),
            query: "fruit".to_string(),
            top_n: Some(2),
            max_retries: 2,
            headers: None,
            provider_options: None,
        };
        let end = RerankEndEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.rerank".to_string(),
            provider: "cohere".to_string(),
            model_id: "rerank-model".to_string(),
            documents: documents.clone(),
            query: "fruit".to_string(),
            ranking: vec![
                RerankRanking::new(1, 0.9, serde_json::json!("banana")),
                RerankRanking::new(0, 0.7, serde_json::json!("apple")),
            ],
            warnings: Vec::new(),
            provider_metadata: None,
            response: response.clone(),
        };
        let start_json = serde_json::to_value(&start).expect("serialize rerank start event");
        let end_json = serde_json::to_value(&end).expect("serialize rerank end event");
        assert_eq!(start_json["documents"][0], serde_json::json!("apple"));
        assert_eq!(start_json["topN"], serde_json::json!(2));
        assert_eq!(
            end_json["ranking"][0]["document"],
            serde_json::json!("banana")
        );

        let model_call_start = RerankingModelCallStartEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.rerank.doRerank".to_string(),
            provider: "cohere".to_string(),
            model_id: "rerank-model".to_string(),
            documents,
            documents_type: "text".to_string(),
            query: "fruit".to_string(),
            top_n: Some(2),
        };
        let model_call_end = RerankingModelCallEndEvent {
            call_id: "call_1".to_string(),
            operation_id: "ai.rerank.doRerank".to_string(),
            provider: "cohere".to_string(),
            model_id: "rerank-model".to_string(),
            documents_type: "text".to_string(),
            ranking: vec![RerankingModelCallRanking::new(1, 0.9)],
        };
        let model_start_json =
            serde_json::to_value(&model_call_start).expect("serialize rerank model start");
        let model_end_json =
            serde_json::to_value(&model_call_end).expect("serialize rerank model end");
        assert_eq!(model_start_json["documentsType"], serde_json::json!("text"));
        assert_eq!(
            model_end_json["ranking"][0]["relevanceScore"],
            serde_json::json!(0.9)
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
    fn ai_sdk_error_index_passive_shapes_match_exported_errors() {
        let base_error = AISDKError::new("AI_TestError", "boom", Some(serde_json::json!("cause")));
        assert_eq!(
            serde_json::to_value(&base_error).expect("serialize base error"),
            serde_json::json!({
                "name": "AI_TestError",
                "message": "boom",
                "cause": "cause"
            })
        );

        let api_error = APICallError::new(
            "rate limited",
            "https://api.example.test",
            serde_json::json!({ "model": "test" }),
            Some(429),
        );
        let api_error_json = serde_json::to_value(&api_error).expect("serialize api error");
        assert_eq!(api_error_json["statusCode"], serde_json::json!(429));
        assert_eq!(api_error_json["isRetryable"], serde_json::json!(true));

        assert_eq!(
            serde_json::to_value(EmptyResponseBodyError::new())
                .expect("serialize empty body error"),
            serde_json::json!({ "message": "Empty response body" })
        );

        assert_eq!(
            serde_json::to_value(InvalidPromptError::new(
                serde_json::json!({ "role": "bad" }),
                "unsupported role",
                None,
            ))
            .expect("serialize invalid prompt")["message"],
            serde_json::json!("Invalid prompt: unsupported role")
        );

        assert_eq!(
            serde_json::to_value(InvalidResponseDataError::new(
                serde_json::json!({ "ok": false }),
            ))
            .expect("serialize invalid response data")["message"],
            serde_json::json!(r#"Invalid response data: {"ok":false}."#)
        );

        assert_eq!(
            serde_json::to_value(JSONParseError::new(
                "not json",
                Some(serde_json::json!("SyntaxError")),
            ))
            .expect("serialize json parse error")["message"],
            serde_json::json!("JSON parsing failed: Text: not json.\nError message: SyntaxError")
        );

        assert_eq!(
            serde_json::to_value(LoadAPIKeyError::new("missing key"))
                .expect("serialize load api key error")["message"],
            serde_json::json!("missing key")
        );
        assert_eq!(
            serde_json::to_value(LoadSettingError::new("missing setting"))
                .expect("serialize load setting error")["message"],
            serde_json::json!("missing setting")
        );
        assert_eq!(
            serde_json::to_value(NoContentGeneratedError::new())
                .expect("serialize no content error")["message"],
            serde_json::json!("No content generated.")
        );

        let no_such_model = NoSuchModelError::new("gpt-test", NoSuchModelType::LanguageModel);
        let no_such_model_json =
            serde_json::to_value(&no_such_model).expect("serialize no such model error");
        assert_eq!(
            no_such_model_json["modelType"],
            serde_json::json!("languageModel")
        );
        assert_eq!(
            no_such_model_json["message"],
            serde_json::json!("No such languageModel: gpt-test")
        );

        let no_provider_reference = NoSuchProviderReferenceError::new(
            "anthropic",
            HashMap::from([("openai".to_string(), "file_1".to_string())]),
        );
        assert_eq!(
            serde_json::to_value(&no_provider_reference)
                .expect("serialize no provider reference error")["message"],
            serde_json::json!(
                "No provider reference found for provider 'anthropic'. Available providers: openai"
            )
        );

        let too_many_embeddings = TooManyEmbeddingValuesForCallError::new(
            "openai",
            "text-embedding-test",
            1,
            vec![serde_json::json!("a"), serde_json::json!("b")],
        );
        assert_eq!(
            serde_json::to_value(&too_many_embeddings)
                .expect("serialize too many embeddings error")["message"],
            serde_json::json!(
                "Too many values for a single embedding call. The openai model \"text-embedding-test\" can only embed up to 1 values per call, but 2 values were provided."
            )
        );

        let type_validation = TypeValidationError::new(
            serde_json::json!({ "x": 1 }),
            Some(serde_json::json!("bad type")),
            Some(TypeValidationContext {
                field: Some("message.parts[0]".to_string()),
                entity_name: Some("tool".to_string()),
                entity_id: Some("call_1".to_string()),
            }),
        );
        assert_eq!(
            serde_json::to_value(&type_validation).expect("serialize type validation error")["message"],
            serde_json::json!(
                "Type validation failed for message.parts[0] (tool, id: \"call_1\"): Value: {\"x\":1}.\nError message: bad type"
            )
        );

        assert_eq!(
            serde_json::to_value(UnsupportedFunctionalityError::new("vision"))
                .expect("serialize unsupported functionality")["message"],
            serde_json::json!("'vision' functionality not supported.")
        );

        let invalid_argument =
            InvalidArgumentError::new("temperature", serde_json::json!(2), "must be <= 1");
        assert_eq!(
            serde_json::to_value(&invalid_argument).expect("serialize invalid argument"),
            serde_json::json!({
                "message": "Invalid argument for parameter temperature: must be <= 1",
                "parameter": "temperature",
                "value": 2
            })
        );

        let invalid_stream_part = InvalidStreamPartError::new(
            LanguageModelStreamPart::<String, JSONValue, ToolResultOutput>::TextDelta(
                TextStreamTextDeltaPart::new("txt_1", "hello"),
            ),
            "Unexpected chunk",
        );
        let invalid_stream_part_json =
            serde_json::to_value(&invalid_stream_part).expect("serialize invalid stream part");
        assert_eq!(
            invalid_stream_part_json["chunk"]["type"],
            serde_json::json!("text-delta")
        );
        assert_eq!(
            invalid_stream_part_json["message"],
            serde_json::json!("Unexpected chunk")
        );

        assert_eq!(
            serde_json::to_value(InvalidToolApprovalError::new("approval_1"))
                .expect("serialize invalid tool approval")["message"],
            serde_json::json!(
                "Tool approval response references unknown approvalId: \"approval_1\". \
                 No matching tool-approval-request found in message history."
            )
        );

        assert_eq!(
            serde_json::to_value(ToolCallNotFoundForApprovalError::new(
                "call_1",
                "approval_1"
            ))
            .expect("serialize missing approval tool call")["message"],
            serde_json::json!(
                "Tool call \"call_1\" not found for approval request \"approval_1\"."
            )
        );

        let unsupported = UnsupportedModelVersionError::new("v1", "test", "model");
        assert_eq!(
            serde_json::to_value(&unsupported).expect("serialize unsupported model version"),
            serde_json::json!({
                "message": "Unsupported model version v1 for provider \"test\" and model \"model\". AI SDK 5 only supports models that implement specification version \"v2\".",
                "version": "v1",
                "provider": "test",
                "modelId": "model"
            })
        );

        let ui_stream_error = UIMessageStreamError::new("text-delta", "txt_1", "Missing start");
        assert_eq!(
            serde_json::to_value(&ui_stream_error).expect("serialize UI stream error"),
            serde_json::json!({
                "message": "Missing start",
                "chunkType": "text-delta",
                "chunkId": "txt_1"
            })
        );

        let invalid_role = InvalidMessageRoleError::new("developer");
        assert_eq!(
            serde_json::to_value(&invalid_role).expect("serialize invalid role")["message"],
            serde_json::json!(
                "Invalid message role: 'developer'. Must be one of: \"system\", \"user\", \"assistant\", \"tool\"."
            )
        );

        let message_conversion = MessageConversionError::new(
            UiMessageWithoutId::new(UiMessageRole::Assistant, vec![UiMessagePart::text("hello")]),
            "Cannot convert assistant message",
        );
        let message_conversion_json =
            serde_json::to_value(&message_conversion).expect("serialize message conversion error");
        assert_eq!(
            message_conversion_json["originalMessage"]["role"],
            serde_json::json!("assistant")
        );
        assert_eq!(
            message_conversion_json["originalMessage"]["parts"][0]["type"],
            serde_json::json!("text")
        );

        let retry = RetryError::new(
            "Retries exhausted",
            RetryErrorReason::MaxRetriesExceeded,
            vec![
                serde_json::json!("first"),
                serde_json::json!({ "message": "last" }),
            ],
        );
        let retry_json = serde_json::to_value(&retry).expect("serialize retry error");
        assert_eq!(
            retry_json["reason"],
            serde_json::json!("maxRetriesExceeded")
        );
        assert_eq!(
            retry_json["lastError"]["message"],
            serde_json::json!("last")
        );

        assert_eq!(
            serde_json::to_value(NoImageGeneratedError::new(None, None))
                .expect("serialize no image error"),
            serde_json::json!({ "message": "No image generated." })
        );
        assert_eq!(
            serde_json::to_value(NoObjectGeneratedError::new(
                Some("{}".to_string()),
                None,
                Some(LanguageModelUsage::new(1, 2)),
                Some(FinishReason::Stop),
                None,
            ))
            .expect("serialize no object error")["finishReason"],
            serde_json::json!("stop")
        );
        assert_eq!(
            serde_json::to_value(NoOutputGeneratedError::new(Some(serde_json::json!(
                "empty response"
            ),)))
            .expect("serialize no output error")["cause"],
            serde_json::json!("empty response")
        );
        assert_eq!(
            serde_json::to_value(NoSpeechGeneratedError::new(Vec::new()))
                .expect("serialize no speech error")["message"],
            serde_json::json!("No speech audio generated.")
        );
        assert_eq!(
            serde_json::to_value(NoTranscriptGeneratedError::new(Vec::new()))
                .expect("serialize no transcript error")["message"],
            serde_json::json!("No transcript generated.")
        );
        assert_eq!(
            serde_json::to_value(NoVideoGeneratedError::new(Vec::new(), None))
                .expect("serialize no video error")["message"],
            serde_json::json!("No video generated.")
        );
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
    fn language_model_stream_parts_match_ai_sdk_model_call_shape() {
        let timestamp = DateTime::parse_from_rfc3339("2026-04-24T00:00:00Z")
            .expect("timestamp")
            .with_timezone(&Utc);
        let usage = LanguageModelUsage {
            input_tokens: Some(7),
            output_tokens: Some(5),
            total_tokens: Some(12),
            ..LanguageModelUsage::default()
        };
        let provider_metadata = ProviderMetadata::from([(
            "openai".to_string(),
            serde_json::json!({ "responseId": "resp_1" }),
        )]);
        let mut end = LanguageModelStreamModelCallEndPart::new(FinishReason::Stop, usage);
        end.raw_finish_reason = Some("stop".to_string());
        end.provider_metadata = Some(provider_metadata);

        let parts: Vec<LanguageModelStreamPart> = vec![
            LanguageModelStreamModelCallStartPart::new(Vec::new()).into(),
            LanguageModelStreamModelCallResponseMetadataPart::new()
                .with_id("resp_1")
                .with_timestamp(timestamp)
                .with_model_id("gpt-test")
                .into(),
            TextStreamTextStartPart::new("text_1").into(),
            TextStreamTextDeltaPart::new("text_1", "hello").into(),
            TextStreamReasoningDeltaPart::new("reasoning_1", "because").into(),
            TextStreamToolInputStartPart::new("call_1", "search").into(),
            ToolCall::new(
                "call_1",
                "search".to_string(),
                serde_json::json!({ "query": "rust" }),
            )
            .into(),
            ToolResult::new(
                "call_1",
                "search".to_string(),
                serde_json::json!({ "query": "rust" }),
                ToolResultOutput::json(serde_json::json!({ "answer": "ok" })),
            )
            .into(),
            TextStreamErrorPart::new(serde_json::json!({ "message": "recoverable" })).into(),
            TextStreamRawPart::new(serde_json::json!({ "chunk": 1 })).into(),
            end.into(),
        ];

        let json = serde_json::to_value(&parts).expect("serialize language model stream parts");
        assert_eq!(json[0]["type"], serde_json::json!("model-call-start"));
        assert_eq!(
            json[1]["type"],
            serde_json::json!("model-call-response-metadata")
        );
        assert_eq!(json[1]["id"], serde_json::json!("resp_1"));
        assert_eq!(json[1]["modelId"], serde_json::json!("gpt-test"));
        assert_eq!(json[3]["type"], serde_json::json!("text-delta"));
        assert_eq!(json[6]["type"], serde_json::json!("tool-call"));
        assert_eq!(json[7]["type"], serde_json::json!("tool-result"));
        assert_eq!(json[10]["type"], serde_json::json!("model-call-end"));
        assert_eq!(json[10]["finishReason"], serde_json::json!("stop"));
        assert_eq!(json[10]["rawFinishReason"], serde_json::json!("stop"));
        assert_eq!(json[10]["usage"]["totalTokens"], serde_json::json!(12));
        assert_eq!(
            json[10]["providerMetadata"]["openai"]["responseId"],
            serde_json::json!("resp_1")
        );

        let roundtrip: Vec<LanguageModelStreamPart> =
            serde_json::from_value(json).expect("deserialize language model stream parts");
        assert_eq!(roundtrip[0].r#type(), "model-call-start");
        assert_eq!(roundtrip[1].r#type(), "model-call-response-metadata");
        assert_eq!(roundtrip[10].r#type(), "model-call-end");
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
            stop_when: vec![StopCondition::is_step_count(2)],
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
        assert_eq!(
            json["stopWhen"][0],
            serde_json::json!({ "type": "step-count", "stepCount": 2 })
        );
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
    #[allow(deprecated)]
    fn stop_condition_helpers_match_ai_sdk_builtin_semantics() {
        fn step_with_tool_calls(
            step_number: u32,
            tool_calls: Vec<ToolCall<String, JSONValue>>,
        ) -> GenerateTextStepResult {
            let timestamp = DateTime::parse_from_rfc3339("2026-04-24T00:00:00Z")
                .expect("timestamp")
                .with_timezone(&Utc);
            let response = GenerateTextResponseMetadata::new(LanguageModelResponseMetadata {
                id: format!("resp_{step_number}"),
                timestamp,
                model_id: "gpt-test".to_string(),
                headers: None,
            });

            GenerateTextStepResult {
                call_id: "call_1".to_string(),
                step_number,
                model: GenerateTextModelInfo::new("openai", "gpt-test"),
                tools_context: Context::new(),
                runtime_context: Context::new(),
                content: Vec::new(),
                text: String::new(),
                reasoning: Vec::new(),
                reasoning_text: None,
                files: Vec::new(),
                sources: Vec::new(),
                tool_calls: tool_calls.clone(),
                static_tool_calls: tool_calls,
                dynamic_tool_calls: Vec::new(),
                tool_results: Vec::new(),
                static_tool_results: Vec::new(),
                dynamic_tool_results: Vec::new(),
                finish_reason: FinishReason::Stop,
                raw_finish_reason: None,
                usage: LanguageModelUsage::default(),
                warnings: None,
                request: LanguageModelRequestMetadata { body: None },
                response,
                provider_metadata: None,
            }
        }

        let steps = vec![
            step_with_tool_calls(0, Vec::new()),
            step_with_tool_calls(
                1,
                vec![ToolCall::new(
                    "call_final",
                    "finalAnswer".to_string(),
                    serde_json::json!({}),
                )],
            ),
        ];

        assert!(StopCondition::is_step_count(2).is_met(&steps));
        assert!(step_count_is(2).is_met(&steps));
        assert!(!StopCondition::is_step_count(1).is_met(&steps));
        assert!(!StopCondition::is_loop_finished().is_met(&steps));
        assert!(StopCondition::has_tool_call(["search", "finalAnswer"]).is_met(&steps));
        assert!(!StopCondition::has_tool_call(["search"]).is_met(&steps));
        assert!(!StopCondition::custom(serde_json::json!({ "name": "custom" })).is_met(&steps));
        assert!(is_stop_condition_met(
            &[is_loop_finished(), has_tool_call(["finalAnswer"])],
            &steps
        ));
    }

    #[test]
    fn filter_active_tools_matches_ai_sdk_helper_semantics() {
        let tools = vec![
            Tool::function(
                "search",
                "Search docs",
                serde_json::json!({ "type": "object" }),
            ),
            Tool::function(
                "weather",
                "Get weather",
                serde_json::json!({ "type": "object" }),
            ),
            Tool::provider_defined("openai.web_search", "webSearch"),
        ];

        let all_tools = filter_active_tools::<String>(Some(&tools), None)
            .expect("tools should be returned when active tools are absent");
        assert_eq!(all_tools.len(), 3);

        let filtered = filter_active_tools(Some(&tools), Some(&["weather", "webSearch"]))
            .expect("filtered tools should be returned");
        let filtered_names: Vec<&str> = filtered.iter().map(tool_name).collect();
        assert_eq!(filtered_names, vec!["weather", "webSearch"]);

        let experimental_filtered =
            experimental_filter_active_tools(Some(&tools), Some(&["search"]))
                .expect("experimental alias should return filtered tools");
        let experimental_names: Vec<&str> = experimental_filtered.iter().map(tool_name).collect();
        assert_eq!(experimental_names, vec!["search"]);

        assert!(filter_active_tools::<String>(None, Some(&[])).is_none());
    }

    #[test]
    fn prune_messages_matches_ai_sdk_reasoning_and_tool_pruning() {
        let messages = vec![
            ModelMessage::Assistant(AssistantModelMessage::new(AssistantContent::parts(vec![
                AssistantContentPart::Text(TextPart::new("hello")),
                AssistantContentPart::Reasoning(ReasoningPart::new("remove this")),
                AssistantContentPart::ToolCall(ToolCallPart::new(
                    "call_1",
                    "search",
                    serde_json::json!({ "query": "rust" }),
                )),
                AssistantContentPart::ToolApprovalRequest(ToolApprovalRequest::new(
                    "approval_1",
                    "call_1",
                )),
            ]))),
            ModelMessage::Tool(ToolModelMessage::new(vec![
                ToolContentPart::ToolResult(ToolResultPart::new(
                    "call_1",
                    "search",
                    ToolResultOutput::json(serde_json::json!({ "answer": "ok" })),
                )),
                ToolContentPart::ToolApprovalResponse(ToolApprovalResponse::new(
                    "approval_1",
                    true,
                )),
            ])),
            ModelMessage::Assistant(AssistantModelMessage::new(AssistantContent::parts(vec![
                AssistantContentPart::Reasoning(ReasoningPart::new("keep this")),
                AssistantContentPart::ToolCall(ToolCallPart::new(
                    "call_2",
                    "weather",
                    serde_json::json!({ "city": "HK" }),
                )),
            ]))),
        ];

        let pruned = prune_messages(
            messages,
            PruneMessagesOptions::new()
                .with_reasoning(PruneReasoningMode::BeforeLastMessage)
                .with_tool_calls(vec![PruneToolCallRule::before_last_message()]),
        );

        assert_eq!(pruned.len(), 2);

        let ModelMessage::Assistant(first) = &pruned[0] else {
            panic!("first pruned message should be assistant");
        };
        let AssistantContent::Parts(first_parts) = &first.content else {
            panic!("first assistant message should keep parts");
        };
        assert_eq!(first_parts.len(), 1);
        assert!(matches!(first_parts[0], AssistantContentPart::Text(_)));

        let ModelMessage::Assistant(last) = &pruned[1] else {
            panic!("last pruned message should be assistant");
        };
        let AssistantContent::Parts(last_parts) = &last.content else {
            panic!("last assistant message should keep parts");
        };
        assert!(
            last_parts
                .iter()
                .any(|part| matches!(part, AssistantContentPart::Reasoning(_)))
        );
        assert!(
            last_parts
                .iter()
                .any(|part| matches!(part, AssistantContentPart::ToolCall(_)))
        );
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
    fn prepare_step_approval_and_repair_shapes_match_ai_sdk_options() {
        let mut provider_options = ProviderOptionsMap::new();
        provider_options.insert("anthropic", serde_json::json!({ "container": "ctn_1" }));
        let mut tools_context = Context::new();
        tools_context.insert("tenant".to_string(), serde_json::json!("docs"));
        let mut runtime_context = Context::new();
        runtime_context.insert("traceId".to_string(), serde_json::json!("trace_1"));
        let messages = vec![ModelMessage::Assistant(AssistantModelMessage::new(
            AssistantContent::text("calling search"),
        ))];

        let prepare_result = PrepareStepResult {
            model: Some(CallbackModelInfo::new("anthropic", "claude-test")),
            tool_choice: Some(ToolChoice::tool("search")),
            active_tools: Some(vec!["search".to_string()]),
            system: Some(SystemPrompt::Text("Use the docs.".to_string())),
            messages: Some(messages.clone()),
            tools_context: Some(tools_context.clone()),
            runtime_context: Some(runtime_context.clone()),
            provider_options: Some(provider_options),
        };
        let prepare_json =
            serde_json::to_value(&prepare_result).expect("serialize prepare-step result");

        assert_eq!(
            prepare_json["model"]["provider"],
            serde_json::json!("anthropic")
        );
        assert_eq!(
            prepare_json["model"]["modelId"],
            serde_json::json!("claude-test")
        );
        assert_eq!(
            prepare_json["toolChoice"]["toolName"],
            serde_json::json!("search")
        );
        assert_eq!(prepare_json["activeTools"][0], serde_json::json!("search"));
        assert_eq!(prepare_json["system"], serde_json::json!("Use the docs."));
        assert_eq!(
            prepare_json["toolsContext"]["tenant"],
            serde_json::json!("docs")
        );
        assert_eq!(
            prepare_json["runtimeContext"]["traceId"],
            serde_json::json!("trace_1")
        );
        assert_eq!(
            prepare_json["providerOptions"]["anthropic"]["container"],
            serde_json::json!("ctn_1")
        );

        let approval_status = ToolApprovalStatus::detailed(
            ToolApprovalStatusType::Denied,
            Some("policy denied".to_string()),
        );
        let approval_status_json =
            serde_json::to_value(&approval_status).expect("serialize approval status");
        assert_eq!(
            approval_status_json,
            serde_json::json!({ "type": "denied", "reason": "policy denied" })
        );
        assert_eq!(
            serde_json::to_value(ToolApprovalStatus::user_approval())
                .expect("serialize simple approval status"),
            serde_json::json!("user-approval")
        );

        let approval_config = ToolApprovalConfiguration::from([(
            "search".to_string(),
            ToolApprovalStatus::approved(),
        )]);
        let approval_config_json =
            serde_json::to_value(&approval_config).expect("serialize approval config");
        assert_eq!(
            approval_config_json["search"],
            serde_json::json!("approved")
        );

        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "query": "rust" }),
        );
        let approval_context = ToolApprovalDecisionContext {
            tool_call: tool_call.clone(),
            tools: Some(vec![Tool::function(
                "search",
                "Search docs",
                serde_json::json!({ "type": "object" }),
            )]),
            tools_context: tools_context.clone(),
            runtime_context: runtime_context.clone(),
            messages: messages.clone(),
        };
        let approval_context_json =
            serde_json::to_value(&approval_context).expect("serialize approval context");
        assert_eq!(
            approval_context_json["toolCall"]["toolName"],
            serde_json::json!("search")
        );
        assert_eq!(
            approval_context_json["toolsContext"]["tenant"],
            serde_json::json!("docs")
        );

        let repair_context = ToolCallRepairContext {
            system: Some(SystemPrompt::Text("Use tools carefully.".to_string())),
            messages,
            tool_call: tool_call.clone(),
            tools: vec![Tool::function(
                "lookup",
                "Lookup docs",
                serde_json::json!({ "type": "object" }),
            )],
            input_schemas: HashMap::from([(
                "lookup".to_string(),
                serde_json::json!({ "type": "object" }),
            )]),
            error: ToolCallRepairFunctionError::NoSuchTool(NoSuchToolError::new(
                "search",
                Some(vec!["lookup".to_string()]),
            )),
        };
        let repair_json = serde_json::to_value(&repair_context).expect("serialize repair context");
        assert_eq!(
            repair_json["toolCall"]["toolName"],
            serde_json::json!("search")
        );
        assert_eq!(
            repair_json["inputSchemas"]["lookup"]["type"],
            serde_json::json!("object")
        );
        assert_eq!(
            repair_json["error"],
            serde_json::json!({
                "type": "no-such-tool",
                "toolName": "search",
                "availableTools": ["lookup"],
                "message": "Model tried to call unavailable tool 'search'. Available tools: lookup."
            })
        );

        let invalid_tool_input_error =
            ToolCallRepairFunctionError::from(InvalidToolInputError::new(
                "search",
                "{",
                Some(serde_json::json!({ "message": "bad input" })),
            ));
        let invalid_tool_input_json =
            serde_json::to_value(&invalid_tool_input_error).expect("serialize invalid input error");
        assert_eq!(
            invalid_tool_input_json,
            serde_json::json!({
                "type": "invalid-tool-input",
                "toolName": "search",
                "toolInput": "{",
                "message": r#"Invalid input for tool search: {"message":"bad input"}"#,
                "cause": { "message": "bad input" }
            })
        );

        let repair_error = ToolCallRepairError::new(
            serde_json::from_value(repair_json["error"].clone()).expect("deserialize repair error"),
            Some(serde_json::json!({ "message": "repair failed" })),
        );
        let repair_error_json =
            serde_json::to_value(&repair_error).expect("serialize repair error wrapper");
        assert_eq!(
            repair_error_json["message"],
            serde_json::json!(r#"Error repairing tool call: {"message":"repair failed"}"#)
        );
        assert_eq!(
            repair_error_json["originalError"]["type"],
            serde_json::json!("no-such-tool")
        );
        assert_eq!(
            repair_error_json["cause"]["message"],
            serde_json::json!("repair failed")
        );

        let repair_result: ToolCallRepairResult = Some(ToolCall::new(
            "call_1",
            "lookup".to_string(),
            serde_json::json!({}),
        ));
        let repair_result_json =
            serde_json::to_value(&repair_result).expect("serialize repair result");
        assert_eq!(repair_result_json["type"], serde_json::json!("tool-call"));
        assert_eq!(repair_result_json["toolName"], serde_json::json!("lookup"));
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
