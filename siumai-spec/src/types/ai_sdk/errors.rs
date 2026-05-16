use crate::types::chat::{UiMessagePart, UiMessageRole};
use crate::types::{FinishReason, ToolResultOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    GenerateObjectResponseMetadata, ImageModelResponseMetadata, JSONValue, LanguageModelStreamPart,
    LanguageModelUsage, SpeechModelResponseMetadata, TranscriptionModelResponseMetadata,
    VideoModelResponseMetadata,
};

pub(super) fn ai_sdk_error_message(cause: Option<&JSONValue>) -> String {
    match cause {
        None | Some(JSONValue::Null) => "unknown error".to_string(),
        Some(JSONValue::String(message)) => message.clone(),
        Some(cause) => cause.to_string(),
    }
}

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

/// AI SDK provider-utils `DownloadError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DownloadError {
    /// Human-readable error message.
    pub message: String,
    /// URL that failed to download.
    pub url: String,
    /// HTTP status code when the failure came from a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_code: Option<u16>,
    /// HTTP status text when the failure came from a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_text: Option<String>,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<Box<JSONValue>>,
}

impl DownloadError {
    /// Create a `DownloadError` with the upstream default message.
    pub fn new(
        url: impl Into<String>,
        status_code: Option<u16>,
        status_text: Option<String>,
        cause: Option<JSONValue>,
    ) -> Self {
        let url = url.into();
        let message = match cause.as_ref() {
            Some(cause) if !cause.is_null() => {
                format!(
                    "Failed to download {url}: {}",
                    ai_sdk_error_message(Some(cause))
                )
            }
            _ => {
                let status_code = status_code.map(|code| code.to_string()).unwrap_or_default();
                let status_text = status_text.as_deref().unwrap_or_default();
                format!("Failed to download {url}: {status_code} {status_text}")
            }
        };

        Self {
            message,
            url,
            status_code,
            status_text,
            cause: cause.map(Box::new),
        }
    }

    /// Create a `DownloadError` from an HTTP response status.
    pub fn from_status(
        url: impl Into<String>,
        status_code: u16,
        status_text: impl Into<String>,
    ) -> Self {
        Self::new(url, Some(status_code), Some(status_text.into()), None)
    }

    /// Create a `DownloadError` from a serializable cause payload.
    pub fn from_cause(url: impl Into<String>, cause: JSONValue) -> Self {
        Self::new(url, None, None, Some(cause))
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

/// AI SDK registry `NoSuchProviderError` passive error data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NoSuchProviderError {
    /// Human-readable error message.
    pub message: String,
    /// Missing model/provider id.
    pub model_id: String,
    /// Requested model family.
    pub model_type: NoSuchModelType,
    /// Missing provider id.
    pub provider_id: String,
    /// Provider ids available in the registry.
    pub available_providers: Vec<String>,
}

impl NoSuchProviderError {
    /// Create a `NoSuchProviderError` with the upstream default message.
    pub fn new(
        model_id: impl Into<String>,
        model_type: NoSuchModelType,
        provider_id: impl Into<String>,
        available_providers: Vec<String>,
    ) -> Self {
        let model_id = model_id.into();
        let provider_id = provider_id.into();
        let available = available_providers.join(",");
        Self {
            message: format!("No such provider: {provider_id} (available providers: {available})"),
            model_id,
            model_type,
            provider_id,
            available_providers,
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
    pub value: Box<JSONValue>,
    /// Provider/application cause payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<Box<JSONValue>>,
    /// Validation context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Box<TypeValidationContext>>,
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
            value: Box::new(value),
            cause: cause.map(Box::new),
            context: context.map(Box::new),
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
    pub response: Option<GenerateObjectResponseMetadata>,
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
        response: Option<GenerateObjectResponseMetadata>,
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
