//! Runtime error classification and presentation policy.
//!
//! The canonical `LlmError` data shape lives in `siumai-spec`. Runtime decisions such as retry
//! eligibility, user-facing messages, and recovery hints live in core so the spec crate remains a
//! passive contract crate.

use crate::error::LlmError;

/// Error category for runtime handling and presentation strategies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Network-related errors (connection, timeout, etc.)
    Network,
    /// Authentication and authorization errors.
    Authentication,
    /// Rate limiting and quota errors.
    RateLimit,
    /// Client-side errors (4xx HTTP status codes).
    Client,
    /// Server-side errors (5xx HTTP status codes).
    Server,
    /// Data parsing and serialization errors.
    Parsing,
    /// Input validation errors.
    Validation,
    /// Configuration errors.
    Configuration,
    /// Unsupported operations or models.
    Unsupported,
    /// Streaming-related errors.
    Stream,
    /// Provider-specific errors.
    Provider,
    /// Unknown or uncategorized errors.
    Unknown,
}

/// Runtime policy helpers for `LlmError`.
///
/// These methods intentionally live as a core-owned extension trait instead of inherent methods on
/// `LlmError`, because retry and presentation policy belongs to the provider-agnostic runtime
/// layer, not to the spec crate's serializable data contract.
pub trait LlmErrorExt {
    /// Checks if the error is retryable.
    fn is_retryable(&self) -> bool;

    /// Checks if the error is authentication-related.
    fn is_auth_error(&self) -> bool;

    /// Checks if the error is rate-limit related.
    fn is_rate_limit_error(&self) -> bool;

    /// Gets the HTTP status code of the error, if available.
    fn status_code(&self) -> Option<u16>;

    /// Gets the runtime error category.
    fn category(&self) -> ErrorCategory;

    /// Gets a concise safe user-facing error message.
    ///
    /// This message is suitable for ordinary UI/CLI display. It must not include raw provider
    /// response bodies, headers, serialized request bodies, provider debug payloads, or fallback
    /// `Display` output from unknown errors. Use the error's `Display` implementation or
    /// diagnostics helpers when raw detail is explicitly needed.
    fn user_message(&self) -> String;

    /// Gets suggested recovery actions for the error.
    fn recovery_suggestions(&self) -> Vec<String>;

    /// Gets the recommended retry delay in seconds.
    fn recommended_retry_delay(&self) -> Option<u64>;

    /// Gets the maximum number of retry attempts recommended for this error.
    fn max_retry_attempts(&self) -> u32;
}

impl LlmErrorExt for LlmError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::HttpError(error) => {
                let retryable_keywords = [
                    "timeout",
                    "connect",
                    "network",
                    "dns",
                    "socket",
                    "connection reset",
                    "connection refused",
                    "temporary failure",
                ];
                let lower = error.to_lowercase();
                retryable_keywords
                    .iter()
                    .any(|keyword| lower.contains(keyword))
            }
            Self::ApiError { code, .. } => matches!(*code, 408 | 429 | 500..=599),
            Self::RateLimitError(_) | Self::TimeoutError(_) | Self::ConnectionError(_) => true,
            Self::ContextualError {
                source_error: Some(source),
                ..
            } => source.is_retryable(),
            _ => false,
        }
    }

    fn is_auth_error(&self) -> bool {
        match self {
            Self::AuthenticationError(_) => true,
            Self::ApiError { code, .. } => *code == 401 || *code == 403,
            _ => false,
        }
    }

    fn is_rate_limit_error(&self) -> bool {
        match self {
            Self::RateLimitError(_) => true,
            Self::ApiError { code, .. } => *code == 429,
            _ => false,
        }
    }

    fn status_code(&self) -> Option<u16> {
        match self {
            Self::ApiError { code, .. } => Some(*code),
            _ => None,
        }
    }

    fn category(&self) -> ErrorCategory {
        match self {
            Self::HttpError(_) | Self::ConnectionError(_) | Self::TimeoutError(_) => {
                ErrorCategory::Network
            }
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => ErrorCategory::Authentication,
            Self::RateLimitError(_) | Self::QuotaExceededError(_) => ErrorCategory::RateLimit,
            Self::ApiError { code, .. } => match *code {
                429 => ErrorCategory::RateLimit,
                400..=499 => ErrorCategory::Client,
                500..=599 => ErrorCategory::Server,
                _ => ErrorCategory::Unknown,
            },
            Self::JsonError(_) | Self::ParseError(_) | Self::NoObjectGenerated { .. } => {
                ErrorCategory::Parsing
            }
            Self::InvalidInput(_) | Self::InvalidParameter(_) | Self::ToolValidationError(_) => {
                ErrorCategory::Validation
            }
            Self::ConfigurationError(_) => ErrorCategory::Configuration,
            Self::ModelNotSupported(_)
            | Self::UnsupportedOperation(_)
            | Self::UnsupportedToolType(_) => ErrorCategory::Unsupported,
            Self::StreamError(_) => ErrorCategory::Stream,
            Self::ProviderError { .. } | Self::ToolCallError(_) => ErrorCategory::Provider,
            Self::ContextualError {
                source_error: Some(source),
                ..
            } => source.category(),
            Self::ContextualError { .. } => ErrorCategory::Unknown,
            _ => ErrorCategory::Unknown,
        }
    }

    fn user_message(&self) -> String {
        match self {
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => {
                "Authentication failed. Please check your API key.".to_string()
            }
            Self::RateLimitError(_) => {
                "Rate limit exceeded. Please wait before making more requests.".to_string()
            }
            Self::QuotaExceededError(_) => {
                "API quota exceeded. Please check your usage limits.".to_string()
            }
            Self::ModelNotSupported(model) => {
                format!("The model '{model}' is not supported by this provider.")
            }
            Self::ConnectionError(_) | Self::TimeoutError(_) => {
                "Network connection failed. Please check your internet connection and try again."
                    .to_string()
            }
            Self::HttpError(_) => {
                "The HTTP request failed. Please check connectivity and provider availability."
                    .to_string()
            }
            Self::ApiError {
                code: 401 | 403, ..
            } => "Authentication failed. Please check your API key and provider permissions."
                .to_string(),
            Self::ApiError { code: 429, .. } => {
                "Rate limit exceeded. Please wait before making more requests.".to_string()
            }
            Self::ApiError {
                code: 400..=499, ..
            } => "The provider rejected the request. Please check your request parameters."
                .to_string(),
            Self::ApiError {
                code: 500..=599, ..
            } => "The service is temporarily unavailable. Please try again later.".to_string(),
            Self::ApiError { .. } => {
                "The provider returned an error. Check diagnostics for provider details."
                    .to_string()
            }
            Self::JsonError(_) | Self::ParseError(_) => {
                "The provider returned data that could not be parsed.".to_string()
            }
            Self::InvalidInput(_) | Self::InvalidParameter(_) | Self::ToolValidationError(_) => {
                "The request is invalid. Please check the input parameters.".to_string()
            }
            Self::IoError(_) => {
                "A local IO operation failed. Check diagnostics for details.".to_string()
            }
            Self::NotFound(_) => "The requested resource was not found.".to_string(),
            Self::StreamError(_) => {
                "The streaming request failed. Please retry the request.".to_string()
            }
            Self::ProviderError { provider, .. } => {
                format!("The provider '{provider}' returned an error.")
            }
            Self::ConfigurationError(_) => {
                "The client is not configured correctly. Please check configuration.".to_string()
            }
            Self::InternalError(_) | Self::ProcessingError(_) | Self::Other(_) => {
                "An internal error occurred. Check diagnostics for details.".to_string()
            }
            Self::UnsupportedOperation(_) => {
                "This operation is not supported by the selected provider or model.".to_string()
            }
            Self::ToolCallError(_) => {
                "A tool call failed. Check diagnostics for tool error details.".to_string()
            }
            Self::UnsupportedToolType(_) => {
                "This tool type is not supported by the selected provider or model.".to_string()
            }
            Self::ContextualError {
                source_error: Some(source),
                ..
            } => source.user_message(),
            Self::ContextualError { .. } => {
                "An error occurred while processing the request. Check diagnostics for details."
                    .to_string()
            }
            Self::NoImageGenerated { .. } => {
                "The provider completed the image request but returned no final image.".to_string()
            }
            Self::NoSpeechGenerated { .. } => {
                "The provider completed the speech request but returned no audio.".to_string()
            }
            Self::NoTranscriptGenerated { .. } => {
                "The provider completed the transcription request but returned no transcript."
                    .to_string()
            }
            Self::NoVideoGenerated { .. } => {
                "The provider completed the video request but returned no final video.".to_string()
            }
            Self::NoObjectGenerated { .. } => {
                "The provider completed the structured-output request but returned no valid object."
                    .to_string()
            }
        }
    }

    fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Self::AuthenticationError(_) | Self::MissingApiKey(_) => {
                vec![
                    "Verify your API key is correct and properly formatted".to_string(),
                    "Check if your API key has the required permissions for this operation"
                        .to_string(),
                    "Ensure your API key is not expired or revoked".to_string(),
                    "Verify you're using the correct API endpoint".to_string(),
                ]
            }
            Self::RateLimitError(_) => {
                vec![
                    "Implement exponential backoff with jitter".to_string(),
                    "Reduce request frequency".to_string(),
                    "Consider upgrading your API plan for higher limits".to_string(),
                    "Use request batching if supported".to_string(),
                ]
            }
            Self::QuotaExceededError(_) => {
                vec![
                    "Check your usage dashboard for current consumption".to_string(),
                    "Upgrade your API plan for higher quotas".to_string(),
                    "Wait for quota reset (usually monthly)".to_string(),
                    "Optimize your requests to use fewer tokens".to_string(),
                ]
            }
            Self::ConnectionError(_) | Self::TimeoutError(_) => {
                vec![
                    "Check your internet connection stability".to_string(),
                    "Retry the request with exponential backoff".to_string(),
                    "Increase timeout settings for large requests".to_string(),
                    "Check if the service is experiencing outages".to_string(),
                ]
            }
            Self::ModelNotSupported(_) => {
                vec![
                    "Use a supported model from the provider's model list".to_string(),
                    "Check the provider's documentation for available models".to_string(),
                    "Verify the model name is spelled correctly".to_string(),
                ]
            }
            Self::ApiError { code: 400, .. } => {
                vec![
                    "Check your request parameters for validity".to_string(),
                    "Ensure all required fields are provided".to_string(),
                    "Verify parameter types and formats".to_string(),
                ]
            }
            Self::ApiError {
                code: 500..=599, ..
            } => {
                vec![
                    "Retry the request after a delay (server error)".to_string(),
                    "Check the service status page for outages".to_string(),
                    "Contact support if the issue persists".to_string(),
                ]
            }
            Self::StreamError(_) => {
                vec![
                    "Retry the streaming request".to_string(),
                    "Check network stability for streaming".to_string(),
                    "Consider using non-streaming mode as fallback".to_string(),
                ]
            }
            Self::InvalidInput(_) | Self::InvalidParameter(_) => {
                vec![
                    "Validate your input parameters".to_string(),
                    "Check parameter constraints and limits".to_string(),
                    "Refer to the API documentation for valid formats".to_string(),
                ]
            }
            Self::NoImageGenerated { .. } => {
                vec![
                    "Check provider response metadata to confirm final images were produced"
                        .to_string(),
                    "Retry the request or simplify prompt/options if the provider returned an empty image list".to_string(),
                    "Inspect provider-specific metadata or moderation signals for filtered image outputs".to_string(),
                ]
            }
            Self::NoSpeechGenerated { .. } => {
                vec![
                    "Check provider response metadata to confirm speech audio bytes were produced"
                        .to_string(),
                    "Retry the request or simplify synthesis options if the provider returned empty audio".to_string(),
                    "Inspect provider-specific metadata for filtered or unsupported speech output".to_string(),
                ]
            }
            Self::NoTranscriptGenerated { .. } => {
                vec![
                    "Check provider response metadata to confirm transcript text was produced"
                        .to_string(),
                    "Retry the request or simplify transcription options if the provider returned empty text".to_string(),
                    "Inspect provider-specific metadata for filtering, language mismatch, or unsupported audio formats".to_string(),
                ]
            }
            Self::NoVideoGenerated { .. } => {
                vec![
                    "Check provider task metadata to confirm final video assets were produced"
                        .to_string(),
                    "Retry the request or reduce prompt/options complexity if the provider returned an empty result".to_string(),
                    "Inspect provider-specific metadata or task responses for moderation/processing filters".to_string(),
                ]
            }
            Self::NoObjectGenerated { .. } => {
                vec![
                    "Inspect the generated text and schema validation cause".to_string(),
                    "Retry with stronger JSON/schema instructions or a stricter response format"
                        .to_string(),
                    "Provide a repair callback when the provider frequently returns near-valid JSON"
                        .to_string(),
                ]
            }
            _ => vec![
                "Check the error details and documentation".to_string(),
                "Contact support if the issue persists".to_string(),
            ],
        }
    }

    fn recommended_retry_delay(&self) -> Option<u64> {
        match self {
            Self::RateLimitError(_) => Some(60),
            Self::ApiError { code: 429, .. } => Some(30),
            Self::ApiError {
                code: 500..=599, ..
            } => Some(5),
            Self::TimeoutError(_) => Some(10),
            Self::ConnectionError(_) => Some(5),
            _ => None,
        }
    }

    fn max_retry_attempts(&self) -> u32 {
        match self {
            Self::RateLimitError(_) => 3,
            Self::ApiError { code: 429, .. } => 3,
            Self::ApiError {
                code: 500..=599, ..
            } => 5,
            Self::TimeoutError(_) | Self::ConnectionError(_) => 3,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_video_generated_error_uses_specialized_message_and_guidance() {
        let error = LlmError::NoVideoGenerated {
            responses: Vec::new(),
        };

        assert_eq!(
            error.user_message(),
            "The provider completed the video request but returned no final video."
        );
        assert!(
            error
                .recovery_suggestions()
                .iter()
                .any(|tip| tip.contains("provider task metadata"))
        );
    }

    #[test]
    fn no_image_generated_error_uses_specialized_message_and_guidance() {
        let error = LlmError::NoImageGenerated {
            responses: Vec::new(),
        };

        assert_eq!(
            error.user_message(),
            "The provider completed the image request but returned no final image."
        );
        assert!(
            error
                .recovery_suggestions()
                .iter()
                .any(|tip| tip.contains("final images"))
        );
    }

    #[test]
    fn no_object_generated_error_uses_specialized_message_and_guidance() {
        let error = LlmError::NoObjectGenerated {
            message: "No object generated.".to_string(),
            text: Some("{\"title\":\"Ada\"}".to_string()),
            response: None,
            usage: None,
            finish_reason: None,
            cause: Some(Box::new(LlmError::ParseError("missing name".to_string()))),
        };

        assert_eq!(
            error.user_message(),
            "The provider completed the structured-output request but returned no valid object."
        );
        assert_eq!(error.category(), ErrorCategory::Parsing);
        assert!(
            error
                .recovery_suggestions()
                .iter()
                .any(|tip| tip.contains("repair callback"))
        );
    }

    #[test]
    fn no_speech_generated_error_uses_specialized_message_and_guidance() {
        let error = LlmError::NoSpeechGenerated {
            responses: Vec::new(),
        };

        assert_eq!(
            error.user_message(),
            "The provider completed the speech request but returned no audio."
        );
        assert!(
            error
                .recovery_suggestions()
                .iter()
                .any(|tip| tip.contains("empty audio"))
        );
    }

    #[test]
    fn no_transcript_generated_error_uses_specialized_message_and_guidance() {
        let error = LlmError::NoTranscriptGenerated {
            responses: Vec::new(),
        };

        assert_eq!(
            error.user_message(),
            "The provider completed the transcription request but returned no transcript."
        );
        assert!(
            error
                .recovery_suggestions()
                .iter()
                .any(|tip| tip.contains("empty text"))
        );
    }

    #[test]
    fn user_message_does_not_expose_api_error_raw_provider_details() {
        let error = LlmError::ApiError {
            code: 400,
            message: "bad request: Authorization: Bearer sk-secret body={\"prompt\":\"private\"}"
                .to_string(),
            details: Some(serde_json::json!({
                "headers": { "authorization": "Bearer sk-secret" },
                "body": { "prompt": "private" }
            })),
        };

        let user_message = error.user_message();
        assert_eq!(
            user_message,
            "The provider rejected the request. Please check your request parameters."
        );
        assert!(!user_message.contains("sk-secret"));
        assert!(!user_message.contains("private"));
        assert!(error.to_string().contains("sk-secret"));
    }

    #[test]
    fn user_message_does_not_fallback_to_display_for_raw_string_variants() {
        let raw = "provider body={\"secret\":\"private\"} header=authorization";
        let cases = [
            LlmError::HttpError(raw.to_string()),
            LlmError::ParseError(raw.to_string()),
            LlmError::ProviderError {
                provider: "provider-a".to_string(),
                message: raw.to_string(),
                error_code: Some("bad_request".to_string()),
            },
            LlmError::InternalError(raw.to_string()),
            LlmError::Other(raw.to_string()),
        ];

        for error in cases {
            let user_message = error.user_message();
            assert!(
                !user_message.contains("secret") && !user_message.contains("authorization"),
                "unsafe user message for {error:?}: {user_message}"
            );
        }
    }

    #[test]
    fn contextual_error_uses_source_safe_message_instead_of_context_raw_text() {
        let error = LlmError::ContextualError {
            context: "provider raw response".to_string(),
            message: "Authorization: Bearer sk-secret".to_string(),
            source_error: Some(Box::new(LlmError::ApiError {
                code: 503,
                message: "upstream body with secret".to_string(),
                details: None,
            })),
            metadata: std::collections::HashMap::from([(
                "request_body".to_string(),
                "{\"prompt\":\"private\"}".to_string(),
            )]),
        };

        let user_message = error.user_message();
        assert_eq!(
            user_message,
            "The service is temporarily unavailable. Please try again later."
        );
        assert!(!user_message.contains("sk-secret"));
        assert!(!user_message.contains("private"));
    }
}
