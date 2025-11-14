//! Anthropic-specific error mapping helpers.
//!
//! These helpers centralize the mapping from Anthropic's structured
//! error responses to `LlmError` so that both the aggregator and any
//! direct users of this crate can share the same behavior.

use siumai_core::error::LlmError;

/// Map Anthropic error types to `LlmError` according to official documentation.
///
/// See: <https://docs.anthropic.com/en/api/errors>
pub fn map_anthropic_error(
    status_code: u16,
    error_type: &str,
    error_message: &str,
    error_details: serde_json::Value,
) -> LlmError {
    match error_type {
        "authentication_error" => LlmError::AuthenticationError(error_message.to_string()),
        "permission_error" => {
            LlmError::AuthenticationError(format!("Permission denied: {error_message}"))
        }
        "invalid_request_error" => LlmError::InvalidInput(error_message.to_string()),
        "not_found_error" => LlmError::NotFound(error_message.to_string()),
        "request_too_large" => {
            LlmError::InvalidInput(format!("Request too large: {error_message}"))
        }
        "rate_limit_error" => LlmError::RateLimitError(error_message.to_string()),
        "api_error" => LlmError::ProviderError {
            provider: "anthropic".to_string(),
            message: format!("Internal API error: {error_message}"),
            error_code: Some("api_error".to_string()),
        },
        "overloaded_error" => LlmError::ProviderError {
            provider: "anthropic".to_string(),
            message: format!("API temporarily overloaded: {error_message}"),
            error_code: Some("overloaded_error".to_string()),
        },
        _ => LlmError::ApiError {
            code: status_code,
            message: format!("Anthropic API error ({error_type}): {error_message}"),
            details: Some(error_details),
        },
    }
}
