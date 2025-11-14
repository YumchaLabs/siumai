//! MiniMaxi-specific error mapping helpers.
//!
//! MiniMaxi generally forwards Anthropic/OpenAI-style errors. This helper
//! provides a single place to translate MiniMaxi's structured error fields
//! into `LlmError` variants so that both the aggregator and provider crate
//! can share consistent behavior.

use siumai_core::error::LlmError;

/// Map a MiniMaxi error type to `LlmError`.
///
/// This is intentionally conservative and mirrors typical API error
/// semantics; it can be extended as we refine provider behavior.
pub fn map_minimaxi_error(
    status_code: u16,
    error_type: &str,
    error_message: &str,
    error_details: serde_json::Value,
) -> LlmError {
    match error_type {
        "authentication_error" | "unauthorized" => {
            LlmError::AuthenticationError(error_message.to_string())
        }
        "permission_error" | "forbidden" => {
            LlmError::AuthenticationError(format!("Permission denied: {error_message}"))
        }
        "invalid_request_error" | "bad_request" => {
            LlmError::InvalidInput(error_message.to_string())
        }
        "not_found_error" | "not_found" => LlmError::NotFound(error_message.to_string()),
        "request_too_large" => {
            LlmError::InvalidInput(format!("Request too large: {error_message}"))
        }
        "rate_limit_error" | "rate_limited" => LlmError::RateLimitError(error_message.to_string()),
        "api_error" | "server_error" => LlmError::ProviderError {
            provider: "minimaxi".to_string(),
            message: format!("Internal API error: {error_message}"),
            error_code: Some(error_type.to_string()),
        },
        "overloaded_error" | "unavailable" => LlmError::ProviderError {
            provider: "minimaxi".to_string(),
            message: format!("API temporarily overloaded: {error_message}"),
            error_code: Some(error_type.to_string()),
        },
        _ => LlmError::ApiError {
            code: status_code,
            message: format!("MiniMaxi API error ({error_type}): {error_message}"),
            details: Some(error_details),
        },
    }
}
