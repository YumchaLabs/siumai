use super::*;

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
        "api_error" => LlmError::ApiError {
            code: status_code,
            message: format!("Anthropic API error: {error_message}"),
            details: Some(error_details),
        },
        "overloaded_error" => LlmError::ApiError {
            // Vercel AI SDK parity: represent overloaded as a synthetic 529.
            code: 529,
            message: format!("Anthropic service overloaded: {error_message}"),
            details: Some(error_details),
        },
        _ => LlmError::ApiError {
            code: status_code,
            message: format!("Anthropic API error ({error_type}): {error_message}"),
            details: Some(error_details),
        },
    }
}

#[cfg(test)]
mod error_mapping_tests {
    use super::*;

    #[test]
    fn map_anthropic_error_overloaded_is_retryable() {
        let err = map_anthropic_error(
            200,
            "overloaded_error",
            "Overloaded",
            serde_json::json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}),
        );
        match err {
            LlmError::ApiError { code, .. } => assert_eq!(code, 529),
            other => panic!("unexpected error variant: {other:?}"),
        }
        assert!(err.is_retryable());
    }
}
