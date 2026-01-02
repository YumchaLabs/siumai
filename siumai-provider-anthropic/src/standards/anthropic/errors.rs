use crate::error::LlmError;
use serde_json::Value;

/// Classify Anthropic HTTP errors by parsing the structured error envelope.
///
/// Anthropic typically returns:
/// `{ "type": "error", "error": { "type": "...", "message": "..." } }`
///
/// Returns `None` when the body doesn't match the Anthropic envelope so callers
/// can fall back to the generic classifier.
pub fn classify_anthropic_http_error(
    provider: &str,
    status: u16,
    body_text: &str,
) -> Option<LlmError> {
    let json: Value = serde_json::from_str(body_text).ok()?;
    let error_obj = json.get("error")?;
    let error_type = error_obj.get("type").and_then(|v| v.as_str())?;
    let error_message = error_obj
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown error");

    let details = json.clone();

    let mapped = match error_type {
        "authentication_error" => LlmError::AuthenticationError(error_message.to_string()),
        "permission_error" => {
            LlmError::AuthenticationError(format!("Permission denied: {error_message}"))
        }
        "not_found_error" => LlmError::NotFound(error_message.to_string()),
        "rate_limit_error" => LlmError::RateLimitError(error_message.to_string()),
        "invalid_request_error" => LlmError::InvalidInput(error_message.to_string()),
        "overloaded_error" => LlmError::ApiError {
            code: 503,
            message: format!("{provider} service overloaded: {error_message}"),
            details: Some(details),
        },
        "api_error" => LlmError::ApiError {
            code: status,
            message: format!("{provider} API error: {error_message}"),
            details: Some(details),
        },
        other => LlmError::ApiError {
            code: status,
            message: format!("{provider} error ({other}): {error_message}"),
            details: Some(details),
        },
    };

    Some(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_error_mapping_authentication_error() {
        let body = r#"{"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}"#;
        let err = classify_anthropic_http_error("anthropic", 401, body).expect("classified");
        match err {
            LlmError::AuthenticationError(msg) => assert_eq!(msg, "Invalid API key"),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn anthropic_error_mapping_rate_limit_error() {
        let body =
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}"#;
        let err = classify_anthropic_http_error("anthropic", 429, body).expect("classified");
        match err {
            LlmError::RateLimitError(msg) => assert_eq!(msg, "Rate limited"),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn anthropic_error_mapping_returns_none_on_non_envelope() {
        let body = r#"{"message":"not anthropic"}"#;
        assert!(classify_anthropic_http_error("anthropic", 400, body).is_none());
    }
}
