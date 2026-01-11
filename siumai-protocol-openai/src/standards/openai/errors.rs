use crate::error::LlmError;
use serde_json::Value;

/// Classify OpenAI-compatible HTTP errors by parsing the standard error envelope.
///
/// OpenAI-style APIs typically return:
/// `{ "error": { "message": "...", "type": "...", "code": "..." } }`
///
/// Returns `None` when the body doesn't match the OpenAI envelope so callers
/// can fall back to the generic classifier.
pub fn classify_openai_compatible_http_error(
    provider: &str,
    status: u16,
    body_text: &str,
) -> Option<LlmError> {
    let json: Value = serde_json::from_str(body_text).ok()?;
    let error_obj = json.get("error")?;

    let message = error_obj
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown error");
    let error_type = error_obj.get("type").and_then(|v| v.as_str());
    let error_code = error_obj.get("code").and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    });

    let details = json.clone();

    // Prefer structured `type`, otherwise fall back to message heuristics.
    let mapped = match error_type.unwrap_or("") {
        "authentication_error" => LlmError::AuthenticationError(message.to_string()),
        "rate_limit_error" => LlmError::RateLimitError(message.to_string()),
        "insufficient_quota" => LlmError::QuotaExceededError(message.to_string()),
        "invalid_request_error" => LlmError::InvalidInput(message.to_string()),
        "not_found_error" => LlmError::NotFound(message.to_string()),
        "" => map_openai_message_heuristics(provider, status, message, details, error_code),
        other => LlmError::ApiError {
            code: status,
            message: format!("{provider} API error ({other}): {message}"),
            details: Some(details),
        },
    };

    Some(mapped)
}

fn map_openai_message_heuristics(
    provider: &str,
    status: u16,
    message: &str,
    details: Value,
    error_code: Option<String>,
) -> LlmError {
    let lower = message.to_lowercase();

    if status == 401 || lower.contains("api key") || lower.contains("unauthorized") {
        return LlmError::AuthenticationError(message.to_string());
    }

    if status == 429 || lower.contains("rate limit") || lower.contains("ratelimit") {
        return LlmError::RateLimitError(message.to_string());
    }

    if lower.contains("quota") || lower.contains("insufficient_quota") {
        return LlmError::QuotaExceededError(message.to_string());
    }

    if status == 404 {
        return LlmError::NotFound(message.to_string());
    }

    if status == 400 || lower.contains("invalid") {
        return LlmError::InvalidInput(message.to_string());
    }

    match error_code {
        Some(code) => LlmError::ProviderError {
            provider: provider.to_string(),
            message: message.to_string(),
            error_code: Some(code),
        },
        None => LlmError::ApiError {
            code: status,
            message: format!("{provider} API error: {message}"),
            details: Some(details),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_error_mapping_invalid_request_error() {
        let body =
            r#"{"error":{"message":"bad request","type":"invalid_request_error","code":null}}"#;
        let err = classify_openai_compatible_http_error("openai", 400, body).expect("classified");
        match err {
            LlmError::InvalidInput(msg) => assert_eq!(msg, "bad request"),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn openai_error_mapping_insufficient_quota() {
        let body = r#"{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}"#;
        let err = classify_openai_compatible_http_error("openai", 429, body).expect("classified");
        match err {
            LlmError::QuotaExceededError(msg) => assert!(msg.contains("quota")),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn openai_error_mapping_supports_numeric_code_field() {
        let body = r#"{"error":{"message":"bad gateway","type":null,"code":123}}"#;
        let err = classify_openai_compatible_http_error("openai", 502, body).expect("classified");
        match err {
            LlmError::ProviderError {
                provider,
                message,
                error_code,
            } => {
                assert_eq!(provider, "openai");
                assert_eq!(message, "bad gateway");
                assert_eq!(error_code.as_deref(), Some("123"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn openai_error_mapping_message_heuristics_resource_exhausted() {
        let body =
            r#"{"error":{"message":"{\"error\":{\"status\":\"RESOURCE_EXHAUSTED\"}}","type":""}}"#;
        let err = classify_openai_compatible_http_error("openai", 429, body).expect("classified");
        match err {
            LlmError::RateLimitError(msg) => assert!(msg.contains("RESOURCE_EXHAUSTED")),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn openai_error_mapping_returns_none_on_non_envelope() {
        let body = r#"{"message":"not openai"}"#;
        assert!(classify_openai_compatible_http_error("openai", 400, body).is_none());
    }
}
