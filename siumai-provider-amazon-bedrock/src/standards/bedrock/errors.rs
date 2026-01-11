//! Amazon Bedrock HTTP error classification.
//!
//! The Bedrock runtime endpoints usually return JSON error bodies shaped like:
//! `{ "message": "...", "__type": "..." }`. This module keeps the provider error
//! message lossless when possible, while mapping status codes into retry-friendly
//! unified error variants.

use crate::error::LlmError;
use reqwest::header::HeaderMap;

fn extract_message(body_text: &str) -> Option<String> {
    let json = serde_json::from_str::<serde_json::Value>(body_text).ok()?;

    let message = json
        .get("message")
        .and_then(|v| v.as_str())
        .or_else(|| json.get("Message").and_then(|v| v.as_str()))
        .or_else(|| json.get("errorMessage").and_then(|v| v.as_str()))
        .or_else(|| {
            json.get("error")
                .and_then(|v| v.get("message"))
                .and_then(|v| v.as_str())
        })
        .map(|s| s.trim().to_string())?;

    if message.is_empty() {
        None
    } else {
        Some(message)
    }
}

fn extract_error_type(body_text: &str) -> Option<String> {
    let json = serde_json::from_str::<serde_json::Value>(body_text).ok()?;
    json.get("__type")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn classify_bedrock_http_error(
    _provider_id: &'static str,
    status: u16,
    body_text: &str,
    _headers: &HeaderMap,
) -> Option<LlmError> {
    if body_text.trim().is_empty() {
        return None;
    }

    let message = extract_message(body_text).unwrap_or_else(|| body_text.to_string());
    let lower = message.to_lowercase();
    let err_type = extract_error_type(body_text)
        .unwrap_or_default()
        .to_lowercase();

    let looks_rate_limited = err_type.contains("throttl") || lower.contains("throttl");
    if status == 429 || looks_rate_limited {
        return Some(LlmError::RateLimitError(message));
    }

    if status == 401 || status == 403 {
        return Some(LlmError::AuthenticationError(message));
    }

    if status == 404 {
        return Some(LlmError::NotFound(message));
    }

    if status == 400 || status == 413 || status == 415 {
        return Some(LlmError::InvalidInput(message));
    }

    Some(LlmError::ApiError {
        code: status,
        message,
        details: serde_json::from_str::<serde_json::Value>(body_text).ok(),
    })
}
