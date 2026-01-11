//! Cohere HTTP error classification.
//!
//! This module aims to preserve Cohere error messages losslessly (when possible)
//! while keeping retry-friendly error variants.

use crate::error::LlmError;
use reqwest::header::HeaderMap;

fn extract_message(body_text: &str) -> Option<String> {
    let json = serde_json::from_str::<serde_json::Value>(body_text).ok()?;

    let message = json
        .get("message")
        .and_then(|v| v.as_str())
        .or_else(|| json.get("error").and_then(|v| v.as_str()))
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

pub fn classify_cohere_http_error(
    _provider_id: &'static str,
    status: u16,
    body_text: &str,
    _headers: &HeaderMap,
) -> Option<LlmError> {
    if body_text.trim().is_empty() {
        return None;
    }

    let message = extract_message(body_text).unwrap_or_else(|| body_text.to_string());
    let lower = body_text.to_lowercase();

    if status == 429 {
        return Some(LlmError::RateLimitError(message));
    }
    if status == 401 || status == 403 {
        return Some(LlmError::AuthenticationError(message));
    }
    if status == 404 {
        return Some(LlmError::NotFound(message));
    }
    if status == 413 || status == 415 || status == 400 {
        // Keep parity with the generic classifier: map bad requests to InvalidInput.
        if status == 400 {
            let quota_like = lower.contains("quota") || lower.contains("exceed");
            let rate_like = lower.contains("rate limit")
                || lower.contains("ratelimit")
                || lower.contains("resource_exhausted")
                || lower.contains("rate_limit_exceeded")
                || lower.contains("ratelimitexceeded")
                || lower.contains("ratelimit exceeded");
            if quota_like {
                return Some(LlmError::QuotaExceededError(message));
            }
            if rate_like {
                return Some(LlmError::RateLimitError(message));
            }
        }
        return Some(LlmError::InvalidInput(message));
    }

    Some(LlmError::ApiError {
        code: status,
        message,
        details: serde_json::from_str::<serde_json::Value>(body_text).ok(),
    })
}
