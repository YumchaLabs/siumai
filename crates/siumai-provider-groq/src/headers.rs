//! Groq HTTP header helpers (extracted)
//!
//! These helpers mirror the behavior of the aggregator's
//! `ProviderHeaders::groq` function so that the aggregator can
//! delegate header construction to this crate when the
//! `provider-groq-external` feature is enabled.

use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use siumai_core::error::LlmError;
use std::collections::HashMap;

/// Build headers for Groq API requests.
///
/// Current behavior:
/// - `Authorization: Bearer <api_key>`
/// - `Content-Type: application/json`
/// - `User-Agent: siumai/0.1.0 (groq-provider)` (legacy value; kept for BC)
/// - Merge custom headers, letting them override defaults on collision.
pub fn build_groq_json_headers(
    api_key: &str,
    http_extra: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderName};

    let mut headers = HeaderMap::new();

    // Bearer auth
    let auth = format!("Bearer {api_key}");
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&auth).map_err(|e| {
            LlmError::ConfigurationError(format!(
                "Invalid Groq API key for Authorization header: {e}"
            ))
        })?,
    );

    // JSON content type
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    // Legacy user agent (kept for BC with in-crate implementation)
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("siumai/0.1.0 (groq-provider)"),
    );

    // Merge remaining custom headers
    for (k, v) in http_extra.iter() {
        let name = HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid header name '{k}': {e}")))?;
        let value = HeaderValue::from_str(v).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header value for '{k}': {e}"))
        })?;
        headers.insert(name, value);
    }

    Ok(headers)
}
