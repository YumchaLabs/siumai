//! xAI HTTP header helpers (extracted)
//!
//! These helpers mirror the behavior of the aggregator's
//! `ProviderHeaders::xai` function so that the aggregator can
//! delegate header construction to this crate when the
//! `provider-xai-external` feature is enabled.

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use siumai_core::error::LlmError;
use std::collections::HashMap;

/// Build headers for xAI API requests.
///
/// Follows xAI's official API documentation:
/// - Use `Authorization: Bearer <api_key>` for authentication.
/// - Set `Content-Type: application/json`.
/// - Merge the remaining custom headers, letting them override defaults
///   when keys collide.
pub fn build_xai_json_headers(
    api_key: &str,
    http_extra: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();

    // Authentication header
    headers.insert(
        reqwest::header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|e| {
            LlmError::ConfigurationError(format!(
                "Invalid xAI API key for Authorization header: {e}"
            ))
        })?,
    );

    // JSON content type
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
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
