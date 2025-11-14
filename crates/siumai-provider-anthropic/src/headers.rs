//! Anthropic HTTP header helpers (extracted)
//!
//! These helpers mirror the behavior of the aggregator's
//! `ProviderHeaders::anthropic` function so that the aggregator can
//! delegate header construction to this crate when the
//! `provider-anthropic-external` feature is enabled.

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use siumai_core::error::LlmError;
use std::collections::HashMap;

/// Build headers for Anthropic API requests.
///
/// This follows Anthropic's official API documentation and keeps the
/// same behavior as the in-crate implementation used by the
/// aggregator:
/// - Use `x-api-key` for authentication.
/// - Set `Content-Type: application/json`.
/// - Set a default `anthropic-version` header.
/// - Respect the `anthropic-beta` header if provided in `http_extra`.
/// - Merge the remaining custom headers, letting them override
///   defaults when keys collide.
pub fn build_anthropic_json_headers(
    api_key: &str,
    http_extra: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();

    // Authentication header (Anthropic uses x-api-key)
    let name = HeaderName::from_static("x-api-key");
    let value = HeaderValue::from_str(api_key)
        .map_err(|e| LlmError::ConfigurationError(format!("Invalid Anthropic API key: {e}")))?;
    headers.insert(name, value);

    // JSON content type
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );

    // Anthropic version header (fixed for now; can be made configurable later)
    headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );

    // Handle anthropic-beta separately so that it can be passed through
    if let Some(beta_features) = http_extra.get("anthropic-beta") {
        headers.insert(
            HeaderName::from_static("anthropic-beta"),
            HeaderValue::from_str(beta_features).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid anthropic-beta header value: {e}"))
            })?,
        );
    }

    // Merge remaining custom headers (excluding anthropic-beta which was handled above)
    for (k, v) in http_extra.iter() {
        if k.eq_ignore_ascii_case("anthropic-beta") {
            continue;
        }

        let name = HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid header name '{k}': {e}")))?;
        let value = HeaderValue::from_str(v).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header value for '{k}': {e}"))
        })?;
        headers.insert(name, value);
    }

    Ok(headers)
}
