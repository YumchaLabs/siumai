//! MiniMaxi HTTP header helpers.
//!
//! MiniMaxi exposes both Anthropic-compatible and OpenAI-compatible
//! endpoints. Chat traffic uses Anthropic-style authentication, while
//! audio/image/video/music use OpenAI-style Bearer authentication.

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use siumai_core::error::LlmError;
use std::collections::HashMap;

/// Build Anthropic-style headers for MiniMaxi chat API.
///
/// This mirrors the behavior of `ProviderHeaders::anthropic` in the
/// aggregator crate, but lives close to the provider implementation.
pub fn build_anthropic_headers(
    api_key: &str,
    custom_headers: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();

    // x-api-key authentication
    headers.insert(
        HeaderName::from_static("x-api-key"),
        HeaderValue::from_str(api_key)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid MiniMaxi API key: {e}")))?,
    );

    // JSON content type
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    // MiniMaxi Anthropic-compatible APIs expect an Anthropic version header as well.
    headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );

    // Pass through additional custom headers
    for (k, v) in custom_headers {
        let name = HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid header name '{k}': {e}")))?;
        let value = HeaderValue::from_str(v).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header value for '{k}': {e}"))
        })?;
        headers.insert(name, value);
    }

    Ok(headers)
}

/// Build OpenAI-style authentication headers for MiniMaxi's
/// OpenAI-compatible endpoints (audio/image/video/music).
pub fn build_openai_auth_headers(api_key: &str) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
    headers
}
