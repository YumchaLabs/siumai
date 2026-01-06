//! Gemini HTTP header helpers.
//!
//! Centralizes header construction for Gemini API requests.
//! Behavior:
//! - Always include `Content-Type: application/json`
//! - If `custom_headers` already contains `Authorization` (case-insensitive), do not inject `x-goog-api-key`
//! - Otherwise, if `api_key` is non-empty, inject `x-goog-api-key`
//! - Always merge `custom_headers` (custom headers win when names collide)
#![deny(unsafe_code)]

use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use reqwest::header::HeaderMap;
use std::collections::HashMap;

pub fn build_gemini_headers(
    api_key: &str,
    custom_headers: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let has_authorization = custom_headers
        .keys()
        .any(|k| k.eq_ignore_ascii_case("authorization"));

    let mut builder = HttpHeaderBuilder::new().with_json_content_type();
    if !has_authorization && !api_key.is_empty() {
        builder = builder.with_custom_auth("x-goog-api-key", api_key)?;
    }

    builder = builder.with_custom_headers(custom_headers)?;
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_gemini_headers_injects_api_key_when_no_authorization() {
        let headers = build_gemini_headers("k", &HashMap::new()).unwrap();
        assert_eq!(
            headers.get("x-goog-api-key").and_then(|v| v.to_str().ok()),
            Some("k")
        );
        assert_eq!(
            headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }

    #[test]
    fn build_gemini_headers_skips_api_key_when_authorization_present() {
        let mut extra = HashMap::new();
        extra.insert("Authorization".to_string(), "Bearer test-token".to_string());

        let headers = build_gemini_headers("k", &extra).unwrap();
        assert_eq!(
            headers.get("Authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer test-token")
        );
        assert!(headers.get("x-goog-api-key").is_none());
    }
}
