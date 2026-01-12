//! OpenAI-compatible header helpers.
//!
//! This module centralizes header construction for OpenAI-compatible protocol specs.

use crate::core::ProviderContext;
use crate::error::LlmError;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};

/// Build OpenAI-compatible JSON headers from a ProviderContext.
///
/// Includes:
/// - `Content-Type: application/json`
/// - `Authorization: Bearer <api_key>` (if present)
/// - `OpenAI-Organization` / `OpenAI-Project` (if present)
/// - `http_extra_headers` passthrough
pub fn build_openai_compatible_json_headers(ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if let Some(api_key) = &ctx.api_key {
        headers.insert(
            "Authorization",
            format!("Bearer {}", api_key)
                .parse()
                .map_err(|e| LlmError::InvalidParameter(format!("Invalid API key: {e}")))?,
        );
    }

    if let Some(org) = ctx.organization.as_deref()
        && !org.is_empty()
    {
        headers.insert(
            "OpenAI-Organization",
            org.parse().map_err(|e| {
                LlmError::InvalidParameter(format!("Invalid OpenAI-Organization header: {e}"))
            })?,
        );
    }

    if let Some(proj) = ctx.project.as_deref()
        && !proj.is_empty()
    {
        headers.insert(
            "OpenAI-Project",
            proj.parse().map_err(|e| {
                LlmError::InvalidParameter(format!("Invalid OpenAI-Project header: {e}"))
            })?,
        );
    }

    for (k, v) in &ctx.http_extra_headers {
        let header_name: HeaderName = k
            .parse()
            .map_err(|e| LlmError::InvalidParameter(format!("Invalid header name '{k}': {e}")))?;
        let header_value: HeaderValue = v
            .parse()
            .map_err(|e| LlmError::InvalidParameter(format!("Invalid header value '{v}': {e}")))?;
        headers.insert(header_name, header_value);
    }

    Ok(headers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_openai_compatible_json_headers_includes_auth_and_org_project() {
        let ctx = ProviderContext::new(
            "openai",
            "https://api.openai.com/v1",
            Some("sk-test".to_string()),
            std::collections::HashMap::new(),
        )
        .with_org_project(Some("org-123".to_string()), Some("proj-456".to_string()));

        let headers = build_openai_compatible_json_headers(&ctx).unwrap();
        assert_eq!(
            headers.get("Authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer sk-test")
        );
        assert_eq!(
            headers
                .get("OpenAI-Organization")
                .and_then(|v| v.to_str().ok()),
            Some("org-123")
        );
        assert_eq!(
            headers.get("OpenAI-Project").and_then(|v| v.to_str().ok()),
            Some("proj-456")
        );
    }
}
