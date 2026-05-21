//! TogetherAI JSON header construction.

use crate::core::ProviderContext;
use crate::error::LlmError;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};

/// Build TogetherAI JSON headers from a provider execution context.
pub(crate) fn build_togetherai_json_headers(ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
    let api_key = ctx.api_key.as_deref().ok_or_else(|| {
        LlmError::ConfigurationError("TogetherAI API key is required".to_string())
    })?;

    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        format!("Bearer {api_key}").parse().map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid TogetherAI API key: {e}"))
        })?,
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    // Preserve custom headers (Vercel-aligned: user headers override defaults).
    for (k, v) in &ctx.http_extra_headers {
        let name = HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
            LlmError::InvalidParameter(format!("Invalid TogetherAI header name '{k}': {e}"))
        })?;
        let value = HeaderValue::from_str(v).map_err(|e| {
            LlmError::InvalidParameter(format!("Invalid TogetherAI header value '{v}': {e}"))
        })?;
        headers.insert(name, value);
    }

    Ok(headers)
}
