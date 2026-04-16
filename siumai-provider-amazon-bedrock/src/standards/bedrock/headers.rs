//! Shared Amazon Bedrock JSON header construction.

use crate::core::ProviderContext;
use crate::error::LlmError;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};

pub(crate) fn build_bedrock_json_headers(ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    // NOTE: Bedrock normally relies on AWS SigV4 signing.
    // This crate keeps auth lightweight for fixture alignment and lets callers inject
    // signed headers through `ProviderContext.http_extra_headers`.
    if let Some(api_key) = ctx
        .api_key
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|error| {
                LlmError::ConfigurationError(format!("Invalid Bedrock bearer token: {error}"))
            })?,
        );
    }

    for (key, value) in &ctx.http_extra_headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            headers.insert(name, value);
        }
    }

    Ok(headers)
}
