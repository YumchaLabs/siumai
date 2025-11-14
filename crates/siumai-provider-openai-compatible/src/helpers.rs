//! Helper utilities for OpenAI-compatible providers

use crate::{adapter::ProviderAdapter, types::RequestType};
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use siumai_core::error::LlmError;
use std::collections::HashMap;

/// Build full URL from base and adapter-provided route
pub fn build_url(base_url: &str, adapter: &dyn ProviderAdapter, req: RequestType) -> String {
    let path = adapter.route_for(req);
    format!("{}/{}", base_url.trim_end_matches('/'), path)
}

/// Build JSON headers for OpenAI-compatible providers with sensible merging order.
///
/// Precedence (later overrides earlier on key collision, case-insensitive):
/// - defaults (Content-Type, Accept, Authorization: Bearer <api_key>)
/// - `http_extra` (HashMap)
/// - `config_headers` (HeaderMap)
/// - `adapter_headers` (HeaderMap)
pub fn build_json_headers(
    api_key: &str,
    http_extra: &HashMap<String, String>,
    config_headers: &HeaderMap,
    adapter_headers: &HeaderMap,
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();

    // Defaults
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));

    // Only inject Authorization if not already present in any source
    let has_auth = |map: &HeaderMap| {
        map.keys()
            .any(|k| k.as_str().eq_ignore_ascii_case("authorization"))
    };
    let extra_has_auth = http_extra
        .keys()
        .any(|k| k.eq_ignore_ascii_case("authorization"));
    if !has_auth(&headers)
        && !extra_has_auth
        && !has_auth(config_headers)
        && !has_auth(adapter_headers)
    {
        let v = format!("Bearer {}", api_key);
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_str(&v)
                .map_err(|e| LlmError::Other(format!("invalid auth header: {}", e)))?,
        );
    }

    // Merge http_extra (string map)
    for (k, v) in http_extra.iter() {
        let name = HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| LlmError::Other(format!("invalid header name '{}': {}", k, e)))?;
        let value = HeaderValue::from_str(v)
            .map_err(|e| LlmError::Other(format!("invalid header value for '{}': {}", k, e)))?;
        headers.insert(name, value);
    }

    // Merge config_headers
    for (k, v) in config_headers.iter() {
        headers.insert(k, v.clone());
    }

    // Merge adapter_headers
    for (k, v) in adapter_headers.iter() {
        headers.insert(k, v.clone());
    }

    Ok(headers)
}

/// Build JSON headers and allow provider-specific policy adjustments.
pub fn build_json_headers_with_provider(
    provider_id: &str,
    api_key: &str,
    http_extra: &HashMap<String, String>,
    config_headers: &HeaderMap,
    adapter_headers: &HeaderMap,
) -> Result<HeaderMap, LlmError> {
    let mut headers = build_json_headers(api_key, http_extra, config_headers, adapter_headers)?;
    // Provider-specific policies（保守实现，避免注入未知值）
    match provider_id {
        // OpenRouter 建议提供 HTTP-Referer 与 X-Title；
        // 若用户仅提供了标准 Referer，我们复制为 HTTP-Referer，便于通过。
        "openrouter" => {
            let http_referer = HeaderName::from_static("http-referer");
            let referer = HeaderName::from_static("referer");
            if !headers.contains_key(&http_referer) {
                if let Some(val) = headers.get(&referer).cloned() {
                    headers.insert(http_referer, val);
                }
            }
            // X-Title 若用户未提供，不做默认注入，避免泄露不必要信息。
        }
        _ => {}
    }
    Ok(headers)
}
