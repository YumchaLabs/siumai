//! Gemini HTTP header helpers (extracted)
//!
//! 该模块复用聚合层 `ProviderHeaders::gemini` 的语义：
//! - 若自定义头中已经包含 `Authorization`（如 Vertex Bearer），则不注入 `x-goog-api-key`
//! - 否则在 API key 非空时注入 `x-goog-api-key`
//! - 始终设置 `Content-Type: application/json`，并合并自定义头

use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use siumai_core::error::LlmError;
use std::collections::HashMap;

/// 构建 Gemini JSON 请求头。
pub fn build_gemini_json_headers(
    api_key: &str,
    http_extra: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();

    // 始终使用 JSON Content-Type
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    // 先合并自定义头
    for (k, v) in http_extra {
        let name = HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header name '{k}': {e}"))
        })?;
        let value = HeaderValue::from_str(v).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header value for '{k}': {e}"))
        })?;
        headers.insert(name, value);
    }

    // 检查是否已有 Authorization（如 Vertex Bearer）
    let has_authorization = http_extra
        .keys()
        .any(|k| k.eq_ignore_ascii_case("authorization"));

    if !has_authorization && !api_key.is_empty() {
        headers.insert(
            HeaderName::from_static("x-goog-api-key"),
            HeaderValue::from_str(api_key).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid Gemini API key: {e}"))
            })?,
        );
    }

    Ok(headers)
}

