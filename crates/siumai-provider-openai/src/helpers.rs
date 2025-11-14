//! OpenAI provider helpers (stateless)
//!
//! 仅包含与路由/后缀相关的无状态工具函数，聚合侧在启用
//! `provider-openai-external` 时调用，以保持行为一致且易于迁移。

/// Chat 路由后缀：根据是否启用 Responses API 返回不同路径
pub fn chat_path(use_responses_api: bool) -> &'static str {
    if use_responses_api {
        "/responses"
    } else {
        "/chat/completions"
    }
}

/// Embedding 路由后缀
pub fn embedding_path() -> &'static str {
    "/embeddings"
}

/// Image 生成路由后缀
pub fn image_generation_path() -> &'static str {
    "/images/generations"
}

/// Image 编辑路由后缀
pub fn image_edit_path() -> &'static str {
    "/images/edits"
}

/// Image 变体路由后缀
pub fn image_variation_path() -> &'static str {
    "/images/variations"
}

/// Responses API 判定辅助（无状态）。
///
/// 目前设计目标是将判定逻辑外提，避免聚合层硬编码；
/// 聚合层可将自身解析到的布尔开关传入此函数，便于后续在此处扩展
/// 更复杂的策略（例如环境变量、provider 级默认策略等）。
pub fn use_responses_api_from_flag(enabled: bool) -> bool {
    enabled
}

use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use siumai_core::error::LlmError;
use std::collections::HashMap;

/// 构建 OpenAI 风格的 JSON 请求头（Bearer + 组织/项目 + 透传自定义）
pub fn build_openai_json_headers(
    api_key: &str,
    organization: Option<&str>,
    project: Option<&str>,
    http_extra: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();
    // Content-Type
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    // Authorization（先注入，后续 http_extra 可以覆盖）
    let auth = format!("Bearer {}", api_key);
    headers.insert(
        HeaderName::from_static("authorization"),
        HeaderValue::from_str(&auth)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid API key: {e}")))?,
    );
    // OpenAI-Organization / OpenAI-Project
    if let Some(org) = organization {
        headers.insert(
            HeaderName::from_static("openai-organization"),
            HeaderValue::from_str(org).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid OpenAI-Organization: {e}"))
            })?,
        );
    }
    if let Some(prj) = project {
        headers.insert(
            HeaderName::from_static("openai-project"),
            HeaderValue::from_str(prj).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid OpenAI-Project: {e}"))
            })?,
        );
    }
    // Merge http_extra (覆盖同名头)
    for (k, v) in http_extra.iter() {
        let name = HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid header name '{k}': {e}")))?;
        let val = HeaderValue::from_str(v).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header value for '{k}': {e}"))
        })?;
        headers.insert(name, val);
    }
    Ok(headers)
}

// -----------------------------------------------------------------------------
// OpenAI Responses API: SSE 事件名常量（无状态）
// -----------------------------------------------------------------------------
pub const RESPONSES_EVENT_COMPLETED: &str = "response.completed";
pub const RESPONSES_EVENT_OUTPUT_TEXT_DELTA: &str = "response.output_text.delta";
pub const RESPONSES_EVENT_TOOL_CALL_DELTA: &str = "response.tool_call.delta";
pub const RESPONSES_EVENT_FUNCTION_CALL_DELTA: &str = "response.function_call.delta";
pub const RESPONSES_EVENT_FUNCTION_CALL_ARGUMENTS_DELTA: &str =
    "response.function_call_arguments.delta";
pub const RESPONSES_EVENT_OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
pub const RESPONSES_EVENT_USAGE: &str = "response.usage";
pub const RESPONSES_EVENT_ERROR: &str = "response.error";
