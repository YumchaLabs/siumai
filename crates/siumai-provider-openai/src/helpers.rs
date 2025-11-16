//! OpenAI provider helpers (stateless).
//!
//! Only contains stateless helpers for routing/path suffixes. The aggregator
//! calls into this module when `provider-openai-external` is enabled to keep
//! behavior consistent and easy to migrate.

/// Chat path suffix: choose between Chat Completions and Responses API.
pub fn chat_path(use_responses_api: bool) -> &'static str {
    if use_responses_api {
        "/responses"
    } else {
        "/chat/completions"
    }
}

/// Embedding path suffix.
pub fn embedding_path() -> &'static str {
    "/embeddings"
}

/// Image generation path suffix.
pub fn image_generation_path() -> &'static str {
    "/images/generations"
}

/// Image edit path suffix.
pub fn image_edit_path() -> &'static str {
    "/images/edits"
}

/// Image variation path suffix.
pub fn image_variation_path() -> &'static str {
    "/images/variations"
}

/// Helper for deciding whether to use the Responses API (stateless).
///
/// The goal is to keep the decision logic outside the aggregator so the
/// aggregator can simply pass a boolean flag and this helper can later evolve
/// to include environment-based or provider-level defaults.
pub fn use_responses_api_from_flag(enabled: bool) -> bool {
    enabled
}

use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use siumai_core::error::LlmError;
use std::collections::HashMap;

/// Build OpenAI-style JSON headers (Bearer auth + org/project + extra custom headers).
pub fn build_openai_json_headers(
    api_key: &str,
    organization: Option<&str>,
    project: Option<&str>,
    http_extra: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();
    // Content-Type
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    // Authorization (insert first; allow http_extra to override).
    let auth = format!("Bearer {}", api_key);
    headers.insert(
        HeaderName::from_static("authorization"),
        HeaderValue::from_str(&auth)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid API key: {e}")))?,
    );
    // OpenAI-Organization / OpenAI-Project.
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
    // Merge http_extra (overwriting existing headers on key collision).
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
// OpenAI Responses API: SSE event name constants (stateless).
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
