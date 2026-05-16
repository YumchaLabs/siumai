//! Anthropic Messages non-streaming JSON response encoder (gateway/proxy).
//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::provider_metadata::anthropic::AnthropicContentPartExt;
use crate::standards::anthropic::server_tools;
use crate::standards::anthropic::utils::{
    raw_container_from_provider_metadata, raw_context_management_from_provider_metadata,
    replay_anthropic_stop_reason,
};
use crate::types::{
    ChatResponse, ContentPart, FinishReason, ToolResultContentPart, ToolResultOutput, Usage,
};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

fn insert_usage_field_if_missing(
    usage: &mut serde_json::Map<String, serde_json::Value>,
    key: &'static str,
    value: serde_json::Value,
) {
    usage.entry(key.to_string()).or_insert(value);
}

fn usage_json(
    u: Option<&Usage>,
    service_tier: Option<&str>,
    raw_usage: Option<&serde_json::Value>,
) -> serde_json::Value {
    let normalized_input = u.map(Usage::normalized_input_tokens).unwrap_or_default();
    let normalized_output = u.map(Usage::normalized_output_tokens).unwrap_or_default();
    let fallback = AnthropicUsage {
        input_tokens: normalized_input.no_cache.unwrap_or(0),
        output_tokens: normalized_output.total.unwrap_or(0),
        service_tier: service_tier.map(ToOwned::to_owned),
    };

    let raw_usage_value = raw_usage
        .cloned()
        .or_else(|| u.and_then(Usage::raw_usage_value));
    let mut usage = raw_usage_value
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    insert_usage_field_if_missing(
        &mut usage,
        "input_tokens",
        serde_json::json!(fallback.input_tokens),
    );
    insert_usage_field_if_missing(
        &mut usage,
        "output_tokens",
        serde_json::json!(fallback.output_tokens),
    );
    if let Some(cache_read_input_tokens) = normalized_input.cache_read {
        insert_usage_field_if_missing(
            &mut usage,
            "cache_read_input_tokens",
            serde_json::json!(cache_read_input_tokens),
        );
    }
    if let Some(cache_creation_input_tokens) = normalized_input.cache_write {
        insert_usage_field_if_missing(
            &mut usage,
            "cache_creation_input_tokens",
            serde_json::json!(cache_creation_input_tokens),
        );
    }

    if let Some(service_tier) = fallback.service_tier {
        insert_usage_field_if_missing(
            &mut usage,
            "service_tier",
            serde_json::Value::String(service_tier),
        );
    }

    serde_json::Value::Object(usage)
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<&'static str>,
    pub stop_sequence: Option<String>,
    pub usage: serde_json::Value,
}

fn text_block(text: String, citations: Option<Vec<serde_json::Value>>) -> serde_json::Value {
    let mut block = serde_json::json!({
        "type": "text",
        "text": text,
    });
    if let Some(citations) = citations
        && !citations.is_empty()
    {
        block["citations"] = serde_json::Value::Array(citations);
    }
    block
}

fn text_part_citations(part: &ContentPart) -> Option<Vec<serde_json::Value>> {
    part.anthropic_text_metadata()
        .and_then(|metadata| metadata.citations)
        .and_then(|citations| serde_json::to_value(citations).ok())
        .and_then(|value| value.as_array().cloned())
        .filter(|citations| !citations.is_empty())
}

fn thinking_block(text: String, signature: Option<String>) -> serde_json::Value {
    let mut block = serde_json::json!({
        "type": "thinking",
        "thinking": text,
    });
    if let Some(signature) = signature {
        block["signature"] = serde_json::Value::String(signature);
    }
    block
}

fn redacted_thinking_block(data: String) -> serde_json::Value {
    serde_json::json!({
        "type": "redacted_thinking",
        "data": data,
    })
}

pub(crate) fn tool_use_block(
    tool_call_id: &str,
    tool_name: &str,
    input: &serde_json::Value,
    caller: Option<serde_json::Value>,
) -> serde_json::Value {
    let mut block = serde_json::json!({
        "type": "tool_use",
        "id": tool_call_id,
        "name": tool_name,
        "input": input,
    });
    if let Some(caller) = caller {
        block["caller"] = caller;
    }
    block
}

fn reserved_json_tool_input_from_text(
    response: &ChatResponse,
    text: &str,
) -> Option<serde_json::Value> {
    if response.raw_finish_reason.as_deref() != Some("tool_use")
        || response.finish_reason.as_ref() != Some(&FinishReason::Stop)
    {
        return None;
    }

    serde_json::from_str(text.trim()).ok()
}

pub(crate) fn provider_tool_use_block(
    tool_call_id: &str,
    tool_name: &str,
    input: &serde_json::Value,
    caller: Option<serde_json::Value>,
    raw_server_tool_name: Option<&str>,
    mcp_server_name: Option<&str>,
) -> serde_json::Value {
    match tool_name {
        "web_search" | "webSearch" | "web_search_preview" | "webSearchPreview" | "web_fetch"
        | "webFetch" | "tool_search" | "toolSearch" | "code_execution" | "codeExecution" => {
            let server_tool_name =
                server_tools::replay_server_tool_name(tool_name, raw_server_tool_name, Some(input));
            let mut block = serde_json::json!({
                "type": "server_tool_use",
                "id": tool_call_id,
                "name": server_tool_name,
                "input": input,
            });
            if let Some(caller) = caller {
                block["caller"] = caller;
            }
            block
        }
        _ => {
            let mut block = serde_json::json!({
                "type": "mcp_tool_use",
                "id": tool_call_id,
                "name": tool_name,
                "input": input,
            });
            if let Some(server_name) = mcp_server_name {
                block["server_name"] = serde_json::Value::String(server_name.to_string());
            }
            block
        }
    }
}

fn tool_result_content_value(output: &ToolResultOutput) -> (serde_json::Value, bool) {
    match output {
        ToolResultOutput::Text { value, .. } => (serde_json::json!(value), false),
        ToolResultOutput::Json { value, .. } => (value.clone(), false),
        ToolResultOutput::ErrorText { value, .. } => (serde_json::json!(value), true),
        ToolResultOutput::ErrorJson { value, .. } => (value.clone(), true),
        ToolResultOutput::ExecutionDenied { reason, .. } => {
            let msg = reason
                .as_ref()
                .map(|r| format!("Execution denied: {r}"))
                .unwrap_or_else(|| "Execution denied".to_string());
            (serde_json::json!(msg), true)
        }
        ToolResultOutput::Content { value, .. } => {
            let content = value
                .iter()
                .map(|part| match part {
                    ToolResultContentPart::Text { text, .. } => {
                        serde_json::json!({"type": "text", "text": text})
                    }
                    ToolResultContentPart::ImageData {
                        data, media_type, ..
                    } => serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        }
                    }),
                    ToolResultContentPart::ImageUrl { url, .. } => serde_json::json!({
                        "type": "text",
                        "text": format!("[Image: {url}]"),
                    }),
                    ToolResultContentPart::ImageFileId { .. }
                    | ToolResultContentPart::ImageFileReference { .. } => {
                        serde_json::json!({"type": "text", "text": "[Image file id attachment]"})
                    }
                    ToolResultContentPart::FileData {
                        data, media_type, ..
                    } if media_type == "application/pdf" => serde_json::json!({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        }
                    }),
                    ToolResultContentPart::FileData { media_type, .. } => {
                        serde_json::json!({
                            "type": "text",
                            "text": format!("[Unsupported file attachment: {media_type}]")
                        })
                    }
                    ToolResultContentPart::FileUrl { url, .. } => serde_json::json!({
                        "type": "document",
                        "source": {
                            "type": "url",
                            "url": url,
                        }
                    }),
                    ToolResultContentPart::FileId { .. }
                    | ToolResultContentPart::FileReference { .. } => {
                        serde_json::json!({"type": "text", "text": "[File id attachment]"})
                    }
                    ToolResultContentPart::Custom { .. } => {
                        serde_json::json!({"type": "text", "text": "[Custom tool content]"})
                    }
                })
                .collect::<Vec<_>>();
            (serde_json::Value::Array(content), false)
        }
    }
}

fn anthropic_web_search_result_content(output: &ToolResultOutput) -> serde_json::Value {
    match output {
        ToolResultOutput::Json { value, .. } => {
            if let Some(arr) = value.as_array() {
                let results = arr
                    .iter()
                    .map(|item| {
                        let Some(obj) = item.as_object() else {
                            return item.clone();
                        };

                        let mut result = serde_json::Map::new();
                        result.insert(
                            "type".to_string(),
                            obj.get("type")
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!("web_search_result")),
                        );
                        if let Some(url) = obj.get("url") {
                            result.insert("url".to_string(), url.clone());
                        }
                        if let Some(title) = obj.get("title") {
                            result.insert("title".to_string(), title.clone());
                        }
                        if let Some(page_age) = obj.get("pageAge").or_else(|| obj.get("page_age")) {
                            result.insert("page_age".to_string(), page_age.clone());
                        }
                        if let Some(encrypted_content) = obj
                            .get("encryptedContent")
                            .or_else(|| obj.get("encrypted_content"))
                        {
                            result
                                .insert("encrypted_content".to_string(), encrypted_content.clone());
                        }
                        for (key, value) in obj {
                            if key != "type"
                                && key != "url"
                                && key != "title"
                                && key != "pageAge"
                                && key != "page_age"
                                && key != "encryptedContent"
                                && key != "encrypted_content"
                            {
                                result.insert(key.clone(), value.clone());
                            }
                        }
                        serde_json::Value::Object(result)
                    })
                    .collect::<Vec<_>>();
                serde_json::Value::Array(results)
            } else if let Some(obj) = value.as_object() {
                if obj
                    .get("type")
                    .and_then(|v| v.as_str())
                    .is_some_and(|t| t == "web_search_tool_result_error")
                {
                    serde_json::json!({
                        "type": "web_search_tool_result_error",
                        "error_code": obj
                            .get("errorCode")
                            .or_else(|| obj.get("error_code"))
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                    })
                } else {
                    value.clone()
                }
            } else {
                value.clone()
            }
        }
        ToolResultOutput::ErrorJson { value, .. } => serde_json::json!({
            "type": "web_search_tool_result_error",
            "error_code": value
                .get("errorCode")
                .or_else(|| value.get("error_code"))
                .cloned()
                .unwrap_or_else(|| value.clone()),
        }),
        ToolResultOutput::ErrorText { value, .. } => serde_json::json!({
            "type": "web_search_tool_result_error",
            "error_code": value,
        }),
        ToolResultOutput::ExecutionDenied { reason, .. } => serde_json::json!({
            "type": "web_search_tool_result_error",
            "error_code": reason.clone().unwrap_or_else(|| "execution_denied".to_string()),
        }),
        other => {
            let (content, _) = tool_result_content_value(other);
            content
        }
    }
}

fn anthropic_tool_search_result_content(output: &ToolResultOutput) -> serde_json::Value {
    match output {
        ToolResultOutput::Json { value, .. } => {
            if let Some(arr) = value.as_array() {
                let tool_references = arr
                    .iter()
                    .map(|item| {
                        let Some(obj) = item.as_object() else {
                            return item.clone();
                        };

                        let mut reference = serde_json::Map::new();
                        reference.insert(
                            "type".to_string(),
                            obj.get("type")
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!("tool_reference")),
                        );
                        if let Some(tool_name) =
                            obj.get("toolName").or_else(|| obj.get("tool_name"))
                        {
                            reference.insert("tool_name".to_string(), tool_name.clone());
                        }
                        for (key, value) in obj {
                            if key != "type" && key != "toolName" && key != "tool_name" {
                                reference.insert(key.clone(), value.clone());
                            }
                        }

                        serde_json::Value::Object(reference)
                    })
                    .collect::<Vec<_>>();

                serde_json::json!({
                    "type": "tool_search_tool_search_result",
                    "tool_references": tool_references,
                })
            } else {
                value.clone()
            }
        }
        ToolResultOutput::ErrorJson { value, .. } => serde_json::json!({
            "type": "tool_search_tool_result_error",
            "error_code": value
                .get("errorCode")
                .or_else(|| value.get("error_code"))
                .cloned()
                .unwrap_or_else(|| value.clone()),
        }),
        ToolResultOutput::ErrorText { value, .. } => serde_json::json!({
            "type": "tool_search_tool_result_error",
            "error_code": value,
        }),
        ToolResultOutput::ExecutionDenied { reason, .. } => serde_json::json!({
            "type": "tool_search_tool_result_error",
            "error_code": reason.clone().unwrap_or_else(|| "execution_denied".to_string()),
        }),
        other => {
            let (content, _) = tool_result_content_value(other);
            content
        }
    }
}

fn anthropic_web_fetch_result_content(output: &ToolResultOutput) -> serde_json::Value {
    match output {
        ToolResultOutput::Json { value, .. } => {
            if let Some(obj) = value.as_object() {
                let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if tpe == "web_fetch_result" {
                    let mut out = serde_json::Map::new();
                    out.insert(
                        "type".to_string(),
                        serde_json::Value::String("web_fetch_result".to_string()),
                    );
                    if let Some(url) = obj.get("url") {
                        out.insert("url".to_string(), url.clone());
                    }
                    if let Some(retrieved_at) =
                        obj.get("retrievedAt").or_else(|| obj.get("retrieved_at"))
                    {
                        out.insert("retrieved_at".to_string(), retrieved_at.clone());
                    }
                    if let Some(content) = obj.get("content").and_then(|v| v.as_object()) {
                        let mut out_content = serde_json::Map::new();
                        if let Some(v) = content.get("type") {
                            out_content.insert("type".to_string(), v.clone());
                        }
                        if let Some(v) = content.get("title") {
                            out_content.insert("title".to_string(), v.clone());
                        }
                        if let Some(v) = content.get("citations") {
                            out_content.insert("citations".to_string(), v.clone());
                        }
                        if let Some(source) = content.get("source").and_then(|v| v.as_object()) {
                            let mut out_source = serde_json::Map::new();
                            if let Some(v) = source.get("type") {
                                out_source.insert("type".to_string(), v.clone());
                            }
                            if let Some(v) =
                                source.get("mediaType").or_else(|| source.get("media_type"))
                            {
                                out_source.insert("media_type".to_string(), v.clone());
                            }
                            if let Some(v) = source.get("data") {
                                out_source.insert("data".to_string(), v.clone());
                            }
                            for (key, value) in source {
                                if key != "type"
                                    && key != "mediaType"
                                    && key != "media_type"
                                    && key != "data"
                                {
                                    out_source.insert(key.clone(), value.clone());
                                }
                            }
                            out_content.insert(
                                "source".to_string(),
                                serde_json::Value::Object(out_source),
                            );
                        }
                        for (key, value) in content {
                            if key != "type"
                                && key != "title"
                                && key != "citations"
                                && key != "source"
                            {
                                out_content.insert(key.clone(), value.clone());
                            }
                        }
                        out.insert(
                            "content".to_string(),
                            serde_json::Value::Object(out_content),
                        );
                    }
                    serde_json::Value::Object(out)
                } else if tpe == "web_fetch_tool_result_error" {
                    serde_json::json!({
                        "type": "web_fetch_tool_result_error",
                        "error_code": obj
                            .get("errorCode")
                            .or_else(|| obj.get("error_code"))
                            .cloned()
                            .unwrap_or_else(|| value.clone()),
                    })
                } else {
                    value.clone()
                }
            } else {
                value.clone()
            }
        }
        ToolResultOutput::ErrorJson { value, .. } => serde_json::json!({
            "type": "web_fetch_tool_result_error",
            "error_code": value
                .get("errorCode")
                .or_else(|| value.get("error_code"))
                .cloned()
                .unwrap_or_else(|| value.clone()),
        }),
        ToolResultOutput::ErrorText { value, .. } => serde_json::json!({
            "type": "web_fetch_tool_result_error",
            "error_code": value,
        }),
        ToolResultOutput::ExecutionDenied { reason, .. } => serde_json::json!({
            "type": "web_fetch_tool_result_error",
            "error_code": reason.clone().unwrap_or_else(|| "execution_denied".to_string()),
        }),
        other => {
            let (content, _) = tool_result_content_value(other);
            content
        }
    }
}

pub(crate) fn provider_tool_result_block(
    tool_call_id: &str,
    tool_name: &str,
    output: &ToolResultOutput,
    raw_server_tool_name: Option<&str>,
) -> serde_json::Value {
    let (content, is_error) = tool_result_content_value(output);
    let block_type =
        server_tools::replay_server_tool_result_block_type(tool_name, raw_server_tool_name, output);

    match tool_name {
        "web_search" | "webSearch" | "web_search_preview" | "webSearchPreview" => {
            serde_json::json!({
                "type": block_type,
                "tool_use_id": tool_call_id,
                "content": anthropic_web_search_result_content(output),
            })
        }
        "web_fetch" | "webFetch" => serde_json::json!({
            "type": block_type,
            "tool_use_id": tool_call_id,
            "content": anthropic_web_fetch_result_content(output),
        }),
        "tool_search" | "toolSearch" => serde_json::json!({
            "type": block_type,
            "tool_use_id": tool_call_id,
            "content": anthropic_tool_search_result_content(output),
        }),
        "code_execution" | "codeExecution" => serde_json::json!({
            "type": block_type,
            "tool_use_id": tool_call_id,
            "content": content,
        }),
        _ => serde_json::json!({
            "type": "mcp_tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
            "is_error": is_error,
        }),
    }
}

#[derive(Debug, Clone)]
pub struct AnthropicMessagesJsonResponseConverter;

impl AnthropicMessagesJsonResponseConverter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AnthropicMessagesJsonResponseConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonResponseConverter for AnthropicMessagesJsonResponseConverter {
    fn serialize_response(
        &self,
        response: &ChatResponse,
        out: &mut Vec<u8>,
        opts: JsonEncodeOptions,
    ) -> Result<(), LlmError> {
        let id = response.id.clone().unwrap_or_else(|| "siumai".to_string());
        let model = response.model.clone().unwrap_or_default();

        let mut content = Vec::new();
        let raw_anthropic_meta = response
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("anthropic"));
        let stop_sequence = raw_anthropic_meta
            .and_then(|meta| meta.get("stopSequence"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned);
        let container = raw_anthropic_meta
            .and_then(|meta| meta.get("container"))
            .and_then(raw_container_from_provider_metadata);
        let context_management = raw_anthropic_meta
            .and_then(|meta| meta.get("contextManagement"))
            .and_then(raw_context_management_from_provider_metadata);
        let raw_usage = raw_anthropic_meta.and_then(|meta| meta.get("usage"));
        let has_text_part_citations = match &response.content {
            crate::types::MessageContent::MultiModal(parts) => {
                parts.iter().any(|part| text_part_citations(part).is_some())
            }
            _ => false,
        };
        let mut projected_citations_fallback = if has_text_part_citations {
            None
        } else {
            raw_anthropic_meta
                .and_then(|meta| meta.get("citations"))
                .and_then(|value| value.as_array())
                .map(|blocks| {
                    blocks
                        .iter()
                        .filter_map(|block| {
                            block
                                .get("citations")
                                .and_then(|value| value.as_array())
                                .cloned()
                        })
                        .flatten()
                        .collect::<Vec<_>>()
                })
                .filter(|citations| !citations.is_empty())
        };

        match &response.content {
            crate::types::MessageContent::Text(text) => {
                if let Some(input) = reserved_json_tool_input_from_text(response, text) {
                    content.push(tool_use_block("toolu_siumai_json", "json", &input, None));
                } else if !text.trim().is_empty() {
                    content.push(text_block(
                        text.clone(),
                        projected_citations_fallback.take(),
                    ));
                }
            }
            crate::types::MessageContent::MultiModal(parts) => {
                let mut raw_server_tool_names_by_id: HashMap<String, String> = HashMap::new();

                for part in parts {
                    match part {
                        ContentPart::Text { text, .. } => {
                            content.push(text_block(
                                text.clone(),
                                text_part_citations(part)
                                    .or_else(|| projected_citations_fallback.take()),
                            ));
                        }
                        ContentPart::Reasoning { text, .. } => {
                            let reasoning_meta = part.anthropic_reasoning_metadata();
                            let redacted_data = reasoning_meta
                                .as_ref()
                                .and_then(|metadata| metadata.redacted_data.clone());
                            let signature = reasoning_meta.and_then(|metadata| metadata.signature);

                            if let Some(redacted_data) = redacted_data {
                                content.push(redacted_thinking_block(redacted_data));
                            } else if !text.is_empty() || signature.is_some() {
                                content.push(thinking_block(text.clone(), signature));
                            }
                        }
                        ContentPart::ToolCall {
                            tool_call_id,
                            tool_name,
                            arguments,
                            provider_executed,
                            ..
                        } => {
                            let tool_call_meta = part.anthropic_tool_call_metadata();
                            let caller = tool_call_meta
                                .as_ref()
                                .and_then(|meta| meta.caller.clone())
                                .and_then(|caller| serde_json::to_value(caller).ok());
                            let raw_server_tool_name = tool_call_meta
                                .as_ref()
                                .and_then(|meta| meta.server_tool_name.as_deref());
                            let mcp_server_name = tool_call_meta
                                .as_ref()
                                .and_then(|meta| meta.server_name.as_deref());

                            if *provider_executed == Some(true) {
                                let replayed_server_tool_name =
                                    server_tools::replay_server_tool_name(
                                        tool_name,
                                        raw_server_tool_name,
                                        Some(arguments),
                                    );
                                if replayed_server_tool_name != *tool_name {
                                    raw_server_tool_names_by_id
                                        .insert(tool_call_id.clone(), replayed_server_tool_name);
                                }
                                content.push(provider_tool_use_block(
                                    tool_call_id,
                                    tool_name,
                                    arguments,
                                    caller,
                                    raw_server_tool_name,
                                    mcp_server_name,
                                ));
                            } else {
                                content.push(tool_use_block(
                                    tool_call_id,
                                    tool_name,
                                    arguments,
                                    caller,
                                ));
                            }
                        }
                        ContentPart::ToolResult {
                            tool_call_id,
                            tool_name,
                            output,
                            provider_executed,
                            ..
                        } if *provider_executed == Some(true) => {
                            let raw_server_tool_name = raw_server_tool_names_by_id
                                .get(tool_call_id)
                                .map(String::as_str);
                            content.push(provider_tool_result_block(
                                tool_call_id,
                                tool_name,
                                output,
                                raw_server_tool_name,
                            ));
                        }
                        _ => {}
                    }
                }
            }
            #[cfg(feature = "structured-messages")]
            crate::types::MessageContent::Json(value) => {
                if response.raw_finish_reason.as_deref() == Some("tool_use")
                    && response.finish_reason.as_ref() == Some(&FinishReason::Stop)
                {
                    content.push(tool_use_block("toolu_siumai_json", "json", value, None));
                } else {
                    let text = serde_json::to_string(value).unwrap_or_default();
                    if !text.trim().is_empty() {
                        content.push(text_block(text, projected_citations_fallback.take()));
                    }
                }
            }
        }

        let body = AnthropicMessageResponse {
            id,
            kind: "message",
            role: "assistant",
            model,
            content,
            container,
            context_management,
            stop_reason: replay_anthropic_stop_reason(
                response.raw_finish_reason.as_deref(),
                response.finish_reason.as_ref(),
            ),
            stop_sequence,
            usage: usage_json(
                response.usage.as_ref(),
                response.service_tier.as_deref(),
                raw_usage,
            ),
        };

        if opts.pretty {
            serde_json::to_writer_pretty(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize Anthropic Messages JSON response: {e}"
                ))
            })?;
        } else {
            serde_json::to_writer(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize Anthropic Messages JSON response: {e}"
                ))
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn anthropic_json_response_encoder_source_does_not_read_request_provider_options() {
        let source = include_str!("json_response.rs");
        let production_source = source
            .split("#[cfg(test)]")
            .next()
            .expect("production source");

        for forbidden in ["provider_options", ".provider_options", "providerOptions"] {
            assert!(
                !production_source.contains(forbidden),
                "Anthropic JSON response encoder must not read request-side provider options fragment `{forbidden}`"
            );
        }
    }

    #[test]
    fn anthropic_encoder_serializes_thinking_redacted_stop_sequence_and_caller() {
        let tool_call = ContentPart::ToolCall {
            tool_call_id: "toolu_1".to_string(),
            tool_name: "weather".to_string(),
            arguments: serde_json::json!({ "city": "Tokyo" }),
            provider_executed: None,
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "caller": {
                        "type": "code_execution_20250825",
                        "tool_id": "srvtoolu_1"
                    }
                }),
            )])),
        };

        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("Visible answer"),
            ContentPart::Reasoning {
                text: "Need to verify assumptions.".to_string(),
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "signature": "sig_1"
                    }),
                )])),
            },
            tool_call,
            ContentPart::Reasoning {
                text: String::new(),
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "redactedData": "redacted_123"
                    }),
                )])),
            },
        ]));
        response.id = Some("msg_1".to_string());
        response.model = Some("claude-sonnet-4-5".to_string());
        response.finish_reason = Some(crate::types::FinishReason::StopSequence);
        response.service_tier = Some("priority".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "anthropic".to_string(),
            serde_json::json!({
                "stopSequence": "</tool>"
            }),
        )]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["stop_reason"], serde_json::json!("stop_sequence"));
        assert_eq!(value["stop_sequence"], serde_json::json!("</tool>"));
        assert_eq!(
            value["usage"]["service_tier"],
            serde_json::json!("priority")
        );
        assert_eq!(value["content"][1]["type"], serde_json::json!("thinking"));
        assert_eq!(value["content"][1]["signature"], serde_json::json!("sig_1"));
        assert_eq!(
            value["content"][2]["caller"]["tool_id"],
            serde_json::json!("srvtoolu_1")
        );
        assert_eq!(
            value["content"][3]["type"],
            serde_json::json!("redacted_thinking")
        );
        assert_eq!(
            value["content"][3]["data"],
            serde_json::json!("redacted_123")
        );
    }

    #[test]
    fn anthropic_encoder_replays_raw_stop_reason_before_unified_finish_reason() {
        let mut response = ChatResponse::new(crate::types::MessageContent::Text(
            "{\"value\":\"ok\"}".to_string(),
        ));
        response.id = Some("msg_json".to_string());
        response.model = Some("claude-sonnet-4-5".to_string());
        response.finish_reason = Some(crate::types::FinishReason::Stop);
        response.raw_finish_reason = Some("tool_use".to_string());

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["stop_reason"], serde_json::json!("tool_use"));
        assert_eq!(value["content"][0]["type"], serde_json::json!("tool_use"));
        assert_eq!(value["content"][0]["name"], serde_json::json!("json"));
        assert_eq!(
            value["content"][0]["input"],
            serde_json::json!({ "value": "ok" })
        );
    }

    #[test]
    fn anthropic_encoder_serializes_provider_hosted_tool_blocks() {
        let response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("searching"),
            ContentPart::ToolCall {
                tool_call_id: "srvtoolu_1".to_string(),
                tool_name: "web_search".to_string(),
                arguments: serde_json::json!({ "query": "rust bridge" }),
                provider_executed: Some(true),
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
            ContentPart::ToolResult {
                tool_call_id: "srvtoolu_1".to_string(),
                tool_name: "web_search".to_string(),
                output: ToolResultOutput::json(serde_json::json!([
                    { "type": "web_search_result", "url": "https://www.rust-lang.org" }
                ])),
                input: None,
                provider_executed: Some(true),
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
            ContentPart::ToolCall {
                tool_call_id: "mcptoolu_1".to_string(),
                tool_name: "echo".to_string(),
                arguments: serde_json::json!({ "message": "hello" }),
                provider_executed: Some(true),
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "serverName": "echo-prod"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "mcptoolu_1".to_string(),
                tool_name: "echo".to_string(),
                output: ToolResultOutput::content(vec![ToolResultContentPart::Text {
                    text: "Tool echo: hello".to_string(),
                    provider_options: crate::types::ProviderOptionsMap::default(),
                }]),
                input: None,
                provider_executed: Some(true),
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        ]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["content"][1]["type"],
            serde_json::json!("server_tool_use")
        );
        assert_eq!(value["content"][1]["name"], serde_json::json!("web_search"));
        assert_eq!(
            value["content"][2]["type"],
            serde_json::json!("web_search_tool_result")
        );
        assert_eq!(
            value["content"][3]["type"],
            serde_json::json!("mcp_tool_use")
        );
        assert_eq!(
            value["content"][3]["server_name"],
            serde_json::json!("echo-prod")
        );
        assert_eq!(
            value["content"][4]["type"],
            serde_json::json!("mcp_tool_result")
        );
        assert!(value["content"][4].get("server_name").is_none());
    }

    #[test]
    fn anthropic_encoder_replays_tool_search_server_tool_name_and_result_shape() {
        let response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolCall {
                tool_call_id: "srvtoolu_2".to_string(),
                tool_name: "tool_search".to_string(),
                arguments: serde_json::json!({ "pattern": "weather", "limit": 2 }),
                provider_executed: Some(true),
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "caller": { "type": "direct" },
                        "serverToolName": "tool_search_tool_regex"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "srvtoolu_2".to_string(),
                tool_name: "tool_search".to_string(),
                output: ToolResultOutput::json(serde_json::json!([
                    { "type": "tool_reference", "toolName": "get_weather" }
                ])),
                input: None,
                provider_executed: Some(true),
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        ]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["content"][0]["name"],
            serde_json::json!("tool_search_tool_regex")
        );
        assert_eq!(
            value["content"][0]["caller"]["type"],
            serde_json::json!("direct")
        );
        assert_eq!(
            value["content"][1]["content"]["type"],
            serde_json::json!("tool_search_tool_search_result")
        );
        assert_eq!(
            value["content"][1]["content"]["tool_references"][0]["tool_name"],
            serde_json::json!("get_weather")
        );
    }

    #[test]
    fn anthropic_encoder_replays_web_search_result_shape() {
        let response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolCall {
                tool_call_id: "srvtoolu_ws".to_string(),
                tool_name: "web_search".to_string(),
                arguments: serde_json::json!({ "query": "latest tech news" }),
                provider_executed: Some(true),
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
            ContentPart::ToolResult {
                tool_call_id: "srvtoolu_ws".to_string(),
                tool_name: "web_search".to_string(),
                output: ToolResultOutput::json(serde_json::json!([
                    {
                        "type": "web_search_result",
                        "url": "https://example.com",
                        "title": "Example",
                        "pageAge": "2 days ago",
                        "encryptedContent": "secret"
                    }
                ])),
                input: None,
                provider_executed: Some(true),
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        ]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["content"][1]["content"][0]["page_age"],
            serde_json::json!("2 days ago")
        );
        assert_eq!(
            value["content"][1]["content"][0]["encrypted_content"],
            serde_json::json!("secret")
        );
        assert!(value["content"][1]["content"][0].get("pageAge").is_none());
        assert!(
            value["content"][1]["content"][0]
                .get("encryptedContent")
                .is_none()
        );
    }

    #[test]
    fn anthropic_encoder_replays_code_execution_server_tool_name_and_result_block_type() {
        let response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolCall {
                tool_call_id: "srvtoolu_code_1".to_string(),
                tool_name: "code_execution".to_string(),
                arguments: serde_json::json!({
                    "type": "text_editor_code_execution"
                }),
                provider_executed: Some(true),
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "serverToolName": "text_editor_code_execution"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "srvtoolu_code_1".to_string(),
                tool_name: "code_execution".to_string(),
                output: ToolResultOutput::json(serde_json::json!({
                    "type": "text_editor_code_execution_create_result",
                    "is_file_update": false
                })),
                input: None,
                provider_executed: Some(true),
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
            ContentPart::ToolCall {
                tool_call_id: "srvtoolu_code_2".to_string(),
                tool_name: "code_execution".to_string(),
                arguments: serde_json::json!({
                    "type": "bash_code_execution"
                }),
                provider_executed: Some(true),
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "serverToolName": "bash_code_execution"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "srvtoolu_code_2".to_string(),
                tool_name: "code_execution".to_string(),
                output: ToolResultOutput::json(serde_json::json!({
                    "type": "bash_code_execution_result",
                    "stdout": "",
                    "stderr": "",
                    "return_code": 0
                })),
                input: None,
                provider_executed: Some(true),
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        ]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["content"][0]["name"],
            serde_json::json!("text_editor_code_execution")
        );
        assert_eq!(
            value["content"][1]["type"],
            serde_json::json!("text_editor_code_execution_tool_result")
        );
        assert_eq!(
            value["content"][2]["name"],
            serde_json::json!("bash_code_execution")
        );
        assert_eq!(
            value["content"][3]["type"],
            serde_json::json!("bash_code_execution_tool_result")
        );
    }

    #[test]
    fn anthropic_encoder_replays_web_fetch_result_shape() {
        let response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolCall {
                tool_call_id: "srvtoolu_3".to_string(),
                tool_name: "web_fetch".to_string(),
                arguments: serde_json::json!({ "url": "https://example.com" }),
                provider_executed: Some(true),
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
            ContentPart::ToolResult {
                tool_call_id: "srvtoolu_3".to_string(),
                tool_name: "web_fetch".to_string(),
                output: ToolResultOutput::json(serde_json::json!({
                    "type": "web_fetch_result",
                    "url": "https://example.com",
                    "retrievedAt": "2025-01-01T00:00:00Z",
                    "content": {
                        "type": "document",
                        "title": "Example",
                        "source": {
                            "type": "text",
                            "mediaType": "text/plain",
                            "data": "hello"
                        }
                    }
                })),
                input: None,
                provider_executed: Some(true),
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        ]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["content"][1]["content"]["type"],
            serde_json::json!("web_fetch_result")
        );
        assert_eq!(
            value["content"][1]["content"]["retrieved_at"],
            serde_json::json!("2025-01-01T00:00:00Z")
        );
        assert_eq!(
            value["content"][1]["content"]["content"]["source"]["media_type"],
            serde_json::json!("text/plain")
        );
    }

    #[test]
    fn anthropic_encoder_replays_raw_usage_and_citations_metadata() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("grounded answer"),
        ]));
        response.usage = Some(
            Usage::builder()
                .prompt_tokens(7)
                .completion_tokens(3)
                .total_tokens(10)
                .build(),
        );
        response.service_tier = Some("standard".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "anthropic".to_string(),
            serde_json::json!({
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 0
                    },
                    "service_tier": "standard",
                    "server_tool_use": {
                        "web_search_requests": 2,
                        "web_fetch_requests": 0
                    }
                },
                "citations": [
                    {
                        "content_block_index": 4,
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "url": "https://example.com",
                                "title": "Example"
                            }
                        ]
                    }
                ]
            }),
        )]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["usage"]["server_tool_use"]["web_search_requests"],
            serde_json::json!(2)
        );
        assert_eq!(
            value["usage"]["cache_creation"]["ephemeral_5m_input_tokens"],
            serde_json::json!(0)
        );
        assert_eq!(
            value["content"][0]["citations"][0]["url"],
            serde_json::json!("https://example.com")
        );
    }

    #[test]
    fn anthropic_encoder_preserves_raw_usage_token_fields_before_normalized_fallback() {
        let raw_usage = serde_json::json!({
            "input_tokens": 45000,
            "output_tokens": 1234,
            "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 500,
            "service_tier": "standard",
            "iterations": [
                {
                    "type": "compaction",
                    "input_tokens": 180000,
                    "output_tokens": 3500
                },
                {
                    "type": "message",
                    "input_tokens": 23000,
                    "output_tokens": 1000
                }
            ]
        });
        let mut response =
            ChatResponse::new(crate::types::MessageContent::Text("done".to_string()));
        response.usage = Some(
            Usage::builder()
                .prompt_tokens(203000)
                .completion_tokens(4500)
                .total_tokens(207500)
                .with_input_total_tokens(204500)
                .with_input_no_cache_tokens(203000)
                .with_input_cache_read_tokens(500)
                .with_input_cache_write_tokens(1000)
                .with_output_total_tokens(4500)
                .with_raw_usage_value(raw_usage)
                .build(),
        );

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["usage"]["input_tokens"], serde_json::json!(45000));
        assert_eq!(value["usage"]["output_tokens"], serde_json::json!(1234));
        assert_eq!(
            value["usage"]["cache_creation_input_tokens"],
            serde_json::json!(1000)
        );
        assert_eq!(
            value["usage"]["cache_read_input_tokens"],
            serde_json::json!(500)
        );
        assert_eq!(
            value["usage"]["service_tier"],
            serde_json::json!("standard")
        );
        assert_eq!(
            value["usage"]["iterations"][0]["input_tokens"],
            serde_json::json!(180000)
        );
    }

    #[test]
    fn anthropic_encoder_prefers_text_part_citations_for_exact_block_replay() {
        let response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("Intro"),
            ContentPart::Text {
                text: "Grounded fact".to_string(),
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "url": "https://example.com/fact",
                                "title": "Fact"
                            }
                        ]
                    }),
                )])),
            },
            ContentPart::text("Outro"),
        ]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert!(value["content"][0].get("citations").is_none());
        assert_eq!(
            value["content"][1]["citations"][0]["url"],
            serde_json::json!("https://example.com/fact")
        );
        assert!(value["content"][2].get("citations").is_none());
    }

    #[test]
    fn anthropic_encoder_replays_container_metadata() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("done"),
        ]));
        response.provider_metadata = Some(HashMap::from([(
            "anthropic".to_string(),
            serde_json::json!({
                "container": {
                    "id": "container_123",
                    "expiresAt": "2025-10-14T10:28:40.590791Z",
                    "skills": [
                        {
                            "type": "anthropic",
                            "skillId": "pptx",
                            "version": "latest"
                        }
                    ]
                }
            }),
        )]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["container"]["id"], serde_json::json!("container_123"));
        assert_eq!(
            value["container"]["expires_at"],
            serde_json::json!("2025-10-14T10:28:40.590791Z")
        );
        assert_eq!(
            value["container"]["skills"][0]["skill_id"],
            serde_json::json!("pptx")
        );
    }

    #[test]
    fn anthropic_encoder_replays_context_management_metadata() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::tool_call(
                "toolu_1",
                "memory",
                serde_json::json!({ "command": "view", "path": "/memories" }),
                None,
            ),
        ]));
        response.provider_metadata = Some(HashMap::from([(
            "anthropic".to_string(),
            serde_json::json!({
                "contextManagement": {
                    "appliedEdits": [
                        {
                            "type": "clear_tool_uses_20250919",
                            "clearedToolUses": 3,
                            "clearedInputTokens": 1000
                        }
                    ]
                }
            }),
        )]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["context_management"]["applied_edits"][0]["type"],
            serde_json::json!("clear_tool_uses_20250919")
        );
        assert_eq!(
            value["context_management"]["applied_edits"][0]["cleared_tool_uses"],
            serde_json::json!(3)
        );
        assert_eq!(
            value["context_management"]["applied_edits"][0]["cleared_input_tokens"],
            serde_json::json!(1000)
        );
    }

    #[test]
    fn anthropic_encoder_serializes_normalized_usage_breakdown_without_raw_usage() {
        let mut response =
            ChatResponse::new(crate::types::MessageContent::Text("hello".to_string()));
        response.usage = Some(
            Usage::builder()
                .prompt_tokens(12)
                .completion_tokens(7)
                .total_tokens(19)
                .with_input_total_tokens(12)
                .with_input_no_cache_tokens(5)
                .with_input_cache_read_tokens(4)
                .with_input_cache_write_tokens(3)
                .with_output_total_tokens(7)
                .with_output_text_tokens(7)
                .build(),
        );

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["usage"]["input_tokens"], serde_json::json!(5));
        assert_eq!(
            value["usage"]["cache_read_input_tokens"],
            serde_json::json!(4)
        );
        assert_eq!(
            value["usage"]["cache_creation_input_tokens"],
            serde_json::json!(3)
        );
        assert_eq!(value["usage"]["output_tokens"], serde_json::json!(7));
    }
}
