//! Anthropic Messages non-streaming JSON response encoder (gateway/proxy).
//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::provider_metadata::anthropic::AnthropicContentPartExt;
use crate::types::{
    ChatResponse, ContentPart, FinishReason, MediaSource, ToolResultContentPart, ToolResultOutput,
    Usage,
};
use serde::Serialize;

fn anthropic_stop_reason(reason: Option<&FinishReason>) -> Option<&'static str> {
    match reason? {
        FinishReason::Stop | FinishReason::StopSequence => Some("end_turn"),
        FinishReason::Length => Some("max_tokens"),
        FinishReason::ToolCalls => Some("tool_use"),
        // Best-effort fallbacks.
        FinishReason::ContentFilter => Some("stop_sequence"),
        FinishReason::Error => Some("end_turn"),
        FinishReason::Unknown => None,
        FinishReason::Other(_) => None,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

fn usage_json(
    u: Option<&Usage>,
    service_tier: Option<&str>,
    raw_usage: Option<&serde_json::Value>,
) -> serde_json::Value {
    let fallback = match u {
        Some(u) => AnthropicUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            service_tier: service_tier.map(ToOwned::to_owned),
        },
        None => AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
            service_tier: service_tier.map(ToOwned::to_owned),
        },
    };

    let mut usage = raw_usage
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    usage.insert(
        "input_tokens".to_string(),
        serde_json::json!(fallback.input_tokens),
    );
    usage.insert(
        "output_tokens".to_string(),
        serde_json::json!(fallback.output_tokens),
    );

    match fallback.service_tier {
        Some(service_tier) => {
            usage.insert(
                "service_tier".to_string(),
                serde_json::Value::String(service_tier),
            );
        }
        None => {
            usage.remove("service_tier");
        }
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

fn tool_use_block(
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

fn provider_tool_use_block(
    tool_call_id: &str,
    tool_name: &str,
    input: &serde_json::Value,
    caller: Option<serde_json::Value>,
    raw_server_tool_name: Option<&str>,
    mcp_server_name: Option<&str>,
) -> serde_json::Value {
    match tool_name {
        "web_search" | "web_fetch" | "tool_search" | "code_execution" => {
            let server_tool_name = if tool_name == "tool_search" {
                raw_server_tool_name.unwrap_or(tool_name)
            } else {
                tool_name
            };
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
        ToolResultOutput::Text { value } => (serde_json::json!(value), false),
        ToolResultOutput::Json { value } => (value.clone(), false),
        ToolResultOutput::ErrorText { value } => (serde_json::json!(value), true),
        ToolResultOutput::ErrorJson { value } => (value.clone(), true),
        ToolResultOutput::ExecutionDenied { reason } => {
            let msg = reason
                .as_ref()
                .map(|r| format!("Execution denied: {r}"))
                .unwrap_or_else(|| "Execution denied".to_string());
            (serde_json::json!(msg), true)
        }
        ToolResultOutput::Content { value } => {
            let content = value
                .iter()
                .map(|part| match part {
                    ToolResultContentPart::Text { text } => {
                        serde_json::json!({"type": "text", "text": text})
                    }
                    ToolResultContentPart::Image { source, .. } => match source {
                        MediaSource::Url { url } => serde_json::json!({
                            "type": "text",
                            "text": format!("[Image: {url}]"),
                        }),
                        MediaSource::Base64 { data } => serde_json::json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": data,
                            }
                        }),
                        MediaSource::Binary { data } => serde_json::json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64::Engine::encode(
                                    &base64::engine::general_purpose::STANDARD,
                                    data
                                ),
                            }
                        }),
                    },
                    ToolResultContentPart::File { .. } => {
                        serde_json::json!({"type": "text", "text": "[File attachment]"})
                    }
                })
                .collect::<Vec<_>>();
            (serde_json::Value::Array(content), false)
        }
    }
}

fn anthropic_tool_search_result_content(output: &ToolResultOutput) -> serde_json::Value {
    match output {
        ToolResultOutput::Json { value } => {
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
        ToolResultOutput::ErrorJson { value } => serde_json::json!({
            "type": "tool_search_tool_result_error",
            "error_code": value
                .get("errorCode")
                .or_else(|| value.get("error_code"))
                .cloned()
                .unwrap_or_else(|| value.clone()),
        }),
        ToolResultOutput::ErrorText { value } => serde_json::json!({
            "type": "tool_search_tool_result_error",
            "error_code": value,
        }),
        ToolResultOutput::ExecutionDenied { reason } => serde_json::json!({
            "type": "tool_search_tool_result_error",
            "error_code": reason.clone().unwrap_or_else(|| "execution_denied".to_string()),
        }),
        other => {
            let (content, _) = tool_result_content_value(other);
            content
        }
    }
}

fn provider_tool_result_block(
    tool_call_id: &str,
    tool_name: &str,
    output: &ToolResultOutput,
) -> serde_json::Value {
    let (content, is_error) = tool_result_content_value(output);

    match tool_name {
        "web_search" => serde_json::json!({
            "type": "web_search_tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }),
        "web_fetch" => serde_json::json!({
            "type": "web_fetch_tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }),
        "tool_search" => serde_json::json!({
            "type": "tool_search_tool_result",
            "tool_use_id": tool_call_id,
            "content": anthropic_tool_search_result_content(output),
        }),
        "code_execution" => serde_json::json!({
            "type": "code_execution_tool_result",
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

fn raw_container_from_provider_metadata(value: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = value.as_object()?;
    let mut out = serde_json::Map::new();

    if let Some(id) = obj.get("id") {
        out.insert("id".to_string(), id.clone());
    }
    if let Some(expires_at) = obj.get("expiresAt") {
        out.insert("expires_at".to_string(), expires_at.clone());
    }
    if let Some(skills) = obj.get("skills").and_then(|value| value.as_array()) {
        let mut out_skills = Vec::with_capacity(skills.len());
        for item in skills {
            let Some(skill) = item.as_object() else {
                out_skills.push(item.clone());
                continue;
            };

            let mut out_skill = serde_json::Map::new();
            if let Some(skill_type) = skill.get("type") {
                out_skill.insert("type".to_string(), skill_type.clone());
            }
            if let Some(skill_id) = skill.get("skillId") {
                out_skill.insert("skill_id".to_string(), skill_id.clone());
            }
            if let Some(version) = skill.get("version") {
                out_skill.insert("version".to_string(), version.clone());
            }
            out_skills.push(serde_json::Value::Object(out_skill));
        }
        out.insert("skills".to_string(), serde_json::Value::Array(out_skills));
    }

    Some(serde_json::Value::Object(out))
}

fn raw_context_management_from_provider_metadata(
    value: &serde_json::Value,
) -> Option<serde_json::Value> {
    let obj = value.as_object()?;
    let edits = obj.get("appliedEdits")?.as_array()?;
    let mut out_edits = Vec::with_capacity(edits.len());

    for edit in edits {
        let Some(edit_obj) = edit.as_object() else {
            continue;
        };
        let Some(edit_type) = edit_obj.get("type").and_then(|value| value.as_str()) else {
            continue;
        };

        match edit_type {
            "clear_tool_uses_20250919" => {
                let mut out = serde_json::Map::new();
                out.insert(
                    "type".to_string(),
                    serde_json::Value::String(edit_type.to_string()),
                );
                if let Some(value) = edit_obj.get("clearedToolUses") {
                    out.insert("cleared_tool_uses".to_string(), value.clone());
                }
                if let Some(value) = edit_obj.get("clearedInputTokens") {
                    out.insert("cleared_input_tokens".to_string(), value.clone());
                }
                out_edits.push(serde_json::Value::Object(out));
            }
            "clear_thinking_20251015" => {
                let mut out = serde_json::Map::new();
                out.insert(
                    "type".to_string(),
                    serde_json::Value::String(edit_type.to_string()),
                );
                if let Some(value) = edit_obj.get("clearedThinkingTurns") {
                    out.insert("cleared_thinking_turns".to_string(), value.clone());
                }
                if let Some(value) = edit_obj.get("clearedInputTokens") {
                    out.insert("cleared_input_tokens".to_string(), value.clone());
                }
                out_edits.push(serde_json::Value::Object(out));
            }
            _ => {}
        }
    }

    Some(serde_json::json!({
        "applied_edits": out_edits
    }))
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
        let thinking_signature = raw_anthropic_meta
            .and_then(|meta| meta.get("thinking_signature"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned);
        let redacted_thinking_data = raw_anthropic_meta
            .and_then(|meta| meta.get("redacted_thinking_data"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned);
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
                if !text.trim().is_empty() {
                    content.push(text_block(
                        text.clone(),
                        projected_citations_fallback.take(),
                    ));
                }
            }
            crate::types::MessageContent::MultiModal(parts) => {
                let mut signature = thinking_signature.clone();

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
                            content.push(thinking_block(text.clone(), signature.take()));
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
                            content.push(provider_tool_result_block(
                                tool_call_id,
                                tool_name,
                                output,
                            ));
                        }
                        _ => {}
                    }
                }
            }
            #[cfg(feature = "structured-messages")]
            crate::types::MessageContent::Json(value) => {
                content.push(text_block(
                    serde_json::to_string(value).unwrap_or_default(),
                    projected_citations_fallback.take(),
                ));
            }
        }

        if let Some(data) = redacted_thinking_data {
            content.push(redacted_thinking_block(data));
        }

        let body = AnthropicMessageResponse {
            id,
            kind: "message",
            role: "assistant",
            model,
            content,
            container,
            context_management,
            stop_reason: anthropic_stop_reason(response.finish_reason.as_ref()),
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
    fn anthropic_encoder_serializes_thinking_redacted_stop_sequence_and_caller() {
        let tool_call = ContentPart::ToolCall {
            tool_call_id: "toolu_1".to_string(),
            tool_name: "weather".to_string(),
            arguments: serde_json::json!({ "city": "Tokyo" }),
            provider_executed: None,
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
            ContentPart::reasoning("Need to verify assumptions."),
            tool_call,
        ]));
        response.id = Some("msg_1".to_string());
        response.model = Some("claude-sonnet-4-5".to_string());
        response.finish_reason = Some(FinishReason::StopSequence);
        response.service_tier = Some("priority".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "anthropic".to_string(),
            HashMap::from([
                ("thinking_signature".to_string(), serde_json::json!("sig_1")),
                (
                    "redacted_thinking_data".to_string(),
                    serde_json::json!("redacted_123"),
                ),
                ("stopSequence".to_string(), serde_json::json!("</tool>")),
            ]),
        )]));

        let mut out = Vec::new();
        AnthropicMessagesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["stop_reason"], serde_json::json!("end_turn"));
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
    fn anthropic_encoder_serializes_provider_hosted_tool_blocks() {
        let response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("searching"),
            ContentPart::ToolCall {
                tool_call_id: "srvtoolu_1".to_string(),
                tool_name: "web_search".to_string(),
                arguments: serde_json::json!({ "query": "rust bridge" }),
                provider_executed: Some(true),
                provider_metadata: None,
            },
            ContentPart::ToolResult {
                tool_call_id: "srvtoolu_1".to_string(),
                tool_name: "web_search".to_string(),
                output: ToolResultOutput::json(serde_json::json!([
                    { "type": "web_search_result", "url": "https://www.rust-lang.org" }
                ])),
                provider_executed: Some(true),
                provider_metadata: None,
            },
            ContentPart::ToolCall {
                tool_call_id: "mcptoolu_1".to_string(),
                tool_name: "echo".to_string(),
                arguments: serde_json::json!({ "message": "hello" }),
                provider_executed: Some(true),
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
                }]),
                provider_executed: Some(true),
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
                provider_executed: Some(true),
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
    fn anthropic_encoder_replays_raw_usage_and_citations_metadata() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("grounded answer"),
        ]));
        response.usage = Some(Usage {
            prompt_tokens: 7,
            completion_tokens: 3,
            total_tokens: 10,
            #[allow(deprecated)]
            reasoning_tokens: None,
            #[allow(deprecated)]
            cached_tokens: None,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        });
        response.service_tier = Some("standard".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "anthropic".to_string(),
            HashMap::from([
                (
                    "usage".to_string(),
                    serde_json::json!({
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
                    }),
                ),
                (
                    "citations".to_string(),
                    serde_json::json!([
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
                    ]),
                ),
            ]),
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
    fn anthropic_encoder_prefers_text_part_citations_for_exact_block_replay() {
        let response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("Intro"),
            ContentPart::Text {
                text: "Grounded fact".to_string(),
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
            HashMap::from([(
                "container".to_string(),
                serde_json::json!({
                    "id": "container_123",
                    "expiresAt": "2025-10-14T10:28:40.590791Z",
                    "skills": [
                        {
                            "type": "anthropic",
                            "skillId": "pptx",
                            "version": "latest"
                        }
                    ]
                }),
            )]),
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
            HashMap::from([(
                "contextManagement".to_string(),
                serde_json::json!({
                    "appliedEdits": [
                        {
                            "type": "clear_tool_uses_20250919",
                            "clearedToolUses": 3,
                            "clearedInputTokens": 1000
                        }
                    ]
                }),
            )]),
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
}
