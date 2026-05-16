//! OpenAI-family non-streaming JSON response encoders (gateway/proxy).
//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::provider_metadata::openai::{
    OpenAiChatResponseExt, OpenAiContentPartExt, OpenAiSource, OpenAiSourceExt,
};
use crate::types::{ChatResponse, ContentPart, FinishReason, Usage};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

fn openai_finish_reason(reason: Option<&FinishReason>) -> Option<&'static str> {
    match reason? {
        FinishReason::Stop | FinishReason::StopSequence => Some("stop"),
        FinishReason::Length => Some("length"),
        FinishReason::ContentFilter => Some("content_filter"),
        FinishReason::ToolCalls => Some("tool_calls"),
        FinishReason::Error => Some("error"),
        FinishReason::Unknown => None,
        FinishReason::Other(_) => None,
    }
}

fn openai_chat_finish_reason(response: &ChatResponse) -> Option<String> {
    match response.raw_finish_reason.as_deref().map(str::trim) {
        Some("stop") => Some("stop".to_string()),
        Some("length") => Some("length".to_string()),
        Some("content_filter") => Some("content_filter".to_string()),
        Some("tool_calls") => Some("tool_calls".to_string()),
        Some("function_call") => Some("function_call".to_string()),
        Some("error") => Some("error".to_string()),
        _ => openai_finish_reason(response.finish_reason.as_ref()).map(ToString::to_string),
    }
}

fn openai_responses_incomplete_reason(response: &ChatResponse) -> Option<String> {
    let raw = response
        .raw_finish_reason
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    match (response.finish_reason.as_ref(), raw) {
        (Some(FinishReason::Length), Some(raw)) => Some(raw.to_string()),
        (Some(FinishReason::Length), None) => Some("max_output_tokens".to_string()),
        (Some(FinishReason::ContentFilter), Some(raw)) => Some(raw.to_string()),
        (Some(FinishReason::ContentFilter), None) => Some("content_filter".to_string()),
        (Some(FinishReason::Unknown | FinishReason::Other(_)), Some(raw)) => Some(raw.to_string()),
        (Some(FinishReason::Other(reason)), None) if !reason.trim().is_empty() => {
            Some(reason.trim().to_string())
        }
        _ => None,
    }
}

fn openai_response_status(response: &ChatResponse) -> &'static str {
    if matches!(response.finish_reason, Some(FinishReason::Error)) {
        "failed"
    } else if openai_responses_incomplete_reason(response).is_some() {
        "incomplete"
    } else {
        "completed"
    }
}

fn openai_arguments_string(arguments: &serde_json::Value) -> String {
    if let Some(raw) = arguments.as_str() {
        raw.to_string()
    } else {
        serde_json::to_string(arguments).unwrap_or_else(|_| arguments.to_string())
    }
}

fn openai_content_part_item_id(part: &ContentPart) -> Option<String> {
    part.openai_metadata().and_then(|meta| meta.item_id)
}

fn openai_web_search_action_type_for_wire(kind: &str) -> &str {
    match kind {
        "openPage" => "open_page",
        "findInPage" => "find_in_page",
        other => other,
    }
}

fn openai_jsonish_value(value: &serde_json::Value) -> Option<serde_json::Value> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(raw) if raw.trim().is_empty() => None,
        serde_json::Value::String(raw) => serde_json::from_str(raw)
            .ok()
            .or_else(|| Some(serde_json::Value::String(raw.clone()))),
        other => Some(other.clone()),
    }
}

fn openai_json_object(
    value: &serde_json::Value,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    openai_jsonish_value(value)?.as_object().cloned()
}

fn openai_output_text_logprobs_groups(value: &serde_json::Value) -> Vec<serde_json::Value> {
    match value {
        serde_json::Value::Array(groups) => groups.clone(),
        _ => Vec::new(),
    }
}

fn chat_usage_json(u: &Usage) -> serde_json::Value {
    crate::standards::openai::utils::openai_chat_usage_value(u)
}

fn responses_usage_json(u: &Usage) -> serde_json::Value {
    crate::standards::openai::utils::openai_responses_usage_value(u)
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatCompletionsJsonResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatChoice {
    pub index: u32,
    pub message: OpenAiChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: OpenAiToolCallFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct OpenAiChatCompletionsJsonResponseConverter;

impl OpenAiChatCompletionsJsonResponseConverter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OpenAiChatCompletionsJsonResponseConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonResponseConverter for OpenAiChatCompletionsJsonResponseConverter {
    fn serialize_response(
        &self,
        response: &ChatResponse,
        out: &mut Vec<u8>,
        opts: JsonEncodeOptions,
    ) -> Result<(), LlmError> {
        let id = response.id.clone().unwrap_or_else(|| "siumai".to_string());
        let model = response.model.clone().unwrap_or_default();

        let text = response.content_text().unwrap_or_default().to_string();

        let tool_calls: Vec<OpenAiToolCall> = response
            .tool_calls()
            .into_iter()
            .filter_map(|p| match p {
                ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    ..
                } => Some(OpenAiToolCall {
                    id: tool_call_id.clone(),
                    kind: "function",
                    function: OpenAiToolCallFunction {
                        name: tool_name.clone(),
                        arguments: serde_json::to_string(arguments)
                            .unwrap_or_else(|_| arguments.to_string()),
                    },
                }),
                _ => None,
            })
            .collect();

        let content = if text.trim().is_empty() && !tool_calls.is_empty() {
            None
        } else {
            Some(text)
        };

        let message = OpenAiChatMessage {
            role: "assistant",
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
        };

        let body = OpenAiChatCompletionsJsonResponse {
            id,
            object: "chat.completion",
            created: 0,
            model,
            choices: vec![OpenAiChatChoice {
                index: 0,
                message,
                finish_reason: openai_chat_finish_reason(response),
            }],
            usage: response.usage.as_ref().map(chat_usage_json),
            system_fingerprint: response.system_fingerprint.clone(),
            service_tier: response.service_tier.clone(),
        };

        if opts.pretty {
            serde_json::to_writer_pretty(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize OpenAI chat response JSON: {e}"
                ))
            })?;
        } else {
            serde_json::to_writer(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize OpenAI chat response JSON: {e}"
                ))
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiResponsesJsonResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<OpenAiResponsesIncompleteDetails>,
    pub output: Vec<serde_json::Value>,
    pub output_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiResponsesIncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        phase: Option<String>,
        content: Vec<OpenAiResponseMessageContent>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        summary: Vec<OpenAiResponseReasoningSummary>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "mcp_approval_request")]
    McpApprovalRequest {
        id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiResponseMessageContent {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<serde_json::Value>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<serde_json::Value>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiResponseReasoningSummary {
    #[serde(rename = "summary_text")]
    SummaryText { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OpenAiResponseAnnotation {
    #[serde(rename = "url_citation")]
    UrlCitation {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    #[serde(rename = "file_citation")]
    FileCitation {
        file_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quote: Option<String>,
    },
    #[serde(rename = "container_file_citation")]
    ContainerFileCitation {
        file_id: String,
        container_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quote: Option<String>,
    },
    #[serde(rename = "file_path")]
    FilePath {
        file_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenAiProviderToolItemKind {
    WebSearch,
    FileSearch,
    CodeInterpreter,
    ImageGeneration,
    Computer,
    Mcp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenAiHostedDynamicToolItemKind {
    LocalShell,
    Shell,
    ApplyPatch,
}

fn openai_provider_tool_item_kind(tool_name: &str) -> Option<OpenAiProviderToolItemKind> {
    match tool_name {
        "webSearch" | "web_search" | "web_search_preview" => {
            Some(OpenAiProviderToolItemKind::WebSearch)
        }
        "fileSearch" | "file_search" => Some(OpenAiProviderToolItemKind::FileSearch),
        "codeExecution" | "code_execution" | "code_interpreter" => {
            Some(OpenAiProviderToolItemKind::CodeInterpreter)
        }
        "generateImage" | "image_generation" => Some(OpenAiProviderToolItemKind::ImageGeneration),
        "computer_use" | "computer_use_preview" => Some(OpenAiProviderToolItemKind::Computer),
        _ if tool_name.starts_with("mcp.") => Some(OpenAiProviderToolItemKind::Mcp),
        _ => None,
    }
}

fn openai_hosted_dynamic_tool_item_kind_for_call(
    tool_name: &str,
    arguments: &serde_json::Value,
    item_id: Option<&str>,
) -> Option<OpenAiHostedDynamicToolItemKind> {
    match tool_name {
        "apply_patch" => Some(OpenAiHostedDynamicToolItemKind::ApplyPatch),
        "shell" => {
            if let Some(item_id) = item_id {
                if item_id.starts_with("lsh_") {
                    return Some(OpenAiHostedDynamicToolItemKind::LocalShell);
                }
                if item_id.starts_with("sh_") {
                    return Some(OpenAiHostedDynamicToolItemKind::Shell);
                }
            }

            let input = openai_json_object(arguments)?;
            let action = input
                .get("action")
                .and_then(serde_json::Value::as_object)
                .unwrap_or(&input);

            if action.contains_key("commands") {
                Some(OpenAiHostedDynamicToolItemKind::Shell)
            } else {
                Some(OpenAiHostedDynamicToolItemKind::LocalShell)
            }
        }
        _ => None,
    }
}

fn openai_hosted_dynamic_tool_item_kind_for_result(
    tool_name: &str,
    output: &crate::types::ToolResultOutput,
    item_id: Option<&str>,
) -> Option<OpenAiHostedDynamicToolItemKind> {
    match tool_name {
        "apply_patch" => Some(OpenAiHostedDynamicToolItemKind::ApplyPatch),
        "shell" => {
            if let Some(item_id) = item_id {
                if item_id.starts_with("lsh_") {
                    return Some(OpenAiHostedDynamicToolItemKind::LocalShell);
                }
                if item_id.starts_with("sh_") {
                    return Some(OpenAiHostedDynamicToolItemKind::Shell);
                }
            }

            let value = match output {
                crate::types::ToolResultOutput::Json { value, .. }
                | crate::types::ToolResultOutput::ErrorJson { value, .. } => Some(value),
                _ => None,
            };

            if let Some(inner) = value
                .and_then(serde_json::Value::as_object)
                .and_then(|obj| obj.get("output"))
                && inner.is_array()
            {
                return Some(OpenAiHostedDynamicToolItemKind::Shell);
            }

            Some(OpenAiHostedDynamicToolItemKind::LocalShell)
        }
        _ => None,
    }
}

fn openai_hosted_dynamic_tool_payload(
    arguments: &serde_json::Value,
    field: &str,
) -> serde_json::Value {
    let Some(input) = openai_json_object(arguments) else {
        return serde_json::Value::Object(serde_json::Map::new());
    };

    input
        .get(field)
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(input))
}

fn openai_is_tool_search_name(tool_name: &str) -> bool {
    matches!(tool_name, "toolSearch" | "tool_search")
}

fn openai_tool_search_call_parts(
    arguments: &serde_json::Value,
) -> (Option<serde_json::Value>, Option<String>) {
    let parsed = match arguments {
        serde_json::Value::String(text) => serde_json::from_str::<serde_json::Value>(text).ok(),
        other => Some(other.clone()),
    };

    let Some(obj) = parsed.as_ref().and_then(serde_json::Value::as_object) else {
        return (None, None);
    };

    let arguments = obj.get("arguments").cloned();
    let call_id = obj
        .get("call_id")
        .or_else(|| obj.get("callId"))
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string);

    (arguments, call_id)
}

fn openai_tool_search_call_output_item(part: &ContentPart) -> Option<serde_json::Value> {
    let ContentPart::ToolCall {
        tool_call_id,
        tool_name,
        arguments,
        ..
    } = part
    else {
        return None;
    };

    if !openai_is_tool_search_name(tool_name) {
        return None;
    }

    let item_id = openai_content_part_item_id(part).unwrap_or_else(|| tool_call_id.clone());
    let (tool_search_args, call_id) = openai_tool_search_call_parts(arguments);
    let execution = if call_id.is_some() {
        "client"
    } else {
        "server"
    };

    let mut item = serde_json::Map::new();
    item.insert("id".to_string(), serde_json::json!(item_id));
    item.insert("type".to_string(), serde_json::json!("tool_search_call"));
    item.insert("execution".to_string(), serde_json::json!(execution));
    item.insert(
        "call_id".to_string(),
        call_id
            .as_deref()
            .map(|id| serde_json::json!(id))
            .unwrap_or(serde_json::Value::Null),
    );
    item.insert("status".to_string(), serde_json::json!("completed"));
    if let Some(arguments) = tool_search_args {
        item.insert("arguments".to_string(), arguments);
    }

    Some(serde_json::Value::Object(item))
}

fn openai_tool_search_output_tools(
    output: &crate::types::ToolResultOutput,
) -> Option<serde_json::Value> {
    let crate::types::ToolResultOutput::Json { value, .. } = output else {
        return None;
    };

    value.get("tools").cloned()
}

fn openai_tool_search_result_output_item(
    part: &ContentPart,
    paired_call: Option<&ContentPart>,
) -> Option<serde_json::Value> {
    let ContentPart::ToolResult {
        tool_call_id,
        tool_name,
        output,
        ..
    } = part
    else {
        return None;
    };

    if !openai_is_tool_search_name(tool_name) {
        return None;
    }

    let item_id =
        openai_content_part_item_id(part).unwrap_or_else(|| format!("tso_{tool_call_id}"));
    let paired_call_id = paired_call.and_then(|call| match call {
        ContentPart::ToolCall { arguments, .. } => {
            let (_, call_id) = openai_tool_search_call_parts(arguments);
            call_id
        }
        _ => None,
    });
    let execution = if paired_call_id.is_some() {
        "client"
    } else {
        "server"
    };

    let mut item = serde_json::Map::new();
    item.insert("id".to_string(), serde_json::json!(item_id));
    item.insert("type".to_string(), serde_json::json!("tool_search_output"));
    item.insert("execution".to_string(), serde_json::json!(execution));
    item.insert(
        "call_id".to_string(),
        paired_call_id
            .as_deref()
            .map(|id| serde_json::json!(id))
            .unwrap_or(serde_json::Value::Null),
    );
    item.insert("status".to_string(), serde_json::json!("completed"));
    item.insert(
        "tools".to_string(),
        openai_tool_search_output_tools(output)
            .unwrap_or_else(|| serde_json::Value::Array(Vec::new())),
    );

    Some(serde_json::Value::Object(item))
}

fn split_openai_sources(
    response: &ChatResponse,
) -> (Vec<OpenAiSource>, HashMap<String, Vec<OpenAiSource>>) {
    let mut message_sources = Vec::new();
    let mut tool_sources: HashMap<String, Vec<OpenAiSource>> = HashMap::new();

    if let Some(sources) = response.openai_metadata().and_then(|meta| meta.sources) {
        for source in sources {
            if let Some(tool_call_id) = source.tool_call_id.clone() {
                tool_sources.entry(tool_call_id).or_default().push(source);
            } else {
                message_sources.push(source);
            }
        }
    }

    (message_sources, tool_sources)
}

fn openai_sources_to_annotations(sources: Vec<OpenAiSource>) -> Vec<serde_json::Value> {
    sources
        .into_iter()
        .filter_map(|source| {
            let source_meta = source.openai_metadata();
            let metadata_type = source_meta
                .as_ref()
                .and_then(|meta| meta.metadata_type.as_deref());
            match source.source_type.as_str() {
                "url" => Some(
                    serde_json::to_value(OpenAiResponseAnnotation::UrlCitation {
                        url: source.url,
                        title: source.title,
                    })
                    .expect("serialize url annotation"),
                ),
                "document" => {
                    let file_id = source_meta
                        .as_ref()
                        .and_then(|meta| meta.file_id.clone())
                        .unwrap_or_else(|| source.url.clone());
                    let quote = source.title.or(source.snippet);

                    if metadata_type == Some("container_file_citation")
                        || source_meta
                            .as_ref()
                            .and_then(|meta| meta.container_id.clone())
                            .is_some()
                    {
                        let container_id = source_meta
                            .as_ref()
                            .and_then(|meta| meta.container_id.clone())?;

                        return Some(
                            serde_json::to_value(OpenAiResponseAnnotation::ContainerFileCitation {
                                file_id,
                                container_id,
                                index: source_meta.as_ref().and_then(|meta| meta.index),
                                filename: source.filename,
                                quote,
                            })
                            .expect("serialize container annotation"),
                        );
                    }

                    if metadata_type == Some("file_path")
                        || source.media_type.as_deref() == Some("application/octet-stream")
                    {
                        return Some(
                            serde_json::to_value(OpenAiResponseAnnotation::FilePath {
                                file_id,
                                index: source_meta.as_ref().and_then(|meta| meta.index),
                                filename: source.filename,
                            })
                            .expect("serialize file path annotation"),
                        );
                    }

                    Some(
                        serde_json::to_value(OpenAiResponseAnnotation::FileCitation {
                            file_id,
                            index: source_meta.as_ref().and_then(|meta| meta.index),
                            filename: source.filename,
                            quote,
                        })
                        .expect("serialize file citation annotation"),
                    )
                }
                _ => None,
            }
        })
        .collect()
}

fn openai_sources_to_web_search_results(sources: &[OpenAiSource]) -> Vec<serde_json::Value> {
    sources
        .iter()
        .filter(|source| source.source_type == "url")
        .map(|source| {
            let mut obj = serde_json::Map::new();
            if !source.id.is_empty() {
                obj.insert("id".to_string(), serde_json::json!(source.id.clone()));
            }
            obj.insert("url".to_string(), serde_json::json!(source.url));
            if let Some(title) = &source.title {
                obj.insert("title".to_string(), serde_json::json!(title));
            }
            if let Some(snippet) = &source.snippet {
                obj.insert("snippet".to_string(), serde_json::json!(snippet));
            }
            serde_json::Value::Object(obj)
        })
        .collect()
}

fn openai_sources_to_file_search_results(sources: &[OpenAiSource]) -> Vec<serde_json::Value> {
    sources
        .iter()
        .filter(|source| source.source_type == "document")
        .map(|source| {
            let source_meta = source.openai_metadata();
            let mut obj = serde_json::Map::new();
            if !source.id.is_empty() {
                obj.insert("id".to_string(), serde_json::json!(source.id.clone()));
            }
            obj.insert(
                "file_id".to_string(),
                serde_json::json!(
                    source_meta
                        .as_ref()
                        .and_then(|meta| meta.file_id.clone())
                        .unwrap_or_else(|| source.url.clone())
                ),
            );
            if let Some(filename) = &source.filename {
                obj.insert("filename".to_string(), serde_json::json!(filename));
            }
            if let Some(title) = &source.title {
                obj.insert("title".to_string(), serde_json::json!(title));
            }
            if let Some(snippet) = &source.snippet {
                obj.insert("snippet".to_string(), serde_json::json!(snippet));
            }
            if let Some(media_type) = &source.media_type {
                obj.insert("media_type".to_string(), serde_json::json!(media_type));
            }
            if let Some(container_id) = source_meta
                .as_ref()
                .and_then(|meta| meta.container_id.clone())
            {
                obj.insert("container_id".to_string(), serde_json::json!(container_id));
            }
            if let Some(index) = source_meta.as_ref().and_then(|meta| meta.index) {
                obj.insert("index".to_string(), serde_json::json!(index));
            }
            serde_json::Value::Object(obj)
        })
        .collect()
}

fn openai_response_output_item_value<T: Serialize>(
    item: &T,
) -> Result<serde_json::Value, LlmError> {
    serde_json::to_value(item).map_err(|e| {
        LlmError::JsonError(format!(
            "Failed to serialize OpenAI Responses output item value: {e}"
        ))
    })
}

fn openai_tool_result_payload(
    tool_result: Option<&ContentPart>,
) -> (serde_json::Value, Option<bool>) {
    match tool_result {
        Some(ContentPart::ToolResult { output, .. }) => match output {
            crate::types::ToolResultOutput::Text { value, .. } => {
                (serde_json::Value::String(value.clone()), Some(false))
            }
            crate::types::ToolResultOutput::Json { value, .. } => (value.clone(), Some(false)),
            crate::types::ToolResultOutput::ExecutionDenied { reason, .. } => (
                serde_json::json!({
                    "reason": reason,
                }),
                Some(true),
            ),
            crate::types::ToolResultOutput::ErrorText { value, .. } => {
                (serde_json::Value::String(value.clone()), Some(true))
            }
            crate::types::ToolResultOutput::ErrorJson { value, .. } => (value.clone(), Some(true)),
            crate::types::ToolResultOutput::Content { value, .. } => (
                serde_json::Value::Array(
                    value
                        .iter()
                        .map(|part| serde_json::to_value(part).unwrap_or(serde_json::Value::Null))
                        .collect(),
                ),
                Some(false),
            ),
        },
        _ => (serde_json::Value::Null, None),
    }
}

fn openai_custom_tool_output_item(
    tool_call: Option<&ContentPart>,
    tool_result: Option<&ContentPart>,
) -> Option<serde_json::Value> {
    let (tool_call_id, tool_name, arguments, provider_executed) = match tool_call.or(tool_result) {
        Some(ContentPart::ToolCall {
            tool_call_id,
            tool_name,
            arguments,
            provider_executed,
            ..
        }) => (
            tool_call_id.clone(),
            tool_name.clone(),
            Some(arguments),
            *provider_executed,
        ),
        Some(ContentPart::ToolResult {
            tool_call_id,
            tool_name,
            provider_executed,
            ..
        }) => (
            tool_call_id.clone(),
            tool_name.clone(),
            None,
            *provider_executed,
        ),
        _ => return None,
    };

    if provider_executed != Some(true) {
        return None;
    }

    let item_id = tool_call
        .and_then(openai_content_part_item_id)
        .or_else(|| tool_result.and_then(openai_content_part_item_id))
        .unwrap_or_else(|| tool_call_id.clone());
    let input = arguments
        .map(openai_arguments_string)
        .unwrap_or_else(|| "{}".to_string());
    let (output, is_error) = openai_tool_result_payload(tool_result);

    let mut item = serde_json::Map::new();
    item.insert("id".to_string(), serde_json::json!(item_id));
    item.insert("type".to_string(), serde_json::json!("custom_tool_call"));
    item.insert("status".to_string(), serde_json::json!("completed"));
    item.insert("name".to_string(), serde_json::json!(tool_name));
    item.insert("input".to_string(), serde_json::json!(input));
    if !output.is_null() {
        item.insert("output".to_string(), output);
    }
    if let Some(is_error) = is_error
        && is_error
    {
        item.insert("is_error".to_string(), serde_json::json!(true));
    }

    Some(serde_json::Value::Object(item))
}

fn openai_provider_tool_output_item(
    tool_call: Option<&ContentPart>,
    tool_result: Option<&ContentPart>,
    tool_sources: &[OpenAiSource],
) -> Option<serde_json::Value> {
    let (tool_call_id, tool_name, arguments, provider_executed) = match tool_call.or(tool_result) {
        Some(ContentPart::ToolCall {
            tool_call_id,
            tool_name,
            arguments,
            provider_executed,
            ..
        }) => (
            tool_call_id.clone(),
            tool_name.clone(),
            Some(arguments),
            *provider_executed,
        ),
        Some(ContentPart::ToolResult {
            tool_call_id,
            tool_name,
            provider_executed,
            ..
        }) => (
            tool_call_id.clone(),
            tool_name.clone(),
            None,
            *provider_executed,
        ),
        _ => return None,
    };

    if provider_executed != Some(true) {
        return None;
    }

    let kind = openai_provider_tool_item_kind(&tool_name)?;
    let item_id = tool_call
        .and_then(openai_content_part_item_id)
        .or_else(|| tool_result.and_then(openai_content_part_item_id))
        .unwrap_or_else(|| tool_call_id.clone());

    let (result_value, _) = openai_tool_result_payload(tool_result);
    let result_obj = result_value.as_object();
    let input_obj = arguments.and_then(openai_json_object);

    let mut item = serde_json::Map::new();
    item.insert("id".to_string(), serde_json::json!(item_id));
    if item_id != tool_call_id {
        item.insert(
            "call_id".to_string(),
            serde_json::json!(tool_call_id.clone()),
        );
    }
    item.insert("status".to_string(), serde_json::json!("completed"));

    match kind {
        OpenAiProviderToolItemKind::WebSearch => {
            item.insert("type".to_string(), serde_json::json!("web_search_call"));

            if let Some(status) = result_obj.and_then(|obj| obj.get("status"))
                && !status.is_null()
            {
                item.insert("status".to_string(), status.clone());
            }

            let normalized_sources = result_obj
                .and_then(|obj| obj.get("sources"))
                .and_then(|value| value.as_array())
                .filter(|sources| !sources.is_empty())
                .cloned();

            if let Some(action) = result_obj.and_then(|obj| obj.get("action"))
                && let Some(mut action_obj) = action.as_object().cloned()
            {
                if let Some(action_type) = action_obj.get("type").and_then(|value| value.as_str()) {
                    action_obj.insert(
                        "type".to_string(),
                        serde_json::Value::String(
                            openai_web_search_action_type_for_wire(action_type).to_string(),
                        ),
                    );
                }
                if !action_obj.contains_key("sources")
                    && let Some(sources) = normalized_sources.clone()
                {
                    action_obj.insert("sources".to_string(), serde_json::Value::Array(sources));
                }
                item.insert("action".to_string(), serde_json::Value::Object(action_obj));
            }

            let results = result_obj
                .and_then(|obj| obj.get("results"))
                .and_then(|value| value.as_array().cloned())
                .or_else(|| {
                    let results = openai_sources_to_web_search_results(tool_sources);
                    (!results.is_empty()).then_some(results)
                });

            if let Some(results) = results {
                item.insert("results".to_string(), serde_json::Value::Array(results));
            }

            if !item.contains_key("action")
                && let Some(input) = input_obj.as_ref()
                && let Some(query) = input.get("query").or_else(|| input.get("q"))
            {
                let mut action = serde_json::Map::new();
                action.insert("type".to_string(), serde_json::json!("search"));
                action.insert("query".to_string(), query.clone());
                if let Some(sources) = normalized_sources {
                    action.insert("sources".to_string(), serde_json::Value::Array(sources));
                }
                item.insert("action".to_string(), serde_json::Value::Object(action));
            }
        }
        OpenAiProviderToolItemKind::FileSearch => {
            item.insert("type".to_string(), serde_json::json!("file_search_call"));

            if let Some(status) = result_obj.and_then(|obj| obj.get("status"))
                && !status.is_null()
            {
                item.insert("status".to_string(), status.clone());
            }

            if let Some(queries) = result_obj.and_then(|obj| obj.get("queries"))
                && !queries.is_null()
            {
                item.insert("queries".to_string(), queries.clone());
            }

            let results = result_obj
                .and_then(|obj| obj.get("results"))
                .and_then(|value| value.as_array().cloned())
                .or_else(|| {
                    let results = openai_sources_to_file_search_results(tool_sources);
                    (!results.is_empty()).then_some(results)
                });

            if let Some(results) = results {
                item.insert("results".to_string(), serde_json::Value::Array(results));
            }
        }
        OpenAiProviderToolItemKind::CodeInterpreter => {
            item.insert(
                "type".to_string(),
                serde_json::json!("code_interpreter_call"),
            );

            if let Some(input) = input_obj.as_ref() {
                if let Some(code) = input.get("code")
                    && !code.is_null()
                {
                    item.insert("code".to_string(), code.clone());
                }
                if let Some(container_id) = input
                    .get("containerId")
                    .or_else(|| input.get("container_id"))
                    && !container_id.is_null()
                {
                    item.insert("container_id".to_string(), container_id.clone());
                }
            }

            if let Some(outputs) = result_obj.and_then(|obj| obj.get("outputs"))
                && !outputs.is_null()
            {
                item.insert("outputs".to_string(), outputs.clone());
            }
        }
        OpenAiProviderToolItemKind::ImageGeneration => {
            item.insert(
                "type".to_string(),
                serde_json::json!("image_generation_call"),
            );

            if let Some(result) = result_obj.and_then(|obj| obj.get("result"))
                && !result.is_null()
            {
                item.insert("result".to_string(), result.clone());
            } else if !result_value.is_null() {
                item.insert("result".to_string(), result_value);
            }
        }
        OpenAiProviderToolItemKind::Computer => {
            item.insert("type".to_string(), serde_json::json!("computer_call"));

            if let Some(status) = result_obj.and_then(|obj| obj.get("status"))
                && !status.is_null()
            {
                item.insert("status".to_string(), status.clone());
            }

            if let Some(action) = result_obj.and_then(|obj| obj.get("action"))
                && action.is_object()
            {
                item.insert("action".to_string(), action.clone());
            } else if let Some(input) = input_obj.as_ref()
                && let Some(action) = input.get("action")
                && action.is_object()
            {
                item.insert("action".to_string(), action.clone());
            }
        }
        OpenAiProviderToolItemKind::Mcp => {
            item.insert("type".to_string(), serde_json::json!("mcp_call"));
            item.insert(
                "name".to_string(),
                serde_json::json!(tool_name.strip_prefix("mcp.").unwrap_or(tool_name.as_str())),
            );

            if let Some(result) = result_obj {
                if let Some(server_label) = result
                    .get("serverLabel")
                    .or_else(|| result.get("server_label"))
                    && !server_label.is_null()
                {
                    item.insert("server_label".to_string(), server_label.clone());
                }
                if let Some(arguments) = result.get("arguments")
                    && !arguments.is_null()
                {
                    item.insert("arguments".to_string(), arguments.clone());
                } else if let Some(arguments) = arguments {
                    item.insert(
                        "arguments".to_string(),
                        serde_json::json!(openai_arguments_string(arguments)),
                    );
                }
                if let Some(output) = result.get("output")
                    && !output.is_null()
                {
                    item.insert("output".to_string(), output.clone());
                }
                if let Some(error) = result.get("error")
                    && !error.is_null()
                {
                    item.insert("error".to_string(), error.clone());
                }
            } else if let Some(arguments) = arguments {
                item.insert(
                    "arguments".to_string(),
                    serde_json::json!(openai_arguments_string(arguments)),
                );
            }
        }
    }

    Some(serde_json::Value::Object(item))
}

fn openai_provider_executed_output_item(
    tool_call: Option<&ContentPart>,
    tool_result: Option<&ContentPart>,
    tool_sources: &[OpenAiSource],
) -> Option<(serde_json::Value, bool)> {
    if let Some(item) = openai_provider_tool_output_item(tool_call, tool_result, tool_sources) {
        return Some((item, true));
    }

    openai_custom_tool_output_item(tool_call, tool_result).map(|item| (item, false))
}

fn openai_hosted_dynamic_tool_call_output_item(part: &ContentPart) -> Option<serde_json::Value> {
    let ContentPart::ToolCall {
        tool_call_id,
        tool_name,
        arguments,
        provider_executed,
        dynamic,
        ..
    } = part
    else {
        return None;
    };

    if *provider_executed == Some(true) || *dynamic != Some(true) {
        return None;
    }

    let item_id = openai_content_part_item_id(part);
    let kind =
        openai_hosted_dynamic_tool_item_kind_for_call(tool_name, arguments, item_id.as_deref())?;

    let mut item = serde_json::Map::new();
    if let Some(item_id) = item_id {
        item.insert("id".to_string(), serde_json::json!(item_id));
    }
    item.insert("status".to_string(), serde_json::json!("completed"));
    item.insert("call_id".to_string(), serde_json::json!(tool_call_id));

    match kind {
        OpenAiHostedDynamicToolItemKind::LocalShell => {
            item.insert("type".to_string(), serde_json::json!("local_shell_call"));
            item.insert(
                "action".to_string(),
                openai_hosted_dynamic_tool_payload(arguments, "action"),
            );
        }
        OpenAiHostedDynamicToolItemKind::Shell => {
            item.insert("type".to_string(), serde_json::json!("shell_call"));
            item.insert(
                "action".to_string(),
                openai_hosted_dynamic_tool_payload(arguments, "action"),
            );
        }
        OpenAiHostedDynamicToolItemKind::ApplyPatch => {
            item.insert("type".to_string(), serde_json::json!("apply_patch_call"));
            item.insert(
                "operation".to_string(),
                openai_hosted_dynamic_tool_payload(arguments, "operation"),
            );
        }
    }

    Some(serde_json::Value::Object(item))
}

fn openai_hosted_dynamic_tool_result_output_item(part: &ContentPart) -> Option<serde_json::Value> {
    let ContentPart::ToolResult {
        tool_call_id,
        tool_name,
        output,
        provider_executed,
        dynamic,
        ..
    } = part
    else {
        return None;
    };

    if *provider_executed == Some(true) || *dynamic != Some(true) {
        return None;
    }

    let item_id = openai_content_part_item_id(part);
    let kind =
        openai_hosted_dynamic_tool_item_kind_for_result(tool_name, output, item_id.as_deref())?;
    let (result_value, _) = openai_tool_result_payload(Some(part));

    let mut item = serde_json::Map::new();
    item.insert("call_id".to_string(), serde_json::json!(tool_call_id));

    match kind {
        OpenAiHostedDynamicToolItemKind::LocalShell => {
            item.insert(
                "type".to_string(),
                serde_json::json!("local_shell_call_output"),
            );
            item.insert(
                "output".to_string(),
                result_value
                    .as_object()
                    .and_then(|obj| obj.get("output").cloned())
                    .unwrap_or(result_value),
            );
        }
        OpenAiHostedDynamicToolItemKind::Shell => {
            item.insert("type".to_string(), serde_json::json!("shell_call_output"));
            item.insert(
                "output".to_string(),
                result_value
                    .as_object()
                    .and_then(|obj| obj.get("output").cloned())
                    .unwrap_or(result_value),
            );
        }
        OpenAiHostedDynamicToolItemKind::ApplyPatch => {
            item.insert(
                "type".to_string(),
                serde_json::json!("apply_patch_call_output"),
            );
            item.insert(
                "status".to_string(),
                result_value
                    .as_object()
                    .and_then(|obj| obj.get("status").cloned())
                    .unwrap_or_else(|| serde_json::json!("completed")),
            );
            if let Some(output) = result_value
                .as_object()
                .and_then(|obj| obj.get("output").cloned())
            {
                item.insert("output".to_string(), output);
            } else if !result_value.is_null() {
                item.insert("output".to_string(), result_value);
            }
        }
    }

    Some(serde_json::Value::Object(item))
}

#[derive(Debug, Clone)]
pub struct OpenAiResponsesJsonResponseConverter;

impl OpenAiResponsesJsonResponseConverter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OpenAiResponsesJsonResponseConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonResponseConverter for OpenAiResponsesJsonResponseConverter {
    fn serialize_response(
        &self,
        response: &ChatResponse,
        out: &mut Vec<u8>,
        opts: JsonEncodeOptions,
    ) -> Result<(), LlmError> {
        let id = response.id.clone().unwrap_or_else(|| "siumai".to_string());
        let model = response.model.clone().unwrap_or_default();
        let mut output = Vec::new();
        let mut emitted_provider_tool_items: HashSet<String> = HashSet::new();
        let first_text_meta = response.content.as_multimodal().and_then(|parts| {
            parts.iter().find_map(|part| match part {
                ContentPart::Text { .. } => part.openai_metadata(),
                _ => None,
            })
        });
        let message_item_id = first_text_meta
            .as_ref()
            .and_then(|meta| meta.item_id.clone())
            .unwrap_or_else(|| format!("msg_{id}"));
        let message_phase = first_text_meta.as_ref().and_then(|meta| meta.phase.clone());
        let (mut message_sources, mut tool_sources_by_call_id) = split_openai_sources(response);

        #[allow(unreachable_patterns)]
        let mut message_content: Vec<OpenAiResponseMessageContent> = match &response.content {
            crate::types::MessageContent::Text(text) => {
                vec![OpenAiResponseMessageContent::OutputText {
                    text: text.clone(),
                    annotations: None,
                    logprobs: None,
                }]
            }
            crate::types::MessageContent::MultiModal(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text, .. } => {
                        let meta = part.openai_metadata();
                        Some(OpenAiResponseMessageContent::OutputText {
                            text: text.clone(),
                            annotations: meta.as_ref().and_then(|meta| meta.annotations.clone()),
                            logprobs: None,
                        })
                    }
                    _ => None,
                })
                .collect(),
            _ => {
                vec![OpenAiResponseMessageContent::OutputText {
                    text: response.content.all_text(),
                    annotations: None,
                    logprobs: None,
                }]
            }
        };

        if let Some(parts) = response.content.as_multimodal() {
            let provider_tool_calls_by_call_id: HashMap<&str, &ContentPart> = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ToolCall {
                        tool_call_id,
                        provider_executed,
                        ..
                    } if *provider_executed == Some(true) => Some((tool_call_id.as_str(), part)),
                    _ => None,
                })
                .collect();
            let approval_requests_by_call_id: HashMap<&str, &ContentPart> = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ToolApprovalRequest { tool_call_id, .. } => {
                        Some((tool_call_id.as_str(), part))
                    }
                    _ => None,
                })
                .collect();
            let provider_tool_results_by_call_id: HashMap<&str, &ContentPart> = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ToolResult {
                        tool_call_id,
                        provider_executed,
                        ..
                    } if *provider_executed == Some(true) => Some((tool_call_id.as_str(), part)),
                    _ => None,
                })
                .collect();
            let tool_search_calls_by_call_id: HashMap<&str, &ContentPart> = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        ..
                    } if openai_is_tool_search_name(tool_name) => {
                        Some((tool_call_id.as_str(), part))
                    }
                    _ => None,
                })
                .collect();

            for (index, part) in parts.iter().enumerate() {
                match part {
                    ContentPart::Reasoning { text, .. } => {
                        let meta = part.openai_metadata();
                        let item_id = meta
                            .as_ref()
                            .and_then(|meta| meta.item_id.clone())
                            .unwrap_or_else(|| format!("rs_{id}_{index}"));
                        let summary = vec![OpenAiResponseReasoningSummary::SummaryText {
                            text: text.clone(),
                        }];

                        output.push(openai_response_output_item_value(
                            &OpenAiResponseOutputItem::Reasoning {
                                id: item_id,
                                summary,
                                encrypted_content: meta
                                    .and_then(|meta| meta.reasoning_encrypted_content.clone()),
                            },
                        )?);
                    }
                    ContentPart::ToolCall {
                        tool_call_id,
                        provider_executed,
                        ..
                    } => {
                        if let Some(item) = openai_hosted_dynamic_tool_call_output_item(part) {
                            output.push(item);
                            continue;
                        }

                        if let Some(item) = openai_tool_search_call_output_item(part) {
                            output.push(item);
                            continue;
                        }

                        if *provider_executed == Some(true) {
                            if approval_requests_by_call_id.contains_key(tool_call_id.as_str()) {
                                continue;
                            }

                            if emitted_provider_tool_items.insert(tool_call_id.clone()) {
                                let tool_sources = tool_sources_by_call_id
                                    .remove(tool_call_id)
                                    .unwrap_or_default();

                                if let Some((item, carried_sources)) =
                                    openai_provider_executed_output_item(
                                        Some(part),
                                        provider_tool_results_by_call_id
                                            .get(tool_call_id.as_str())
                                            .copied(),
                                        tool_sources.as_slice(),
                                    )
                                {
                                    output.push(item);
                                    if !carried_sources {
                                        message_sources.extend(tool_sources);
                                    }
                                } else {
                                    message_sources.extend(tool_sources);
                                }
                            }
                            continue;
                        }

                        let ContentPart::ToolCall {
                            tool_name,
                            arguments,
                            ..
                        } = part
                        else {
                            continue;
                        };
                        let meta = part.openai_metadata();
                        let item_id = meta
                            .as_ref()
                            .and_then(|meta| meta.item_id.clone())
                            .unwrap_or_else(|| format!("fc_{tool_call_id}"));

                        output.push(openai_response_output_item_value(
                            &OpenAiResponseOutputItem::FunctionCall {
                                id: item_id,
                                call_id: tool_call_id.clone(),
                                name: tool_name.clone(),
                                arguments: openai_arguments_string(arguments),
                            },
                        )?);
                    }
                    ContentPart::ToolApprovalRequest {
                        approval_id,
                        tool_call_id,
                        ..
                    } => {
                        let Some(ContentPart::ToolCall {
                            tool_name,
                            arguments,
                            provider_executed,
                            ..
                        }) = provider_tool_calls_by_call_id
                            .get(tool_call_id.as_str())
                            .copied()
                        else {
                            continue;
                        };

                        if *provider_executed != Some(true) {
                            continue;
                        }

                        output.push(openai_response_output_item_value(
                            &OpenAiResponseOutputItem::McpApprovalRequest {
                                id: approval_id.clone(),
                                name: tool_name
                                    .strip_prefix("mcp.")
                                    .unwrap_or(tool_name.as_str())
                                    .to_string(),
                                arguments: openai_arguments_string(arguments),
                            },
                        )?);
                    }
                    ContentPart::ToolResult {
                        tool_call_id,
                        provider_executed,
                        ..
                    } => {
                        if let Some(item) = openai_hosted_dynamic_tool_result_output_item(part) {
                            output.push(item);
                            continue;
                        }

                        if let Some(item) = openai_tool_search_result_output_item(
                            part,
                            tool_search_calls_by_call_id
                                .get(tool_call_id.as_str())
                                .copied(),
                        ) {
                            output.push(item);
                            continue;
                        }

                        if *provider_executed == Some(true)
                            && emitted_provider_tool_items.insert(tool_call_id.clone())
                        {
                            let tool_sources = tool_sources_by_call_id
                                .remove(tool_call_id)
                                .unwrap_or_default();

                            if let Some((item, carried_sources)) =
                                openai_provider_executed_output_item(
                                    None,
                                    Some(part),
                                    tool_sources.as_slice(),
                                )
                            {
                                output.push(item);
                                if !carried_sources {
                                    message_sources.extend(tool_sources);
                                }
                            } else {
                                message_sources.extend(tool_sources);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        for leftover_sources in tool_sources_by_call_id.into_values() {
            message_sources.extend(leftover_sources);
        }

        let annotations = {
            let annotations = openai_sources_to_annotations(message_sources);
            (!annotations.is_empty()).then_some(annotations)
        };

        let has_part_annotations = message_content.iter().any(|part| {
            matches!(
                part,
                OpenAiResponseMessageContent::OutputText {
                    annotations: Some(annotations),
                    ..
                } if !annotations.is_empty()
            )
        });

        if let Some(annotations) = annotations
            && !has_part_annotations
        {
            if message_content.is_empty() {
                message_content.push(OpenAiResponseMessageContent::OutputText {
                    text: String::new(),
                    annotations: Some(annotations),
                    logprobs: None,
                });
            } else if let Some(OpenAiResponseMessageContent::OutputText {
                annotations: slot, ..
            }) = message_content.first_mut()
            {
                *slot = Some(annotations);
            }
        }

        if let Some(logprobs) = response.openai_metadata().and_then(|meta| meta.logprobs) {
            let mut groups = openai_output_text_logprobs_groups(&logprobs).into_iter();
            for part in &mut message_content {
                let OpenAiResponseMessageContent::OutputText { logprobs: slot, .. } = part;
                *slot = groups.next();
            }
        }

        if !message_content.is_empty() {
            output.insert(
                0,
                openai_response_output_item_value(&OpenAiResponseOutputItem::Message {
                    id: message_item_id,
                    role: "assistant",
                    phase: message_phase,
                    content: message_content,
                })?,
            );
        }

        let body = OpenAiResponsesJsonResponse {
            id,
            object: "response",
            created: 0,
            model,
            status: openai_response_status(response),
            incomplete_details: openai_responses_incomplete_reason(response)
                .map(|reason| OpenAiResponsesIncompleteDetails { reason }),
            output,
            output_text: response.content.all_text(),
            finish_reason: openai_finish_reason(response.finish_reason.as_ref()),
            usage: response.usage.as_ref().map(responses_usage_json),
            system_fingerprint: response.system_fingerprint.clone(),
            service_tier: response.service_tier.clone(),
        };

        if opts.pretty {
            serde_json::to_writer_pretty(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize OpenAI Responses JSON response: {e}"
                ))
            })?;
        } else {
            serde_json::to_writer(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize OpenAI Responses JSON response: {e}"
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
    fn openai_json_response_encoder_source_does_not_read_request_provider_options() {
        let source = include_str!("json_response.rs");
        let production_source = source
            .split("#[cfg(test)]")
            .next()
            .expect("production source");

        for forbidden in ["provider_options", ".provider_options", "providerOptions"] {
            assert!(
                !production_source.contains(forbidden),
                "OpenAI JSON response encoders must not read request-side provider options fragment `{forbidden}`"
            );
        }
    }

    #[test]
    fn responses_encoder_serializes_reasoning_ids_annotations_and_native_fields() {
        let reasoning = ContentPart::Reasoning {
            text: "Need to compare both options.".to_string(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({
                    "itemId": "rs_1",
                    "reasoningEncryptedContent": "enc_123"
                }),
            )])),
        };

        let tool_call = ContentPart::ToolCall {
            tool_call_id: "call_1".to_string(),
            tool_name: "get_weather".to_string(),
            arguments: serde_json::Value::String(r#"{"city":"Tokyo"}"#.to_string()),
            provider_executed: None,
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({
                    "itemId": "fc_1"
                }),
            )])),
        };

        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            reasoning,
            ContentPart::Text {
                text: "It is sunny.".to_string(),
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "msg_1"
                    }),
                )])),
            },
            tool_call,
        ]));
        response.id = Some("resp_1".to_string());
        response.model = Some("gpt-5-mini".to_string());
        response.finish_reason = Some(FinishReason::ToolCalls);
        response.system_fingerprint = Some("fp_123".to_string());
        response.service_tier = Some("priority".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "sources": [
                    {
                        "id": "src_1",
                        "source_type": "url",
                        "url": "https://example.com",
                        "title": "Example"
                    },
                    {
                        "id": "src_2",
                        "source_type": "document",
                        "url": "file_123",
                        "title": "Design Doc",
                        "filename": "design.md",
                        "provider_metadata": {
                            "openai": {
                                "type": "file_citation",
                                "fileId": "file_123",
                                "index": 7
                            }
                        }
                    }
                ]
            }),
        )]));

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["finish_reason"], serde_json::json!("tool_calls"));
        assert_eq!(value["system_fingerprint"], serde_json::json!("fp_123"));
        assert_eq!(value["service_tier"], serde_json::json!("priority"));
        assert_eq!(value["output"][0]["id"], serde_json::json!("msg_1"));
        assert_eq!(value["output"][0]["phase"], serde_json::Value::Null);
        assert_eq!(
            value["output"][0]["content"][0]["annotations"][0]["type"],
            serde_json::json!("url_citation")
        );
        assert_eq!(
            value["output"][0]["content"][0]["annotations"][1]["type"],
            serde_json::json!("file_citation")
        );
        assert_eq!(
            value["output"][0]["content"][0]["annotations"][1]["index"],
            serde_json::json!(7)
        );
        assert_eq!(value["output"][1]["type"], serde_json::json!("reasoning"));
        assert_eq!(value["output"][1]["id"], serde_json::json!("rs_1"));
        assert_eq!(
            value["output"][1]["encrypted_content"],
            serde_json::json!("enc_123")
        );
        assert_eq!(value["output"][2]["id"], serde_json::json!("fc_1"));
        assert_eq!(
            value["output"][2]["arguments"],
            serde_json::json!(r#"{"city":"Tokyo"}"#)
        );
    }

    #[test]
    fn responses_encoder_replays_incomplete_raw_finish_reason() {
        let mut response =
            ChatResponse::new(crate::types::MessageContent::Text("partial".to_string()));
        response.id = Some("resp_incomplete".to_string());
        response.model = Some("gpt-5-mini".to_string());
        response.finish_reason = Some(FinishReason::Length);
        response.raw_finish_reason = Some("max_output_tokens".to_string());

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["status"], serde_json::json!("incomplete"));
        assert_eq!(
            value["incomplete_details"]["reason"],
            serde_json::json!("max_output_tokens")
        );
        assert_eq!(value["finish_reason"], serde_json::json!("length"));
    }

    #[test]
    fn responses_encoder_replays_unknown_incomplete_raw_finish_reason() {
        let mut response = ChatResponse::new(crate::types::MessageContent::Text(
            "provider-specific".to_string(),
        ));
        response.id = Some("resp_unknown_incomplete".to_string());
        response.model = Some("gpt-5-mini".to_string());
        response.finish_reason = Some(FinishReason::Other("other".to_string()));
        response.raw_finish_reason = Some("quota_exhausted".to_string());

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["status"], serde_json::json!("incomplete"));
        assert_eq!(
            value["incomplete_details"]["reason"],
            serde_json::json!("quota_exhausted")
        );
        assert!(value.get("finish_reason").is_none());
    }

    #[test]
    fn responses_encoder_serializes_provider_executed_web_search_with_tool_sources() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolCall {
                tool_call_id: "ws_1".to_string(),
                tool_name: "webSearch".to_string(),
                arguments: serde_json::json!({
                    "query": "rust release notes"
                }),
                provider_executed: Some(true),
                dynamic: Some(true),
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "ws_item_1"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "ws_1".to_string(),
                tool_name: "webSearch".to_string(),
                output: crate::types::ToolResultOutput::json(serde_json::json!({
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "rust release notes"
                    }
                })),
                input: Some(serde_json::json!({
                    "query": "rust release notes"
                })),
                provider_executed: Some(true),
                dynamic: Some(true),
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        ]));
        response.id = Some("resp_provider_tool".to_string());
        response.model = Some("gpt-5-mini".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "sources": [
                    {
                        "id": "src_1",
                        "source_type": "url",
                        "url": "https://blog.rust-lang.org/",
                        "title": "Rust Blog",
                        "snippet": "Release notes",
                        "tool_call_id": "ws_1"
                    }
                ]
            }),
        )]));

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["output"][0]["type"],
            serde_json::json!("web_search_call")
        );
        assert_eq!(value["output"][0]["id"], serde_json::json!("ws_item_1"));
        assert_eq!(value["output"][0]["call_id"], serde_json::json!("ws_1"));
        assert_eq!(
            value["output"][0]["action"]["query"],
            serde_json::json!("rust release notes")
        );
        assert_eq!(
            value["output"][0]["results"][0]["id"],
            serde_json::json!("src_1")
        );
        assert_eq!(
            value["output"][0]["results"][0]["url"],
            serde_json::json!("https://blog.rust-lang.org/")
        );
        assert_eq!(
            value["output"][0]["results"][0]["snippet"],
            serde_json::json!("Release notes")
        );
    }

    #[test]
    fn responses_encoder_ignores_legacy_top_level_message_item_id() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::text("hello"),
        ]));
        response.id = Some("resp_legacy_msg".to_string());
        response.model = Some("gpt-5-mini".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "itemId": "msg_legacy"
            }),
        )]));

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["output"][0]["type"], serde_json::json!("message"));
        assert_eq!(
            value["output"][0]["id"],
            serde_json::json!("msg_resp_legacy_msg")
        );
    }

    #[test]
    fn responses_encoder_replays_normalized_web_search_tool_result_sources() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolCall {
                tool_call_id: "ws_1".to_string(),
                tool_name: "webSearch".to_string(),
                arguments: serde_json::json!({
                    "query": "rust release notes"
                }),
                provider_executed: Some(true),
                dynamic: Some(true),
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "ws_item_1"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "ws_1".to_string(),
                tool_name: "webSearch".to_string(),
                output: crate::types::ToolResultOutput::json(serde_json::json!({
                    "action": {
                        "type": "search",
                        "query": "rust release notes"
                    },
                    "sources": [
                        {
                            "type": "url",
                            "url": "https://blog.rust-lang.org/"
                        }
                    ]
                })),
                input: Some(serde_json::json!({
                    "query": "rust release notes"
                })),
                provider_executed: Some(true),
                dynamic: Some(true),
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        ]));
        response.id = Some("resp_provider_tool".to_string());
        response.model = Some("gpt-5-mini".to_string());

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["output"][0]["type"],
            serde_json::json!("web_search_call")
        );
        assert_eq!(value["output"][0]["id"], serde_json::json!("ws_item_1"));
        assert_eq!(
            value["output"][0]["action"]["query"],
            serde_json::json!("rust release notes")
        );
        assert_eq!(
            value["output"][0]["action"]["sources"][0]["url"],
            serde_json::json!("https://blog.rust-lang.org/")
        );
        assert!(
            value["output"][0].get("results").is_none(),
            "normalized web-search tool-result sources should replay through action.sources"
        );
    }

    #[test]
    fn responses_encoder_falls_back_unknown_provider_tool_to_custom_item() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolCall {
                tool_call_id: "browser_1".to_string(),
                tool_name: "browser_agent".to_string(),
                arguments: serde_json::Value::String(
                    r#"{"url":"https://example.com"}"#.to_string(),
                ),
                provider_executed: Some(true),
                dynamic: Some(true),
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "ct_1"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "browser_1".to_string(),
                tool_name: "browser_agent".to_string(),
                output: crate::types::ToolResultOutput::error_json(serde_json::json!({
                    "message": "blocked"
                })),
                input: Some(serde_json::Value::String(
                    r#"{"url":"https://example.com"}"#.to_string(),
                )),
                provider_executed: Some(true),
                dynamic: Some(true),
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        ]));
        response.id = Some("resp_custom_tool".to_string());
        response.model = Some("gpt-5-mini".to_string());
        response.provider_metadata = Some(HashMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "sources": [
                    {
                        "id": "src_custom_1",
                        "source_type": "url",
                        "url": "https://example.com",
                        "title": "Example",
                        "tool_call_id": "browser_1"
                    }
                ]
            }),
        )]));

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["output"][0]["type"], serde_json::json!("message"));
        assert_eq!(
            value["output"][0]["content"][0]["annotations"][0]["type"],
            serde_json::json!("url_citation")
        );
        assert_eq!(
            value["output"][1]["type"],
            serde_json::json!("custom_tool_call")
        );
        assert_eq!(value["output"][1]["id"], serde_json::json!("ct_1"));
        assert_eq!(
            value["output"][1]["input"],
            serde_json::json!(r#"{"url":"https://example.com"}"#)
        );
        assert_eq!(value["output"][1]["is_error"], serde_json::json!(true));
        assert_eq!(
            value["output"][1]["output"]["message"],
            serde_json::json!("blocked")
        );
    }

    #[test]
    fn responses_encoder_serializes_tool_search_call_and_output_items() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolCall {
                tool_call_id: "tsc_server_1".to_string(),
                tool_name: "toolSearch".to_string(),
                arguments: serde_json::Value::String(
                    r#"{"arguments":{"paths":["get_weather"]},"call_id":null}"#.to_string(),
                ),
                provider_executed: Some(true),
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "tsc_server_item"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "tsc_server_1".to_string(),
                tool_name: "toolSearch".to_string(),
                output: crate::types::ToolResultOutput::json(serde_json::json!({
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "parameters": { "type": "object" }
                        }
                    ]
                })),
                input: None,
                provider_executed: None,
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "tso_server_item"
                    }),
                )])),
            },
            ContentPart::ToolCall {
                tool_call_id: "call_final".to_string(),
                tool_name: "toolSearch".to_string(),
                arguments: serde_json::Value::String(
                    r#"{"arguments":{"goal":"Find the weather tool"},"call_id":"call_final"}"#
                        .to_string(),
                ),
                provider_executed: None,
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "tsc_client_item"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "call_final".to_string(),
                tool_name: "toolSearch".to_string(),
                output: crate::types::ToolResultOutput::json(serde_json::json!({
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_forecast",
                            "parameters": { "type": "object" }
                        }
                    ]
                })),
                input: None,
                provider_executed: None,
                dynamic: None,
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "tso_client_item"
                    }),
                )])),
            },
        ]));
        response.id = Some("resp_tool_search".to_string());
        response.model = Some("gpt-5.4".to_string());

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["output"][0]["type"],
            serde_json::json!("tool_search_call")
        );
        assert_eq!(
            value["output"][0]["id"],
            serde_json::json!("tsc_server_item")
        );
        assert_eq!(value["output"][0]["execution"], serde_json::json!("server"));
        assert_eq!(value["output"][0]["call_id"], serde_json::Value::Null);
        assert_eq!(
            value["output"][0]["arguments"]["paths"],
            serde_json::json!(["get_weather"])
        );
        assert_eq!(
            value["output"][1]["type"],
            serde_json::json!("tool_search_output")
        );
        assert_eq!(
            value["output"][1]["id"],
            serde_json::json!("tso_server_item")
        );
        assert_eq!(value["output"][1]["execution"], serde_json::json!("server"));
        assert_eq!(value["output"][1]["call_id"], serde_json::Value::Null);
        assert_eq!(
            value["output"][1]["tools"][0]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            value["output"][2]["type"],
            serde_json::json!("tool_search_call")
        );
        assert_eq!(value["output"][2]["execution"], serde_json::json!("client"));
        assert_eq!(
            value["output"][2]["call_id"],
            serde_json::json!("call_final")
        );
        assert_eq!(
            value["output"][3]["type"],
            serde_json::json!("tool_search_output")
        );
        assert_eq!(value["output"][3]["execution"], serde_json::json!("client"));
        assert_eq!(
            value["output"][3]["call_id"],
            serde_json::json!("call_final")
        );
        assert_eq!(
            value["output"][3]["tools"][0]["name"],
            serde_json::json!("get_forecast")
        );
    }

    #[test]
    fn responses_encoder_serializes_dynamic_hosted_tool_call_items() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolCall {
                tool_call_id: "local_shell_call_1".to_string(),
                tool_name: "shell".to_string(),
                arguments: serde_json::Value::String(
                    r#"{"action":{"type":"exec","command":["ls"],"working_directory":"/root","env":{}}}"#
                        .to_string(),
                ),
                provider_executed: None,
                dynamic: Some(true),
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "lsh_1"
                    }),
                )])),
            },
            ContentPart::ToolCall {
                tool_call_id: "apply_patch_call_1".to_string(),
                tool_name: "apply_patch".to_string(),
                arguments: serde_json::Value::String(
                    r#"{"operation":{"type":"create_file","path":"notes.md","diff":"+hello\n"}}"#
                        .to_string(),
                ),
                provider_executed: None,
                dynamic: Some(true),
                invalid: None,
                error: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "apc_1"
                    }),
                )])),
            },
        ]));
        response.id = Some("resp_dynamic_hosted".to_string());
        response.model = Some("gpt-5".to_string());

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["output"][0]["type"],
            serde_json::json!("local_shell_call")
        );
        assert_eq!(value["output"][0]["id"], serde_json::json!("lsh_1"));
        assert_eq!(
            value["output"][0]["call_id"],
            serde_json::json!("local_shell_call_1")
        );
        assert_eq!(
            value["output"][0]["action"]["command"][0],
            serde_json::json!("ls")
        );
        assert_eq!(
            value["output"][1]["type"],
            serde_json::json!("apply_patch_call")
        );
        assert_eq!(value["output"][1]["id"], serde_json::json!("apc_1"));
        assert_eq!(
            value["output"][1]["operation"]["type"],
            serde_json::json!("create_file")
        );
    }

    #[test]
    fn responses_encoder_serializes_dynamic_hosted_tool_output_items() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::ToolResult {
                tool_call_id: "shell_call_1".to_string(),
                tool_name: "shell".to_string(),
                output: crate::types::ToolResultOutput::json(serde_json::json!({
                    "output": [
                        {
                            "stdout": "ok",
                            "stderr": "",
                            "outcome": {
                                "type": "exit",
                                "exitCode": 0
                            }
                        }
                    ]
                })),
                input: None,
                provider_executed: None,
                dynamic: Some(true),
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "itemId": "sh_1"
                    }),
                )])),
            },
            ContentPart::ToolResult {
                tool_call_id: "apply_patch_call_1".to_string(),
                tool_name: "apply_patch".to_string(),
                output: crate::types::ToolResultOutput::json(serde_json::json!({
                    "status": "completed",
                    "output": "patched"
                })),
                input: None,
                provider_executed: None,
                dynamic: Some(true),
                preliminary: None,
                title: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            },
        ]));
        response.id = Some("resp_dynamic_hosted_output".to_string());
        response.model = Some("gpt-5".to_string());

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["output"][0]["type"],
            serde_json::json!("shell_call_output")
        );
        assert_eq!(
            value["output"][0]["output"][0]["stdout"],
            serde_json::json!("ok")
        );
        assert_eq!(
            value["output"][1]["type"],
            serde_json::json!("apply_patch_call_output")
        );
        assert_eq!(value["output"][1]["status"], serde_json::json!("completed"));
        assert_eq!(value["output"][1]["output"], serde_json::json!("patched"));
    }

    #[test]
    fn responses_encoder_serializes_mcp_tool_approval_request_items() {
        let mut response = ChatResponse::new(crate::types::MessageContent::MultiModal(vec![
            ContentPart::tool_call(
                "id-0",
                "mcp.create_short_url",
                serde_json::json!({
                    "alias": "",
                    "description": "",
                    "max_clicks": 100,
                    "password": "",
                    "url": "https://ai-sdk.dev/"
                }),
                Some(true),
            ),
            ContentPart::tool_approval_request("mcpr_1", "id-0"),
        ]));
        response.id = Some("resp_approval".to_string());
        response.model = Some("gpt-5-mini".to_string());

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["output"].as_array().map(Vec::len), Some(1));
        assert_eq!(
            value["output"][0]["type"],
            serde_json::json!("mcp_approval_request")
        );
        assert_eq!(value["output"][0]["id"], serde_json::json!("mcpr_1"));
        assert_eq!(
            value["output"][0]["name"],
            serde_json::json!("create_short_url")
        );
        assert_eq!(
            value["output"][0]["arguments"],
            serde_json::json!(
                r#"{"alias":"","description":"","max_clicks":100,"password":"","url":"https://ai-sdk.dev/"}"#
            )
        );
    }

    #[test]
    fn chat_completions_encoder_serializes_normalized_usage_and_preserves_extensions() {
        let mut response =
            ChatResponse::new(crate::types::MessageContent::Text("hello".to_string()));
        response.model = Some("gpt-4.1".to_string());
        response.usage = Some(
            Usage::builder()
                .prompt_tokens(10)
                .completion_tokens(8)
                .total_tokens(18)
                .with_input_total_tokens(10)
                .with_input_no_cache_tokens(6)
                .with_input_cache_read_tokens(4)
                .with_output_total_tokens(8)
                .with_output_text_tokens(5)
                .with_output_reasoning_tokens(3)
                .with_raw_usage_value(serde_json::json!({
                    "citation_tokens": 7,
                    "input_tokens": 10
                }))
                .build(),
        );

        let mut out = Vec::new();
        OpenAiChatCompletionsJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["usage"]["prompt_tokens"], serde_json::json!(10));
        assert_eq!(value["usage"]["completion_tokens"], serde_json::json!(8));
        assert_eq!(
            value["usage"]["prompt_tokens_details"]["cached_tokens"],
            serde_json::json!(4)
        );
        assert_eq!(
            value["usage"]["completion_tokens_details"]["reasoning_tokens"],
            serde_json::json!(3)
        );
        assert_eq!(value["usage"]["citation_tokens"], serde_json::json!(7));
        assert!(value["usage"].get("input_tokens").is_none());
    }

    #[test]
    fn responses_encoder_serializes_responses_usage_shape_and_preserves_extensions() {
        let mut response =
            ChatResponse::new(crate::types::MessageContent::Text("hello".to_string()));
        response.model = Some("gpt-5".to_string());
        response.usage = Some(
            Usage::builder()
                .prompt_tokens(12)
                .completion_tokens(9)
                .total_tokens(21)
                .with_cached_tokens(2)
                .with_reasoning_tokens(4)
                .with_raw_usage_value(serde_json::json!({
                    "citation_tokens": 5,
                    "prompt_tokens": 999
                }))
                .build(),
        );

        let mut out = Vec::new();
        OpenAiResponsesJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["usage"]["input_tokens"], serde_json::json!(12));
        assert_eq!(value["usage"]["output_tokens"], serde_json::json!(9));
        assert_eq!(
            value["usage"]["input_tokens_details"]["cached_tokens"],
            serde_json::json!(2)
        );
        assert_eq!(
            value["usage"]["output_tokens_details"]["reasoning_tokens"],
            serde_json::json!(4)
        );
        assert_eq!(value["usage"]["citation_tokens"], serde_json::json!(5));
        assert!(value["usage"].get("prompt_tokens").is_none());
    }
}
