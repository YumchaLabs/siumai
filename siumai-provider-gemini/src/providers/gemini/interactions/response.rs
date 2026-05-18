use serde_json::{Map, Value};
use std::collections::HashSet;

use crate::LlmError;
use crate::types::{
    ChatResponse, ContentPart, FilePartSource, FinishReason, MessageContent, ProviderMetadataMap,
    ProviderOptionsMap, SourcePart, ToolResultOutput, Usage, UsageInputTokens, UsageOutputTokens,
};

const BUILTIN_TOOL_CALL_TYPES: &[&str] = &[
    "google_search_call",
    "code_execution_call",
    "url_context_call",
    "file_search_call",
    "google_maps_call",
    "mcp_server_tool_call",
];

const BUILTIN_TOOL_RESULT_TYPES: &[&str] = &[
    "google_search_result",
    "code_execution_result",
    "url_context_result",
    "file_search_result",
    "google_maps_result",
    "mcp_server_tool_result",
];

pub(crate) fn parse_interactions_response(
    raw: Value,
    generate_id: &mut dyn FnMut() -> String,
) -> Result<ChatResponse, LlmError> {
    let object = raw.as_object().ok_or_else(|| {
        LlmError::ParseError("google.interactions response must be a JSON object".to_string())
    })?;
    let status = string_field(object, "status").ok_or_else(|| {
        LlmError::ParseError("google.interactions response is missing status".to_string())
    })?;
    let interaction_id = string_field(object, "id").filter(|value| !value.is_empty());
    let service_tier = string_field(object, "service_tier").filter(|value| !value.is_empty());
    let model = string_field(object, "model").or_else(|| string_field(object, "agent"));

    let parsed = parse_outputs(
        object.get("steps").and_then(Value::as_array),
        generate_id,
        interaction_id,
    );

    let mut response = ChatResponse::new(MessageContent::MultiModal(parsed.content));
    response.id = interaction_id.map(ToOwned::to_owned);
    response.model = model.map(ToOwned::to_owned);
    response.usage = object.get("usage").and_then(convert_usage);
    response.finish_reason = Some(map_finish_reason(status, parsed.has_function_call));
    response.raw_finish_reason = Some(status.to_string());
    response.service_tier = service_tier.map(ToOwned::to_owned);
    response.provider_metadata = response_provider_metadata(interaction_id, service_tier);

    Ok(response)
}

struct ParsedOutputs {
    content: Vec<ContentPart>,
    has_function_call: bool,
}

fn parse_outputs(
    steps: Option<&Vec<Value>>,
    generate_id: &mut dyn FnMut() -> String,
    interaction_id: Option<&str>,
) -> ParsedOutputs {
    let mut content = Vec::new();
    let mut has_function_call = false;

    let Some(steps) = steps else {
        return ParsedOutputs {
            content,
            has_function_call,
        };
    };

    for step in steps {
        let Some(step_obj) = step.as_object() else {
            continue;
        };
        let Some(step_type) = string_field(step_obj, "type") else {
            continue;
        };

        match step_type {
            "user_input" => {}
            "model_output" => {
                if let Some(blocks) = step_obj.get("content").and_then(Value::as_array) {
                    parse_model_output_blocks(blocks, generate_id, interaction_id, &mut content);
                }
            }
            "thought" => {
                content.push(parse_thought_step(step_obj, interaction_id));
            }
            "function_call" => {
                has_function_call = true;
                if let Some(part) = parse_function_call_step(step_obj, interaction_id) {
                    content.push(part);
                }
            }
            other if BUILTIN_TOOL_CALL_TYPES.contains(&other) => {
                content.push(parse_builtin_tool_call_step(other, step_obj, generate_id));
            }
            other if BUILTIN_TOOL_RESULT_TYPES.contains(&other) => {
                content.push(parse_builtin_tool_result_step(other, step_obj, generate_id));
                content.extend(builtin_tool_result_to_sources(other, step_obj, generate_id));
            }
            _ => {}
        }
    }

    ParsedOutputs {
        content,
        has_function_call,
    }
}

fn parse_model_output_blocks(
    blocks: &[Value],
    generate_id: &mut dyn FnMut() -> String,
    interaction_id: Option<&str>,
    content: &mut Vec<ContentPart>,
) {
    for block in blocks {
        let Some(block_obj) = block.as_object() else {
            continue;
        };
        match string_field(block_obj, "type") {
            Some("text") => {
                let text = string_field(block_obj, "text").unwrap_or_default();
                content.push(ContentPart::Text {
                    text: text.to_string(),
                    provider_options: ProviderOptionsMap::default(),
                    provider_metadata: part_provider_metadata(None, interaction_id),
                });
                content.extend(annotations_to_sources(
                    block_obj.get("annotations").and_then(Value::as_array),
                    generate_id,
                ));
            }
            Some("image") => {
                if let Some(part) = image_block_to_file_part(block_obj, interaction_id) {
                    content.push(part);
                }
            }
            _ => {}
        }
    }
}

fn image_block_to_file_part(
    block: &Map<String, Value>,
    interaction_id: Option<&str>,
) -> Option<ContentPart> {
    let media_type = string_field(block, "mime_type")
        .filter(|value| !value.is_empty())
        .unwrap_or("image/png")
        .to_string();

    let source = if let Some(data) = string_field(block, "data").filter(|value| !value.is_empty()) {
        FilePartSource::base64(data)
    } else if let Some(uri) = string_field(block, "uri").filter(|value| !value.is_empty()) {
        FilePartSource::url(uri)
    } else {
        return None;
    };

    Some(ContentPart::File {
        source,
        media_type,
        filename: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: part_provider_metadata(None, interaction_id),
    })
}

fn parse_thought_step(step: &Map<String, Value>, interaction_id: Option<&str>) -> ContentPart {
    let text = step
        .get("summary")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let item = item.as_object()?;
                    (string_field(item, "type") == Some("text"))
                        .then(|| string_field(item, "text"))
                        .flatten()
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    ContentPart::Reasoning {
        text,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: part_provider_metadata(string_field(step, "signature"), interaction_id),
    }
}

fn parse_function_call_step(
    step: &Map<String, Value>,
    interaction_id: Option<&str>,
) -> Option<ContentPart> {
    let id = string_field(step, "id")?;
    let name = string_field(step, "name")?;
    Some(ContentPart::ToolCall {
        tool_call_id: id.to_string(),
        tool_name: name.to_string(),
        arguments: arguments_value(step.get("arguments")),
        provider_executed: None,
        dynamic: None,
        invalid: None,
        error: None,
        title: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: part_provider_metadata(string_field(step, "signature"), interaction_id),
    })
}

fn parse_builtin_tool_call_step(
    step_type: &str,
    step: &Map<String, Value>,
    generate_id: &mut dyn FnMut() -> String,
) -> ContentPart {
    let tool_name = if step_type == "mcp_server_tool_call" {
        string_field(step, "name").unwrap_or("mcp_server_tool")
    } else {
        step_type.strip_suffix("_call").unwrap_or(step_type)
    };

    ContentPart::ToolCall {
        tool_call_id: string_field(step, "id")
            .map(ToOwned::to_owned)
            .unwrap_or_else(generate_id),
        tool_name: tool_name.to_string(),
        arguments: arguments_value(step.get("arguments")),
        provider_executed: Some(true),
        dynamic: None,
        invalid: None,
        error: None,
        title: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    }
}

fn parse_builtin_tool_result_step(
    step_type: &str,
    step: &Map<String, Value>,
    generate_id: &mut dyn FnMut() -> String,
) -> ContentPart {
    let tool_name = if step_type == "mcp_server_tool_result" {
        string_field(step, "name").unwrap_or("mcp_server_tool")
    } else {
        step_type.strip_suffix("_result").unwrap_or(step_type)
    };
    let result = step.get("result").cloned().unwrap_or(Value::Null);
    let output = if step
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        ToolResultOutput::error_json(result)
    } else {
        ToolResultOutput::json(result)
    };

    ContentPart::ToolResult {
        tool_call_id: string_field(step, "call_id")
            .map(ToOwned::to_owned)
            .unwrap_or_else(generate_id),
        tool_name: tool_name.to_string(),
        output,
        input: None,
        provider_executed: Some(true),
        dynamic: None,
        preliminary: None,
        title: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    }
}

fn arguments_value(value: Option<&Value>) -> Value {
    match value {
        Some(Value::Object(_)) => value.cloned().unwrap_or_else(|| serde_json::json!({})),
        Some(Value::Null) | None => serde_json::json!({}),
        Some(other) => other.clone(),
    }
}

fn map_finish_reason(status: &str, has_function_call: bool) -> FinishReason {
    match status {
        "completed" if has_function_call => FinishReason::ToolCalls,
        "completed" => FinishReason::Stop,
        "requires_action" => FinishReason::ToolCalls,
        "failed" => FinishReason::Error,
        "incomplete" => FinishReason::Length,
        "cancelled" | "in_progress" => FinishReason::Other(status.to_string()),
        other => FinishReason::Other(other.to_string()),
    }
}

fn convert_usage(value: &Value) -> Option<Usage> {
    let usage = value.as_object()?;
    let input_total = u32_field(usage, "total_input_tokens");
    let output_text = u32_field(usage, "total_output_tokens");
    let output_reasoning = u32_field(usage, "total_thought_tokens");
    let cached = u32_field(usage, "total_cached_tokens");
    let output_total = match (output_text, output_reasoning) {
        (None, None) => None,
        (text, reasoning) => Some(text.unwrap_or(0).saturating_add(reasoning.unwrap_or(0))),
    };

    let mut builder = Usage::builder();
    if let Some(input_total) = input_total {
        builder = builder.prompt_tokens(input_total);
    }
    if let Some(output_total) = output_total {
        builder = builder.completion_tokens(output_total);
    }
    if let Some(total_tokens) = u32_field(usage, "total_tokens") {
        builder = builder.total_tokens(total_tokens);
    }
    if let Some(cached) = cached {
        builder = builder.with_cached_tokens(cached);
    }
    if let Some(reasoning) = output_reasoning {
        builder = builder.with_reasoning_tokens(reasoning);
    }

    builder = builder.with_input_tokens(UsageInputTokens {
        total: input_total,
        no_cache: input_total.map(|total| total.saturating_sub(cached.unwrap_or(0))),
        cache_read: cached,
        cache_write: None,
    });
    builder = builder.with_output_tokens(UsageOutputTokens {
        total: output_total,
        text: output_text,
        reasoning: output_reasoning,
    });
    builder = builder.with_raw_usage(usage.clone());

    Some(builder.build())
}

fn annotations_to_sources(
    annotations: Option<&Vec<Value>>,
    generate_id: &mut dyn FnMut() -> String,
) -> Vec<ContentPart> {
    let mut seen = HashSet::new();
    let mut sources = Vec::new();
    let Some(annotations) = annotations else {
        return sources;
    };

    for annotation in annotations {
        let Some(annotation) = annotation.as_object() else {
            continue;
        };
        let Some(source) = annotation_to_source(annotation, generate_id) else {
            continue;
        };
        let key = source_dedupe_key(&source);
        if seen.insert(key) {
            sources.push(source);
        }
    }

    sources
}

fn annotation_to_source(
    annotation: &Map<String, Value>,
    generate_id: &mut dyn FnMut() -> String,
) -> Option<ContentPart> {
    match string_field(annotation, "type")? {
        "url_citation" => {
            let url = string_field(annotation, "url").filter(|value| !value.is_empty())?;
            Some(source_url_part(
                generate_id(),
                url,
                string_field(annotation, "title").map(ToOwned::to_owned),
            ))
        }
        "file_citation" => {
            let uri = string_field(annotation, "url")
                .or_else(|| string_field(annotation, "document_uri"))
                .or_else(|| string_field(annotation, "file_name"))
                .filter(|value| !value.is_empty())?;
            if is_http_url(uri) {
                Some(source_url_part(
                    generate_id(),
                    uri,
                    string_field(annotation, "file_name").map(ToOwned::to_owned),
                ))
            } else {
                let filename = string_field(annotation, "file_name")
                    .map(ToOwned::to_owned)
                    .or_else(|| basename(uri).map(ToOwned::to_owned));
                Some(source_document_part(
                    generate_id(),
                    infer_doc_media_type(uri),
                    string_field(annotation, "file_name")
                        .map(ToOwned::to_owned)
                        .or_else(|| filename.clone())
                        .unwrap_or_else(|| uri.to_string()),
                    filename,
                ))
            }
        }
        "place_citation" => {
            let url = string_field(annotation, "url").filter(|value| !value.is_empty())?;
            Some(source_url_part(
                generate_id(),
                url,
                string_field(annotation, "name").map(ToOwned::to_owned),
            ))
        }
        _ => None,
    }
}

fn builtin_tool_result_to_sources(
    step_type: &str,
    step: &Map<String, Value>,
    generate_id: &mut dyn FnMut() -> String,
) -> Vec<ContentPart> {
    match step_type {
        "url_context_result" => url_context_result_to_sources(step, generate_id),
        "google_search_result" => google_search_result_to_sources(step, generate_id),
        "google_maps_result" => google_maps_result_to_sources(step, generate_id),
        "file_search_result" => file_search_result_to_sources(step, generate_id),
        _ => Vec::new(),
    }
}

fn url_context_result_to_sources(
    step: &Map<String, Value>,
    generate_id: &mut dyn FnMut() -> String,
) -> Vec<ContentPart> {
    result_entries(step)
        .filter_map(|entry| {
            let entry = entry.as_object()?;
            let url = string_field(entry, "url").filter(|value| !value.is_empty())?;
            let status = string_field(entry, "status");
            if status.is_some_and(|value| value != "success") {
                return None;
            }
            Some(source_url_part(generate_id(), url, None))
        })
        .collect()
}

fn google_search_result_to_sources(
    step: &Map<String, Value>,
    generate_id: &mut dyn FnMut() -> String,
) -> Vec<ContentPart> {
    result_entries(step)
        .filter_map(|entry| {
            let entry = entry.as_object()?;
            let url = string_field(entry, "url").filter(|value| !value.is_empty())?;
            Some(source_url_part(
                generate_id(),
                url,
                string_field(entry, "title").map(ToOwned::to_owned),
            ))
        })
        .collect()
}

fn google_maps_result_to_sources(
    step: &Map<String, Value>,
    generate_id: &mut dyn FnMut() -> String,
) -> Vec<ContentPart> {
    result_entries(step)
        .filter_map(Value::as_object)
        .flat_map(|entry| {
            entry
                .get("places")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|place| {
            let place = place.as_object()?;
            let url = string_field(place, "url").filter(|value| !value.is_empty())?;
            Some(source_url_part(
                generate_id(),
                url,
                string_field(place, "name").map(ToOwned::to_owned),
            ))
        })
        .collect()
}

fn file_search_result_to_sources(
    step: &Map<String, Value>,
    generate_id: &mut dyn FnMut() -> String,
) -> Vec<ContentPart> {
    result_entries(step)
        .filter_map(|entry| {
            let entry = entry.as_object()?;
            let uri = string_field(entry, "url")
                .or_else(|| string_field(entry, "document_uri"))
                .or_else(|| string_field(entry, "file_name"))
                .filter(|value| !value.is_empty())?;
            if is_http_url(uri) {
                return Some(source_url_part(
                    generate_id(),
                    uri,
                    string_field(entry, "title").map(ToOwned::to_owned),
                ));
            }
            let filename = string_field(entry, "file_name")
                .map(ToOwned::to_owned)
                .or_else(|| basename(uri).map(ToOwned::to_owned));
            Some(source_document_part(
                generate_id(),
                infer_doc_media_type(uri),
                string_field(entry, "title")
                    .map(ToOwned::to_owned)
                    .or_else(|| string_field(entry, "file_name").map(ToOwned::to_owned))
                    .or_else(|| filename.clone())
                    .unwrap_or_else(|| uri.to_string()),
                filename,
            ))
        })
        .collect()
}

fn result_entries(step: &Map<String, Value>) -> impl Iterator<Item = &Value> {
    step.get("result")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

fn source_url_part(id: String, url: &str, title: Option<String>) -> ContentPart {
    ContentPart::Source {
        id,
        source: SourcePart::Url {
            url: url.to_string(),
            title,
        },
        provider_metadata: None,
    }
}

fn source_document_part(
    id: String,
    media_type: &'static str,
    title: String,
    filename: Option<String>,
) -> ContentPart {
    ContentPart::Source {
        id,
        source: SourcePart::Document {
            media_type: media_type.to_string(),
            title,
            filename,
        },
        provider_metadata: None,
    }
}

fn source_dedupe_key(source: &ContentPart) -> String {
    match source {
        ContentPart::Source {
            source: SourcePart::Url { url, .. },
            ..
        } => format!("url:{url}"),
        ContentPart::Source {
            source: SourcePart::Document {
                filename, title, ..
            },
            ..
        } => format!("doc:{}", filename.as_deref().unwrap_or(title)),
        _ => String::new(),
    }
}

fn part_provider_metadata(
    signature: Option<&str>,
    interaction_id: Option<&str>,
) -> Option<ProviderMetadataMap> {
    let mut google = Map::new();
    if let Some(signature) = signature.filter(|value| !value.is_empty()) {
        google.insert(
            "signature".to_string(),
            Value::String(signature.to_string()),
        );
    }
    if let Some(interaction_id) = interaction_id.filter(|value| !value.is_empty()) {
        google.insert(
            "interactionId".to_string(),
            Value::String(interaction_id.to_string()),
        );
    }
    (!google.is_empty()).then(|| ProviderMetadataMap::from([("google".to_string(), google.into())]))
}

fn response_provider_metadata(
    interaction_id: Option<&str>,
    service_tier: Option<&str>,
) -> Option<ProviderMetadataMap> {
    let mut google = Map::new();
    if let Some(interaction_id) = interaction_id.filter(|value| !value.is_empty()) {
        google.insert(
            "interactionId".to_string(),
            Value::String(interaction_id.to_string()),
        );
    }
    if let Some(service_tier) = service_tier.filter(|value| !value.is_empty()) {
        google.insert(
            "serviceTier".to_string(),
            Value::String(service_tier.to_string()),
        );
    }
    (!google.is_empty()).then(|| ProviderMetadataMap::from([("google".to_string(), google.into())]))
}

fn string_field<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    object.get(key).and_then(Value::as_str)
}

fn u32_field(object: &Map<String, Value>, key: &str) -> Option<u32> {
    object
        .get(key)?
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
}

fn basename(uri_or_name: &str) -> Option<&str> {
    uri_or_name
        .split('/')
        .next_back()
        .filter(|value| !value.is_empty())
}

fn is_http_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

fn infer_doc_media_type(uri_or_name: &str) -> &'static str {
    let lower = uri_or_name.to_ascii_lowercase();
    if lower.ends_with(".pdf") {
        "application/pdf"
    } else if lower.ends_with(".txt") {
        "text/plain"
    } else if lower.ends_with(".md") || lower.ends_with(".markdown") {
        "text/markdown"
    } else if lower.ends_with(".doc") {
        "application/msword"
    } else if lower.ends_with(".docx") {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    } else {
        "application/octet-stream"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MediaSource, ToolResultOutput};

    fn id_generator() -> impl FnMut() -> String {
        let mut next = 0;
        move || {
            next += 1;
            format!("id-{next}")
        }
    }

    fn parse(value: Value) -> ChatResponse {
        let mut generate_id = id_generator();
        parse_interactions_response(value, &mut generate_id).expect("parse interactions response")
    }

    #[test]
    fn google_interactions_response_parses_completed_text_usage_and_metadata() {
        let response = parse(serde_json::json!({
            "id": "iact_123",
            "created": "2026-05-18T00:00:00Z",
            "status": "completed",
            "model": "gemini-2.5-flash",
            "service_tier": "priority",
            "steps": [
                {
                    "type": "user_input",
                    "content": [{ "type": "text", "text": "hi" }]
                },
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": "answer",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/a",
                                    "title": "A"
                                },
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/a",
                                    "title": "A duplicate"
                                }
                            ]
                        }
                    ]
                }
            ],
            "usage": {
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_thought_tokens": 2,
                "total_cached_tokens": 3,
                "total_tokens": 17
            }
        }));

        assert_eq!(response.id.as_deref(), Some("iact_123"));
        assert_eq!(response.model.as_deref(), Some("gemini-2.5-flash"));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.raw_finish_reason.as_deref(), Some("completed"));
        assert_eq!(response.service_tier.as_deref(), Some("priority"));
        assert_eq!(
            response.get_metadata("google", "interactionId"),
            Some(&serde_json::json!("iact_123"))
        );
        assert_eq!(
            response.get_metadata("google", "serviceTier"),
            Some(&serde_json::json!("priority"))
        );

        let usage = response.usage.expect("usage");
        assert_eq!(usage.prompt_tokens(), Some(10));
        assert_eq!(usage.completion_tokens(), Some(7));
        assert_eq!(usage.total_tokens(), Some(17));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(7));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(3));
        assert_eq!(usage.normalized_output_tokens().text, Some(5));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(2));
        assert_eq!(
            usage
                .raw_usage_value()
                .and_then(|raw| raw.get("total_input_tokens").cloned()),
            Some(serde_json::json!(10))
        );

        let MessageContent::MultiModal(parts) = response.content else {
            panic!("expected multimodal content");
        };
        assert_eq!(parts.len(), 2);
        let ContentPart::Text {
            text,
            provider_metadata,
            ..
        } = &parts[0]
        else {
            panic!("expected text part");
        };
        assert_eq!(text, "answer");
        assert_eq!(
            provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("google"))
                .and_then(|google| google.get("interactionId")),
            Some(&serde_json::json!("iact_123"))
        );
        let Some((source_id, SourcePart::Url { url, title })) = parts[1].as_source() else {
            panic!("expected url source");
        };
        assert_eq!(source_id, "id-1");
        assert_eq!(url, "https://example.com/a");
        assert_eq!(title.as_deref(), Some("A"));
    }

    #[test]
    fn google_interactions_response_maps_reasoning_signature_and_function_call() {
        let response = parse(serde_json::json!({
            "id": "iact_tool",
            "status": "completed",
            "model": "gemini-2.5-flash",
            "steps": [
                {
                    "type": "thought",
                    "signature": "sig_thought",
                    "summary": [
                        { "type": "text", "text": "plan" },
                        { "type": "text", "text": "call weather" }
                    ]
                },
                {
                    "type": "function_call",
                    "id": "call_weather",
                    "name": "weather",
                    "arguments": { "city": "Shanghai" },
                    "signature": "sig_call"
                }
            ]
        }));

        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        let MessageContent::MultiModal(parts) = response.content else {
            panic!("expected multimodal content");
        };
        assert_eq!(parts.len(), 2);

        let ContentPart::Reasoning {
            text,
            provider_metadata,
            ..
        } = &parts[0]
        else {
            panic!("expected reasoning");
        };
        assert_eq!(text, "plan\ncall weather");
        let reasoning_google = provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("google"))
            .expect("google metadata");
        assert_eq!(
            reasoning_google["signature"],
            serde_json::json!("sig_thought")
        );
        assert_eq!(
            reasoning_google["interactionId"],
            serde_json::json!("iact_tool")
        );

        let tool_call = parts[1].as_tool_call().expect("tool call");
        assert_eq!(tool_call.tool_call_id, "call_weather");
        assert_eq!(tool_call.tool_name, "weather");
        assert_eq!(
            tool_call.arguments,
            &serde_json::json!({ "city": "Shanghai" })
        );
        let ContentPart::ToolCall {
            provider_metadata, ..
        } = &parts[1]
        else {
            unreachable!();
        };
        let tool_google = provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("google"))
            .expect("google metadata");
        assert_eq!(tool_google["signature"], serde_json::json!("sig_call"));
        assert_eq!(tool_google["interactionId"], serde_json::json!("iact_tool"));
    }

    #[test]
    fn google_interactions_response_maps_images_builtin_tools_and_sources() {
        let response = parse(serde_json::json!({
            "id": "iact_media",
            "status": "completed",
            "model": "gemini-2.5-flash-image",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "data": "aGVsbG8="
                        }
                    ]
                },
                {
                    "type": "google_search_call",
                    "id": "search_1",
                    "arguments": { "query": "rust" },
                    "signature": "sig_search"
                },
                {
                    "type": "google_search_result",
                    "call_id": "search_1",
                    "result": [
                        { "url": "https://example.com/rust", "title": "Rust" },
                        { "search_suggestions": "<html></html>" }
                    ],
                    "signature": "sig_result"
                }
            ]
        }));

        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        let MessageContent::MultiModal(parts) = response.content else {
            panic!("expected multimodal content");
        };
        assert_eq!(parts.len(), 4);

        let ContentPart::File {
            source,
            media_type,
            provider_metadata,
            ..
        } = &parts[0]
        else {
            panic!("expected file part");
        };
        assert_eq!(media_type, "image/png");
        assert!(matches!(
            source.as_media_source(),
            Some(MediaSource::Base64 { data }) if data == "aGVsbG8="
        ));
        assert_eq!(
            provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("google"))
                .and_then(|google| google.get("interactionId")),
            Some(&serde_json::json!("iact_media"))
        );

        let call = parts[1].as_tool_call().expect("builtin tool call");
        assert_eq!(call.tool_call_id, "search_1");
        assert_eq!(call.tool_name, "google_search");
        assert_eq!(call.provider_executed, Some(&true));
        let ContentPart::ToolCall {
            provider_metadata, ..
        } = &parts[1]
        else {
            unreachable!();
        };
        assert!(provider_metadata.is_none());

        let result = parts[2].as_tool_result().expect("builtin tool result");
        assert_eq!(result.tool_call_id, "search_1");
        assert_eq!(result.tool_name, "google_search");
        assert_eq!(result.provider_executed, Some(&true));
        assert!(matches!(result.output, ToolResultOutput::Json { .. }));
        let ContentPart::ToolResult {
            provider_metadata, ..
        } = &parts[2]
        else {
            unreachable!();
        };
        assert!(provider_metadata.is_none());

        let Some((source_id, SourcePart::Url { url, title })) = parts[3].as_source() else {
            panic!("expected search result source");
        };
        assert_eq!(source_id, "id-1");
        assert_eq!(url, "https://example.com/rust");
        assert_eq!(title.as_deref(), Some("Rust"));
    }
}
