use super::*;
use crate::provider_metadata::anthropic::AnthropicSource;
use crate::standards::anthropic::params::AnthropicParams;
use crate::standards::anthropic::streaming::AnthropicCitationDocument;
use crate::types::SourcePart;
use std::collections::HashMap;

pub fn parse_response_content(content_blocks: &[AnthropicContentBlock]) -> MessageContent {
    // Find the first text block (skip thinking blocks for main content)
    for content_block in content_blocks {
        if content_block.r#type.as_str() == "text" {
            return MessageContent::Text(content_block.text.clone().unwrap_or_default());
        }
    }
    MessageContent::Text(String::new())
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedAnthropicResponseContent {
    pub content: MessageContent,
    pub sources: Vec<AnthropicSource>,
}

fn next_anthropic_source_id(index: &mut usize) -> String {
    let id = format!("id-{}", *index);
    *index += 1;
    id
}

fn anthropic_part_provider_metadata(
    value: serde_json::Value,
) -> Option<HashMap<String, serde_json::Value>> {
    Some(HashMap::from([("anthropic".to_string(), value)]))
}

fn create_citation_source_part(
    citation: &serde_json::Value,
    citation_documents: &[AnthropicCitationDocument],
    next_source_index: &mut usize,
) -> Option<(crate::types::ContentPart, AnthropicSource)> {
    let obj = citation.as_object()?;
    let citation_type = obj.get("type").and_then(|value| value.as_str())?;
    let id = next_anthropic_source_id(next_source_index);

    match citation_type {
        "web_search_result_location" => {
            let url = obj.get("url").and_then(|value| value.as_str())?.trim();
            if url.is_empty() {
                return None;
            }

            let title = obj
                .get("title")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
                .filter(|value| !value.is_empty());

            let mut provider_meta = serde_json::Map::new();
            provider_meta.insert(
                "citedText".to_string(),
                obj.get("cited_text")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            );
            if let Some(encrypted_index) = obj.get("encrypted_index") {
                provider_meta.insert("encryptedIndex".to_string(), encrypted_index.clone());
            }
            let provider_meta = serde_json::Value::Object(provider_meta);

            Some((
                crate::types::ContentPart::Source {
                    id: id.clone(),
                    source: SourcePart::Url {
                        url: url.to_string(),
                        title: title.clone(),
                    },
                    provider_metadata: anthropic_part_provider_metadata(provider_meta.clone()),
                },
                AnthropicSource {
                    id,
                    source_type: "url".to_string(),
                    url: Some(url.to_string()),
                    title,
                    media_type: None,
                    filename: None,
                    page_age: None,
                    encrypted_content: None,
                    tool_call_id: None,
                    provider_metadata: Some(provider_meta),
                },
            ))
        }
        "page_location" | "char_location" => {
            let document_index = obj
                .get("document_index")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)?;
            let document = citation_documents.get(document_index)?;
            let title = obj
                .get("document_title")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| document.title.clone());

            let provider_meta = match citation_type {
                "page_location" => serde_json::json!({
                    "citedText": obj.get("cited_text").cloned().unwrap_or(serde_json::Value::Null),
                    "startPageNumber": obj.get("start_page_number").cloned().unwrap_or(serde_json::Value::Null),
                    "endPageNumber": obj.get("end_page_number").cloned().unwrap_or(serde_json::Value::Null),
                }),
                "char_location" => serde_json::json!({
                    "citedText": obj.get("cited_text").cloned().unwrap_or(serde_json::Value::Null),
                    "startCharIndex": obj.get("start_char_index").cloned().unwrap_or(serde_json::Value::Null),
                    "endCharIndex": obj.get("end_char_index").cloned().unwrap_or(serde_json::Value::Null),
                }),
                _ => return None,
            };

            Some((
                crate::types::ContentPart::Source {
                    id: id.clone(),
                    source: SourcePart::Document {
                        media_type: document.media_type.clone(),
                        title: title.clone(),
                        filename: document.filename.clone(),
                    },
                    provider_metadata: anthropic_part_provider_metadata(provider_meta.clone()),
                },
                AnthropicSource {
                    id,
                    source_type: "document".to_string(),
                    url: None,
                    title: Some(title),
                    media_type: Some(document.media_type.clone()),
                    filename: document.filename.clone(),
                    page_age: None,
                    encrypted_content: None,
                    tool_call_id: None,
                    provider_metadata: Some(provider_meta),
                },
            ))
        }
        _ => None,
    }
}

fn create_web_search_source_part(
    result: &serde_json::Value,
    tool_call_id: &str,
    next_source_index: &mut usize,
) -> Option<(crate::types::ContentPart, AnthropicSource)> {
    let obj = result.as_object()?;
    let url = obj.get("url").and_then(|value| value.as_str())?.trim();
    if url.is_empty() {
        return None;
    }

    let title = obj
        .get("title")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .filter(|value| !value.is_empty());
    let page_age = obj
        .get("page_age")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned);
    let encrypted_content = obj
        .get("encrypted_content")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned);
    let provider_meta = serde_json::json!({
        "pageAge": page_age,
    });
    let id = next_anthropic_source_id(next_source_index);

    Some((
        crate::types::ContentPart::Source {
            id: id.clone(),
            source: SourcePart::Url {
                url: url.to_string(),
                title: title.clone(),
            },
            provider_metadata: anthropic_part_provider_metadata(provider_meta),
        },
        AnthropicSource {
            id,
            source_type: "url".to_string(),
            url: Some(url.to_string()),
            title,
            media_type: None,
            filename: None,
            page_age,
            encrypted_content,
            tool_call_id: Some(tool_call_id.to_string()),
            provider_metadata: None,
        },
    ))
}

pub fn parse_response_content_and_tools(
    content_blocks: &[AnthropicContentBlock],
) -> MessageContent {
    parse_response_content_and_tools_with_context_and_params(
        content_blocks,
        &[],
        &AnthropicParams::default(),
    )
    .content
}

#[cfg(test)]
pub(crate) fn parse_response_content_and_tools_with_context(
    content_blocks: &[AnthropicContentBlock],
    citation_documents: &[AnthropicCitationDocument],
) -> ParsedAnthropicResponseContent {
    parse_response_content_and_tools_with_context_and_params(
        content_blocks,
        citation_documents,
        &AnthropicParams::default(),
    )
}

pub(crate) fn parse_response_content_and_tools_with_context_and_params(
    content_blocks: &[AnthropicContentBlock],
    citation_documents: &[AnthropicCitationDocument],
    params: &AnthropicParams,
) -> ParsedAnthropicResponseContent {
    use crate::types::ContentPart;
    use crate::types::ToolResultOutput;

    let mut parts = Vec::new();
    let mut sources = Vec::new();
    let mut next_source_index = 0usize;
    let mut tool_names_by_id: HashMap<String, String> = HashMap::new();

    fn text_part_provider_metadata(
        citations: Option<&Vec<serde_json::Value>>,
    ) -> Option<HashMap<String, serde_json::Value>> {
        let citations = citations.filter(|citations| !citations.is_empty())?;
        Some(HashMap::from([(
            "anthropic".to_string(),
            serde_json::json!({
                "citations": citations
            }),
        )]))
    }

    fn reasoning_part_provider_metadata(
        signature: Option<&str>,
        redacted_data: Option<&str>,
    ) -> Option<HashMap<String, serde_json::Value>> {
        let mut anthropic = serde_json::Map::new();

        if let Some(signature) = signature {
            anthropic.insert("signature".to_string(), serde_json::json!(signature));
        }
        if let Some(redacted_data) = redacted_data {
            anthropic.insert("redactedData".to_string(), serde_json::json!(redacted_data));
        }

        (!anthropic.is_empty()).then(|| {
            HashMap::from([(
                "anthropic".to_string(),
                serde_json::Value::Object(anthropic),
            )])
        })
    }

    for content_block in content_blocks {
        match content_block.r#type.as_str() {
            "thinking" => {
                let text = content_block.thinking.clone().unwrap_or_default();
                let provider_metadata =
                    reasoning_part_provider_metadata(content_block.signature.as_deref(), None);

                if !text.is_empty() || provider_metadata.is_some() {
                    parts.push(ContentPart::Reasoning {
                        text,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                }
            }
            "redacted_thinking" => {
                let provider_metadata =
                    reasoning_part_provider_metadata(None, content_block.data.as_deref());

                if provider_metadata.is_some() {
                    parts.push(ContentPart::Reasoning {
                        text: String::new(),
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                }
            }
            "text" => {
                let text = content_block.text.clone().unwrap_or_default();
                let provider_metadata =
                    text_part_provider_metadata(content_block.citations.as_ref());

                if !text.is_empty() || provider_metadata.is_some() {
                    parts.push(ContentPart::Text {
                        text,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                }

                if let Some(citations) = content_block.citations.as_ref() {
                    for citation in citations {
                        if let Some((source_part, source_meta)) = create_citation_source_part(
                            citation,
                            citation_documents,
                            &mut next_source_index,
                        ) {
                            parts.push(source_part);
                            sources.push(source_meta);
                        }
                    }
                }
            }
            "tool_use" => {
                // Add tool call
                if let (Some(id), Some(name), Some(input)) =
                    (&content_block.id, &content_block.name, &content_block.input)
                {
                    tool_names_by_id.insert(id.clone(), name.clone());
                    let provider_metadata = content_block.caller.as_ref().map(|caller| {
                        let mut anthropic = serde_json::Map::new();
                        anthropic.insert("caller".to_string(), caller.clone());

                        let mut all = HashMap::new();
                        all.insert(
                            "anthropic".to_string(),
                            serde_json::Value::Object(anthropic),
                        );
                        all
                    });

                    parts.push(ContentPart::ToolCall {
                        tool_call_id: id.clone(),
                        tool_name: name.clone(),
                        arguments: input.clone(),
                        provider_executed: None,
                        dynamic: None,
                        invalid: None,
                        error: None,
                        title: None,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                }
            }
            "server_tool_use" => {
                // Provider-hosted tool call (e.g. web_search)
                if let (Some(id), Some(name), Some(input)) =
                    (&content_block.id, &content_block.name, &content_block.input)
                {
                    let raw_tool_name = name.clone();
                    let tool_name =
                        server_tools::normalize_server_tool_name(raw_tool_name.as_str())
                            .to_string();
                    let input = server_tools::normalize_server_tool_input(
                        raw_tool_name.as_str(),
                        input.clone(),
                    );
                    tool_names_by_id.insert(id.clone(), tool_name.clone());

                    let mut anthropic_meta = serde_json::Map::new();
                    if let Some(caller) = &content_block.caller {
                        anthropic_meta.insert("caller".to_string(), caller.clone());
                    }
                    if raw_tool_name != tool_name {
                        anthropic_meta.insert(
                            "serverToolName".to_string(),
                            serde_json::Value::String(raw_tool_name.clone()),
                        );
                    }
                    let provider_metadata = (!anthropic_meta.is_empty()).then(|| {
                        HashMap::from([(
                            "anthropic".to_string(),
                            serde_json::Value::Object(anthropic_meta),
                        )])
                    });

                    parts.push(ContentPart::ToolCall {
                        tool_call_id: id.clone(),
                        tool_name,
                        arguments: input,
                        provider_executed: Some(true),
                        dynamic: (params.should_mark_code_execution_dynamic()
                            && raw_tool_name == "code_execution")
                            .then_some(true),
                        invalid: None,
                        error: None,
                        title: None,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                }
            }
            "mcp_tool_use" => {
                // Provider-hosted MCP tool call
                if let (Some(id), Some(name), Some(input)) =
                    (&content_block.id, &content_block.name, &content_block.input)
                {
                    tool_names_by_id.insert(id.clone(), name.clone());
                    let provider_metadata = content_block.server_name.as_ref().map(|server_name| {
                        HashMap::from([(
                            "anthropic".to_string(),
                            serde_json::json!({
                                "serverName": server_name
                            }),
                        )])
                    });
                    parts.push(ContentPart::ToolCall {
                        tool_call_id: id.clone(),
                        tool_name: name.clone(),
                        arguments: input.clone(),
                        provider_executed: Some(true),
                        dynamic: None,
                        invalid: None,
                        error: None,
                        title: None,
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                }
            }
            block_type if block_type.ends_with("_tool_result") => {
                let Some(tool_use_id) = &content_block.tool_use_id else {
                    continue;
                };
                let Some(content) = &content_block.content else {
                    continue;
                };

                let tool_name = if block_type == "mcp_tool_result" {
                    tool_names_by_id
                        .get(tool_use_id)
                        .cloned()
                        .or_else(|| content_block.server_name.clone())
                        .unwrap_or_else(|| "mcp".to_string())
                } else {
                    server_tools::normalize_server_tool_result_name(block_type).to_string()
                };

                let output = if block_type == "mcp_tool_result" {
                    let mut out_parts: Vec<crate::types::ToolResultContentPart> = Vec::new();
                    if let Some(arr) = content.as_array() {
                        for item in arr {
                            let Some(obj) = item.as_object() else {
                                continue;
                            };
                            let t = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if t == "text"
                                && let Some(text) = obj.get("text").and_then(|v| v.as_str())
                            {
                                out_parts.push(crate::types::ToolResultContentPart::Text {
                                    text: text.to_string(),
                                    provider_options: crate::types::ProviderOptionsMap::default(),
                                });
                            }
                        }
                    }

                    if out_parts.is_empty() {
                        ToolResultOutput::json(content.clone())
                    } else {
                        ToolResultOutput::content(out_parts)
                    }
                } else if let Some((result, is_error)) =
                    server_tools::normalize_server_tool_result(block_type, content)
                {
                    if is_error {
                        ToolResultOutput::error_json(result)
                    } else {
                        ToolResultOutput::json(result)
                    }
                } else {
                    let inferred_error = content_block.is_error.unwrap_or(false)
                        || content.get("error_code").is_some_and(|v| !v.is_null());

                    match (inferred_error, content) {
                        (true, serde_json::Value::String(s)) => {
                            ToolResultOutput::error_text(s.clone())
                        }
                        (true, other) => ToolResultOutput::error_json(other.clone()),
                        (false, serde_json::Value::String(s)) => ToolResultOutput::text(s.clone()),
                        (false, other) => ToolResultOutput::json(other.clone()),
                    }
                };

                parts.push(ContentPart::ToolResult {
                    tool_call_id: tool_use_id.clone(),
                    tool_name,
                    output,
                    input: None,
                    provider_executed: Some(true),
                    dynamic: None,
                    preliminary: None,
                    title: None,
                    provider_options: crate::types::ProviderOptionsMap::default(),
                    provider_metadata: None,
                });

                if block_type == "web_search_tool_result"
                    && let Some(items) = content.as_array()
                {
                    for item in items {
                        if let Some((source_part, source_meta)) =
                            create_web_search_source_part(item, tool_use_id, &mut next_source_index)
                        {
                            parts.push(source_part);
                            sources.push(source_meta);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Return appropriate content type
    let content = if parts.is_empty() {
        MessageContent::Text(String::new())
    } else if let [
        ContentPart::Text {
            text,
            provider_metadata: None,
            ..
        },
    ] = parts.as_slice()
    {
        MessageContent::Text(text.clone())
    } else {
        MessageContent::MultiModal(parts)
    };

    ParsedAnthropicResponseContent { content, sources }
}

pub fn extract_thinking_content(content_blocks: &[AnthropicContentBlock]) -> Option<String> {
    for content_block in content_blocks {
        if content_block.r#type == "thinking" {
            return content_block.thinking.clone();
        }
    }
    None
}

fn raw_usage_object_from_response(
    usage: &AnthropicUsage,
) -> serde_json::Map<String, serde_json::Value> {
    let mut raw_usage = usage.extra.clone();
    raw_usage.insert(
        "input_tokens".to_string(),
        serde_json::json!(usage.input_tokens),
    );
    raw_usage.insert(
        "output_tokens".to_string(),
        serde_json::json!(usage.output_tokens),
    );

    if let Some(cache_creation_input_tokens) = usage.cache_creation_input_tokens {
        raw_usage.insert(
            "cache_creation_input_tokens".to_string(),
            serde_json::json!(cache_creation_input_tokens),
        );
    }
    if let Some(cache_read_input_tokens) = usage.cache_read_input_tokens {
        raw_usage.insert(
            "cache_read_input_tokens".to_string(),
            serde_json::json!(cache_read_input_tokens),
        );
    }
    if let Some(service_tier) = usage.service_tier.clone() {
        raw_usage.insert("service_tier".to_string(), serde_json::json!(service_tier));
    }
    if let Some(server_tool_use) = usage.server_tool_use.as_ref()
        && let Ok(value) = serde_json::to_value(server_tool_use)
    {
        raw_usage.insert("server_tool_use".to_string(), value);
    }
    if let Some(iterations) = usage.iterations.as_ref()
        && let Ok(value) = serde_json::to_value(iterations)
    {
        raw_usage.insert("iterations".to_string(), value);
    }

    raw_usage
}

pub fn create_usage_from_response(usage: Option<AnthropicUsage>) -> Option<Usage> {
    usage.map(|u| {
        let raw_usage = raw_usage_object_from_response(&u);

        let mut builder = Usage::builder()
            .prompt_tokens(u.input_tokens)
            .completion_tokens(u.output_tokens)
            .total_tokens(u.input_tokens.saturating_add(u.output_tokens))
            .with_input_tokens(crate::types::UsageInputTokens {
                total: Some(
                    u.input_tokens
                        .saturating_add(u.cache_read_input_tokens.unwrap_or(0))
                        .saturating_add(u.cache_creation_input_tokens.unwrap_or(0)),
                ),
                no_cache: Some(u.input_tokens),
                cache_read: u.cache_read_input_tokens,
                cache_write: u.cache_creation_input_tokens,
            })
            .with_output_tokens(crate::types::UsageOutputTokens {
                total: Some(u.output_tokens),
                text: Some(u.output_tokens),
                reasoning: None,
            })
            .with_raw_usage_value(serde_json::Value::Object(raw_usage));

        if let Some(cached) = u.cache_read_input_tokens {
            builder = builder.with_cached_tokens(cached);
        }

        builder.build()
    })
}

pub fn create_usage_from_json_value(usage: Option<&serde_json::Value>) -> Option<Usage> {
    let usage = usage?;
    let usage_obj = usage.as_object()?.clone();
    let parsed =
        serde_json::from_value::<AnthropicUsage>(serde_json::Value::Object(usage_obj.clone()))
            .ok()?;

    let mut usage = create_usage_from_response(Some(parsed))?;
    usage.raw = Some(usage_obj);
    Some(usage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MessageContent;

    #[test]
    fn test_parse_response_content_and_tools() {
        let content_blocks = vec![
            AnthropicContentBlock {
                r#type: "text".to_string(),
                text: Some("I'll help you get the weather.".to_string()),
                thinking: None,
                signature: None,
                data: None,
                id: None,
                name: None,
                input: None,
                caller: None,
                server_name: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                citations: None,
            },
            AnthropicContentBlock {
                r#type: "tool_use".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: Some("toolu_123".to_string()),
                name: Some("get_weather".to_string()),
                input: Some(serde_json::json!({"location": "San Francisco"})),
                caller: None,
                server_name: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                citations: None,
            },
        ];

        let content = parse_response_content_and_tools(&content_blocks);

        // Check content - should be multimodal with text and tool call
        match &content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 2);

                // First part should be text
                if let ContentPart::Text { text, .. } = &parts[0] {
                    assert_eq!(text, "I'll help you get the weather.");
                } else {
                    panic!("Expected text content part");
                }

                // Second part should be tool call
                if let ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    ..
                } = &parts[1]
                {
                    assert_eq!(tool_call_id, "toolu_123");
                    assert_eq!(tool_name, "get_weather");
                    assert_eq!(arguments, &serde_json::json!({"location": "San Francisco"}));
                } else {
                    panic!("Expected tool call content part");
                }
            }
            _ => panic!("Expected multimodal content"),
        }
    }

    #[test]
    fn test_parse_response_content_and_tools_text_only() {
        let content_blocks = vec![AnthropicContentBlock {
            r#type: "text".to_string(),
            text: Some("Hello world".to_string()),
            thinking: None,
            signature: None,
            data: None,
            id: None,
            name: None,
            input: None,
            caller: None,
            server_name: None,
            tool_use_id: None,
            content: None,
            is_error: None,
            citations: None,
        }];

        let content = parse_response_content_and_tools(&content_blocks);

        // Check content - should be simple text
        match content {
            MessageContent::Text(text) => assert_eq!(text, "Hello world"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_parse_response_content_and_tools_preserves_text_block_citations() {
        let content_blocks = vec![
            AnthropicContentBlock {
                r#type: "text".to_string(),
                text: Some("Overview".to_string()),
                thinking: None,
                signature: None,
                data: None,
                id: None,
                name: None,
                input: None,
                caller: None,
                server_name: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                citations: None,
            },
            AnthropicContentBlock {
                r#type: "text".to_string(),
                text: Some("Grounded fact".to_string()),
                thinking: None,
                signature: None,
                data: None,
                id: None,
                name: None,
                input: None,
                caller: None,
                server_name: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                citations: Some(vec![serde_json::json!({
                    "type": "web_search_result_location",
                    "url": "https://example.com",
                    "title": "Example"
                })]),
            },
        ];

        let content = parse_response_content_and_tools(&content_blocks);

        match &content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 3);

                if let ContentPart::Text {
                    text,
                    provider_metadata,
                    ..
                } = &parts[0]
                {
                    assert_eq!(text, "Overview");
                    assert!(provider_metadata.is_none());
                } else {
                    panic!("Expected first text content part");
                }

                if let ContentPart::Text {
                    text,
                    provider_metadata,
                    ..
                } = &parts[1]
                {
                    assert_eq!(text, "Grounded fact");
                    let citations = provider_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("anthropic"))
                        .and_then(|metadata| metadata.get("citations"))
                        .and_then(serde_json::Value::as_array)
                        .expect("citations array");
                    assert_eq!(citations.len(), 1);
                    assert_eq!(
                        citations[0]["url"],
                        serde_json::json!("https://example.com")
                    );
                } else {
                    panic!("Expected second text content part");
                }

                if let ContentPart::Source {
                    id,
                    source,
                    provider_metadata,
                } = &parts[2]
                {
                    assert_eq!(id, "id-0");
                    assert_eq!(
                        source,
                        &crate::types::SourcePart::Url {
                            url: "https://example.com".to_string(),
                            title: Some("Example".to_string()),
                        }
                    );
                    assert_eq!(
                        provider_metadata
                            .as_ref()
                            .and_then(|metadata| metadata.get("anthropic"))
                            .and_then(|metadata| metadata.get("citedText")),
                        Some(&serde_json::Value::Null)
                    );
                } else {
                    panic!("Expected third source content part");
                }
            }
            _ => panic!("Expected multimodal content"),
        }
    }

    #[test]
    fn test_parse_response_content_and_tools_with_context_emits_document_source_parts() {
        let content_blocks = vec![AnthropicContentBlock {
            r#type: "text".to_string(),
            text: Some("Grounded fact".to_string()),
            thinking: None,
            signature: None,
            data: None,
            id: None,
            name: None,
            input: None,
            caller: None,
            server_name: None,
            tool_use_id: None,
            content: None,
            is_error: None,
            citations: Some(vec![serde_json::json!({
                "type": "page_location",
                "cited_text": "Revenue increased by 25%",
                "document_index": 0,
                "document_title": null,
                "start_page_number": 5,
                "end_page_number": 6,
            })]),
        }];

        let parsed = parse_response_content_and_tools_with_context(
            &content_blocks,
            &[AnthropicCitationDocument {
                title: "Financial Report 2023".to_string(),
                filename: Some("financial-report.pdf".to_string()),
                media_type: "application/pdf".to_string(),
            }],
        );

        match &parsed.content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 2);
                if let ContentPart::Source {
                    id,
                    source,
                    provider_metadata,
                } = &parts[1]
                {
                    assert_eq!(id, "id-0");
                    assert_eq!(
                        source,
                        &crate::types::SourcePart::Document {
                            media_type: "application/pdf".to_string(),
                            title: "Financial Report 2023".to_string(),
                            filename: Some("financial-report.pdf".to_string()),
                        }
                    );
                    assert_eq!(
                        provider_metadata
                            .as_ref()
                            .and_then(|metadata| metadata.get("anthropic"))
                            .and_then(|metadata| metadata.get("startPageNumber")),
                        Some(&serde_json::json!(5))
                    );
                } else {
                    panic!("Expected document source content part");
                }
            }
            _ => panic!("Expected multimodal content"),
        }

        assert_eq!(parsed.sources.len(), 1);
        assert_eq!(parsed.sources[0].id, "id-0");
        assert_eq!(parsed.sources[0].source_type, "document");
        assert_eq!(
            parsed.sources[0].media_type.as_deref(),
            Some("application/pdf")
        );
    }

    #[test]
    fn test_parse_response_content_and_tools_server_web_search() {
        let content_blocks = vec![
            AnthropicContentBlock {
                r#type: "server_tool_use".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: Some("srvtoolu_1".to_string()),
                name: Some("web_search".to_string()),
                input: Some(serde_json::json!({"query": "rust 1.85"})),
                caller: None,
                server_name: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                citations: None,
            },
            AnthropicContentBlock {
                r#type: "web_search_tool_result".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: None,
                name: None,
                input: None,
                caller: None,
                server_name: None,
                tool_use_id: Some("srvtoolu_1".to_string()),
                content: Some(serde_json::json!([
                    {
                        "type": "web_search_result",
                        "title": "Rust 1.85.0",
                        "url": "https://blog.rust-lang.org/",
                        "encrypted_content": "..."
                    }
                ])),
                is_error: None,
                citations: None,
            },
            AnthropicContentBlock {
                r#type: "text".to_string(),
                text: Some("Here is what I found.".to_string()),
                thinking: None,
                signature: None,
                data: None,
                id: None,
                name: None,
                input: None,
                caller: None,
                server_name: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                citations: None,
            },
        ];

        let content = parse_response_content_and_tools(&content_blocks);

        match &content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 4);

                // provider-hosted tool call
                if let ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    provider_executed,
                    ..
                } = &parts[0]
                {
                    assert_eq!(tool_call_id, "srvtoolu_1");
                    assert_eq!(tool_name, "web_search");
                    assert_eq!(arguments, &serde_json::json!({"query": "rust 1.85"}));
                    assert_eq!(*provider_executed, Some(true));
                } else {
                    panic!("Expected tool call content part");
                }

                // provider-hosted tool result
                if let ContentPart::ToolResult {
                    tool_call_id,
                    tool_name,
                    output,
                    provider_executed,
                    ..
                } = &parts[1]
                {
                    assert_eq!(tool_call_id, "srvtoolu_1");
                    assert_eq!(tool_name, "web_search");
                    assert_eq!(*provider_executed, Some(true));

                    match output {
                        crate::types::ToolResultOutput::Json { value, .. } => {
                            assert!(value.is_array());
                            assert_eq!(value[0]["pageAge"], serde_json::Value::Null,);
                            assert_eq!(value[0]["encryptedContent"], serde_json::json!("..."),);
                            assert!(value[0].get("page_age").is_none());
                            assert!(value[0].get("encrypted_content").is_none());
                        }
                        other => panic!("Expected JSON output, got {:?}", other),
                    }
                } else {
                    panic!("Expected tool result content part");
                }

                if let ContentPart::Source {
                    id,
                    source,
                    provider_metadata,
                } = &parts[2]
                {
                    assert_eq!(id, "id-0");
                    assert_eq!(
                        source,
                        &crate::types::SourcePart::Url {
                            url: "https://blog.rust-lang.org/".to_string(),
                            title: Some("Rust 1.85.0".to_string()),
                        }
                    );
                    assert!(
                        provider_metadata
                            .as_ref()
                            .and_then(|metadata| metadata.get("anthropic"))
                            .and_then(|metadata| metadata.get("pageAge"))
                            .is_some_and(serde_json::Value::is_null)
                    );
                } else {
                    panic!("Expected source content part");
                }

                // text content
                if let ContentPart::Text { text, .. } = &parts[3] {
                    assert_eq!(text, "Here is what I found.");
                } else {
                    panic!("Expected text content part");
                }
            }
            _ => panic!("Expected multimodal content"),
        }
    }

    #[test]
    fn test_parse_response_content_and_tools_tool_search_normalization() {
        let content_blocks = vec![
            AnthropicContentBlock {
                r#type: "server_tool_use".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: Some("srvtoolu_2".to_string()),
                name: Some("tool_search_tool_regex".to_string()),
                input: Some(serde_json::json!({"pattern": "weather", "limit": 2})),
                caller: Some(serde_json::json!({ "type": "direct" })),
                server_name: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                citations: None,
            },
            AnthropicContentBlock {
                r#type: "tool_search_tool_result".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: None,
                name: None,
                input: None,
                caller: None,
                server_name: None,
                tool_use_id: Some("srvtoolu_2".to_string()),
                content: Some(serde_json::json!({
                    "type": "tool_search_tool_search_result",
                    "tool_references": [{"type":"tool_reference","tool_name":"get_weather"}]
                })),
                is_error: None,
                citations: None,
            },
        ];

        let content = parse_response_content_and_tools(&content_blocks);
        match &content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 2);

                if let ContentPart::ToolCall {
                    tool_name,
                    provider_metadata,
                    ..
                } = &parts[0]
                {
                    assert_eq!(tool_name, "tool_search");
                    let anthropic = provider_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("anthropic"))
                        .expect("anthropic provider metadata");
                    assert_eq!(
                        anthropic.get("serverToolName"),
                        Some(&serde_json::json!("tool_search_tool_regex"))
                    );
                    assert_eq!(
                        anthropic.get("caller").and_then(|value| value.get("type")),
                        Some(&serde_json::json!("direct"))
                    );
                } else {
                    panic!("Expected tool call part");
                }

                if let ContentPart::ToolResult {
                    tool_name, output, ..
                } = &parts[1]
                {
                    assert_eq!(tool_name, "tool_search");
                    match output {
                        crate::types::ToolResultOutput::Json { value, .. } => {
                            assert!(value.is_array());
                            assert_eq!(value[0]["toolName"], serde_json::json!("get_weather"));
                        }
                        other => panic!("Expected JSON output, got {:?}", other),
                    }
                } else {
                    panic!("Expected tool result part");
                }
            }
            _ => panic!("Expected multimodal content"),
        }
    }

    #[test]
    fn test_parse_response_content_and_tools_web_fetch_normalization() {
        let content_blocks = vec![
            AnthropicContentBlock {
                r#type: "server_tool_use".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: Some("srvtoolu_4".to_string()),
                name: Some("web_fetch".to_string()),
                input: Some(serde_json::json!({"url": "https://example.com"})),
                caller: None,
                server_name: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                citations: None,
            },
            AnthropicContentBlock {
                r#type: "web_fetch_tool_result".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: None,
                name: None,
                input: None,
                caller: None,
                server_name: None,
                tool_use_id: Some("srvtoolu_4".to_string()),
                content: Some(serde_json::json!({
                    "type": "web_fetch_result",
                    "url": "https://example.com",
                    "retrieved_at": "2025-01-01T00:00:00Z",
                    "content": {
                        "type": "document",
                        "title": "Example",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "hello"
                        }
                    }
                })),
                is_error: None,
                citations: None,
            },
        ];

        let content = parse_response_content_and_tools(&content_blocks);
        match &content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 2);

                if let ContentPart::ToolResult {
                    tool_name, output, ..
                } = &parts[1]
                {
                    assert_eq!(tool_name, "web_fetch");
                    match output {
                        crate::types::ToolResultOutput::Json { value, .. } => {
                            assert_eq!(value["type"], serde_json::json!("web_fetch_result"));
                            assert_eq!(
                                value["retrievedAt"],
                                serde_json::json!("2025-01-01T00:00:00Z")
                            );
                            assert_eq!(
                                value["content"]["source"]["mediaType"],
                                serde_json::json!("text/plain")
                            );
                        }
                        other => panic!("Expected JSON output, got {:?}", other),
                    }
                } else {
                    panic!("Expected tool result part");
                }
            }
            _ => panic!("Expected multimodal content"),
        }
    }

    #[test]
    fn test_parse_response_content_and_tools_code_execution_normalization() {
        let content_blocks = vec![
            AnthropicContentBlock {
                r#type: "server_tool_use".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: Some("srvtoolu_3".to_string()),
                name: Some("code_execution".to_string()),
                input: Some(serde_json::json!({"code": "print(1+1)"})),
                caller: None,
                server_name: None,
                tool_use_id: None,
                content: None,
                is_error: None,
                citations: None,
            },
            AnthropicContentBlock {
                r#type: "code_execution_tool_result".to_string(),
                text: None,
                thinking: None,
                signature: None,
                data: None,
                id: None,
                name: None,
                input: None,
                caller: None,
                server_name: None,
                tool_use_id: Some("srvtoolu_3".to_string()),
                content: Some(serde_json::json!({
                    "type": "code_execution_result",
                    "stdout": "2\n",
                    "stderr": "",
                    "return_code": 0
                })),
                is_error: None,
                citations: None,
            },
        ];

        let content = parse_response_content_and_tools(&content_blocks);
        match &content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 2);

                if let ContentPart::ToolCall { tool_name, .. } = &parts[0] {
                    assert_eq!(tool_name, "code_execution");
                } else {
                    panic!("Expected tool call part");
                }

                if let ContentPart::ToolResult {
                    tool_name, output, ..
                } = &parts[1]
                {
                    assert_eq!(tool_name, "code_execution");
                    match output {
                        crate::types::ToolResultOutput::Json { value, .. } => {
                            assert_eq!(value["type"], serde_json::json!("code_execution_result"));
                            assert_eq!(value["return_code"], serde_json::json!(0));
                        }
                        other => panic!("Expected JSON output, got {:?}", other),
                    }
                } else {
                    panic!("Expected tool result part");
                }
            }
            _ => panic!("Expected multimodal content"),
        }
    }

    #[test]
    fn parse_response_marks_code_execution_dynamic_for_2026_web_tool_injection() {
        let content_blocks = vec![AnthropicContentBlock {
            r#type: "server_tool_use".to_string(),
            text: None,
            thinking: None,
            signature: None,
            data: None,
            id: Some("srvtoolu_dyn".to_string()),
            name: Some("code_execution".to_string()),
            input: Some(serde_json::json!({"code": "print(1)"})),
            caller: None,
            server_name: None,
            tool_use_id: None,
            content: None,
            is_error: None,
            citations: None,
        }];

        let parsed = parse_response_content_and_tools_with_context_and_params(
            &content_blocks,
            &[],
            &AnthropicParams::default()
                .with_tools(&[crate::tools::anthropic::web_fetch_20260209()]),
        );

        let MessageContent::MultiModal(parts) = &parsed.content else {
            panic!("Expected multimodal content");
        };
        match &parts[0] {
            ContentPart::ToolCall { dynamic, .. } => assert_eq!(*dynamic, Some(true)),
            other => panic!("Expected tool call part, got {:?}", other),
        }
    }

    #[test]
    fn parse_response_keeps_code_execution_static_when_explicit_tool_exists() {
        let content_blocks = vec![AnthropicContentBlock {
            r#type: "server_tool_use".to_string(),
            text: None,
            thinking: None,
            signature: None,
            data: None,
            id: Some("srvtoolu_static".to_string()),
            name: Some("code_execution".to_string()),
            input: Some(serde_json::json!({"code": "print(1)"})),
            caller: None,
            server_name: None,
            tool_use_id: None,
            content: None,
            is_error: None,
            citations: None,
        }];

        let parsed = parse_response_content_and_tools_with_context_and_params(
            &content_blocks,
            &[],
            &AnthropicParams::default().with_tools(&[
                crate::tools::anthropic::web_search_20260209(),
                crate::tools::anthropic::code_execution_20260120(),
            ]),
        );

        let MessageContent::MultiModal(parts) = &parsed.content else {
            panic!("Expected multimodal content");
        };
        match &parts[0] {
            ContentPart::ToolCall { dynamic, .. } => assert_eq!(*dynamic, None),
            other => panic!("Expected tool call part, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_response_content_and_tools_mcp_server_name_metadata() {
        let content_blocks = vec![AnthropicContentBlock {
            r#type: "mcp_tool_use".to_string(),
            text: None,
            thinking: None,
            signature: None,
            data: None,
            id: Some("mcptoolu_1".to_string()),
            name: Some("echo".to_string()),
            input: Some(serde_json::json!({"message": "hello"})),
            caller: None,
            server_name: Some("echo-prod".to_string()),
            tool_use_id: None,
            content: None,
            is_error: None,
            citations: None,
        }];

        let content = parse_response_content_and_tools(&content_blocks);
        match &content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 1);
                if let ContentPart::ToolCall {
                    tool_name,
                    provider_metadata,
                    ..
                } = &parts[0]
                {
                    assert_eq!(tool_name, "echo");
                    let anthropic = provider_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("anthropic"))
                        .expect("anthropic provider metadata");
                    assert_eq!(
                        anthropic.get("serverName"),
                        Some(&serde_json::json!("echo-prod"))
                    );
                } else {
                    panic!("Expected tool call part");
                }
            }
            _ => panic!("Expected multimodal content"),
        }
    }

    #[test]
    fn test_create_usage_from_json_value_preserves_full_raw_usage_object() {
        let raw_usage = serde_json::json!({
            "input_tokens": 843,
            "output_tokens": 28,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 0
            },
            "service_tier": "standard"
        });

        let usage = create_usage_from_json_value(Some(&raw_usage)).expect("usage");

        assert_eq!(usage.raw_usage_value(), Some(raw_usage));
        assert_eq!(usage.normalized_input_tokens().total, Some(843));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(0));
        assert_eq!(usage.normalized_output_tokens().total, Some(28));
    }
}
