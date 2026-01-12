use super::*;

pub fn parse_response_content(content_blocks: &[AnthropicContentBlock]) -> MessageContent {
    // Find the first text block (skip thinking blocks for main content)
    for content_block in content_blocks {
        if content_block.r#type.as_str() == "text" {
            return MessageContent::Text(content_block.text.clone().unwrap_or_default());
        }
    }
    MessageContent::Text(String::new())
}

pub fn parse_response_content_and_tools(
    content_blocks: &[AnthropicContentBlock],
) -> MessageContent {
    use crate::types::ContentPart;
    use crate::types::ToolResultOutput;
    use std::collections::HashMap;

    let mut parts = Vec::new();
    let mut text_content = String::new();
    let mut tool_names_by_id: HashMap<String, String> = HashMap::new();

    for content_block in content_blocks {
        match content_block.r#type.as_str() {
            "text" => {
                if let Some(text) = &content_block.text {
                    if !text_content.is_empty() {
                        text_content.push('\n');
                    }
                    text_content.push_str(text);
                }
            }
            "tool_use" => {
                // First, add accumulated text if any
                if !text_content.is_empty() {
                    parts.push(ContentPart::text(&text_content));
                    text_content.clear();
                }

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
                        provider_metadata,
                    });
                }
            }
            "server_tool_use" => {
                // First, add accumulated text if any
                if !text_content.is_empty() {
                    parts.push(ContentPart::text(&text_content));
                    text_content.clear();
                }

                // Provider-hosted tool call (e.g. web_search)
                if let (Some(id), Some(name), Some(input)) =
                    (&content_block.id, &content_block.name, &content_block.input)
                {
                    let tool_name =
                        server_tools::normalize_server_tool_name(name.as_str()).to_string();
                    let input =
                        server_tools::normalize_server_tool_input(name.as_str(), input.clone());
                    tool_names_by_id.insert(id.clone(), tool_name.clone());
                    parts.push(ContentPart::tool_call(
                        id.clone(),
                        tool_name,
                        input,
                        Some(true),
                    ));
                }
            }
            "mcp_tool_use" => {
                // First, add accumulated text if any
                if !text_content.is_empty() {
                    parts.push(ContentPart::text(&text_content));
                    text_content.clear();
                }

                // Provider-hosted MCP tool call
                if let (Some(id), Some(name), Some(input)) =
                    (&content_block.id, &content_block.name, &content_block.input)
                {
                    tool_names_by_id.insert(id.clone(), name.clone());
                    parts.push(ContentPart::tool_call(
                        id.clone(),
                        name.clone(),
                        input.clone(),
                        Some(true),
                    ));
                }
            }
            block_type if block_type.ends_with("_tool_result") => {
                // First, add accumulated text if any
                if !text_content.is_empty() {
                    parts.push(ContentPart::text(&text_content));
                    text_content.clear();
                }

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
                    match block_type {
                        "tool_search_tool_result" => "tool_search".to_string(),
                        "text_editor_code_execution_tool_result"
                        | "bash_code_execution_tool_result" => "code_execution".to_string(),
                        _ => block_type
                            .strip_suffix("_tool_result")
                            .unwrap_or(block_type)
                            .to_string(),
                    }
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
                                });
                            }
                        }
                    }

                    if out_parts.is_empty() {
                        ToolResultOutput::json(content.clone())
                    } else {
                        ToolResultOutput::content(out_parts)
                    }
                } else if block_type == "tool_search_tool_result" {
                    if let Some(obj) = content.as_object() {
                        let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if tpe == "tool_search_tool_search_result" {
                            let refs = obj
                                .get("tool_references")
                                .and_then(|v| v.as_array())
                                .cloned()
                                .unwrap_or_default()
                                .into_iter()
                                .filter_map(|v| v.as_object().cloned())
                                .map(|ref_obj| {
                                    serde_json::json!({
                                        "type": ref_obj.get("type").cloned().unwrap_or_else(|| serde_json::json!("tool_reference")),
                                        "toolName": ref_obj.get("tool_name").cloned().unwrap_or(serde_json::Value::Null),
                                    })
                                })
                                .collect::<Vec<_>>();
                            ToolResultOutput::json(serde_json::Value::Array(refs))
                        } else {
                            let error_code = obj
                                .get("error_code")
                                .cloned()
                                .unwrap_or(serde_json::Value::Null);
                            ToolResultOutput::error_json(serde_json::json!({
                                "type": "tool_search_tool_result_error",
                                "errorCode": error_code,
                            }))
                        }
                    } else {
                        ToolResultOutput::json(content.clone())
                    }
                } else if block_type == "code_execution_tool_result" {
                    if let Some(obj) = content.as_object() {
                        let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if tpe == "code_execution_result" {
                            let mut out = serde_json::json!({
                                "type": "code_execution_result",
                                "stdout": obj.get("stdout").cloned().unwrap_or(serde_json::Value::Null),
                                "stderr": obj.get("stderr").cloned().unwrap_or(serde_json::Value::Null),
                                "return_code": obj.get("return_code").cloned().unwrap_or(serde_json::Value::Null),
                            });
                            if let Some(v) = obj.get("content") {
                                out["content"] = v.clone();
                            }
                            ToolResultOutput::json(out)
                        } else if tpe == "code_execution_tool_result_error" {
                            let error_code = obj
                                .get("error_code")
                                .cloned()
                                .unwrap_or(serde_json::Value::Null);
                            ToolResultOutput::error_json(serde_json::json!({
                                "type": "code_execution_tool_result_error",
                                "errorCode": error_code,
                            }))
                        } else {
                            ToolResultOutput::json(content.clone())
                        }
                    } else {
                        ToolResultOutput::json(content.clone())
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
                    provider_executed: Some(true),
                    provider_metadata: None,
                });
            }
            _ => {}
        }
    }

    // Add any remaining text
    if !text_content.is_empty() {
        parts.push(ContentPart::text(&text_content));
    }

    // Return appropriate content type
    if parts.is_empty() {
        MessageContent::Text(String::new())
    } else if parts.len() == 1 && parts[0].is_text() {
        MessageContent::Text(text_content)
    } else {
        MessageContent::MultiModal(parts)
    }
}

pub fn extract_thinking_content(content_blocks: &[AnthropicContentBlock]) -> Option<String> {
    for content_block in content_blocks {
        if content_block.r#type == "thinking" {
            return content_block.thinking.clone();
        }
    }
    None
}

pub fn create_usage_from_response(usage: Option<AnthropicUsage>) -> Option<Usage> {
    usage.map(|u| Usage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.input_tokens + u.output_tokens,
        #[allow(deprecated)]
        reasoning_tokens: None,
        #[allow(deprecated)]
        cached_tokens: u.cache_read_input_tokens,
        prompt_tokens_details: u.cache_read_input_tokens.map(|cached| {
            crate::types::PromptTokensDetails {
                audio_tokens: None,
                cached_tokens: Some(cached),
            }
        }),
        completion_tokens_details: None,
    })
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
                assert_eq!(parts.len(), 3);

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
                        crate::types::ToolResultOutput::Json { value } => {
                            assert!(value.is_array());
                        }
                        other => panic!("Expected JSON output, got {:?}", other),
                    }
                } else {
                    panic!("Expected tool result content part");
                }

                // text content
                if let ContentPart::Text { text, .. } = &parts[2] {
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
                caller: None,
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

                if let ContentPart::ToolCall { tool_name, .. } = &parts[0] {
                    assert_eq!(tool_name, "tool_search");
                } else {
                    panic!("Expected tool call part");
                }

                if let ContentPart::ToolResult {
                    tool_name, output, ..
                } = &parts[1]
                {
                    assert_eq!(tool_name, "tool_search");
                    match output {
                        crate::types::ToolResultOutput::Json { value } => {
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
                        crate::types::ToolResultOutput::Json { value } => {
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
}
