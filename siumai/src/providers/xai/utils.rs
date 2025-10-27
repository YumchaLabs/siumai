//! `xAI` Utility Functions
//!
//! This module contains utility functions for the `xAI` provider.

use base64::Engine;
use reqwest::header::HeaderMap;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
use crate::types::{ChatMessage, ContentPart, FinishReason, MessageContent};

/// Build HTTP headers for `xAI` API requests
pub fn build_headers(
    api_key: &str,
    additional_headers: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    ProviderHeaders::xai(api_key, additional_headers)
}

/// Convert internal message format to `xAI` format
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut converted = Vec::new();

    for message in messages {
        let mut msg = serde_json::json!({
            "role": message.role
        });

        match &message.content {
            MessageContent::Text(text) => {
                msg["content"] = serde_json::Value::String(text.clone());
            }
            MessageContent::MultiModal(parts) => {
                let mut content_parts = Vec::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": text
                            }));
                        }
                        ContentPart::Image { source, detail } => {
                            // xAI supports image URLs and base64
                            let url = match source {
                                crate::types::chat::MediaSource::Url { url } => url.clone(),
                                crate::types::chat::MediaSource::Base64 { data } => {
                                    format!("data:image/jpeg;base64,{}", data)
                                }
                                crate::types::chat::MediaSource::Binary { data } => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    format!("data:image/jpeg;base64,{}", encoded)
                                }
                            };

                            let mut image_part = serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": url
                                }
                            });
                            if let Some(detail) = detail {
                                image_part["image_url"]["detail"] = serde_json::json!(detail);
                            }
                            content_parts.push(image_part);
                        }
                        ContentPart::Audio { .. } => {
                            return Err(LlmError::UnsupportedOperation(
                                "Audio content not supported by xAI".to_string(),
                            ));
                        }
                        ContentPart::File { .. } => {
                            return Err(LlmError::UnsupportedOperation(
                                "File content not supported by xAI".to_string(),
                            ));
                        }
                        ContentPart::ToolCall { .. } => {
                            // Tool calls are handled separately in convert_messages
                            // Skip them here as they're not part of content array
                        }
                        ContentPart::ToolResult { .. } => {
                            // Tool results are handled separately in convert_messages
                            // Skip them here as they're not part of content array
                        }
                        ContentPart::Reasoning { .. } => {
                            // Reasoning/thinking is handled separately
                            // Skip it here as it's not part of content array
                        }
                    }
                }
                msg["content"] = serde_json::Value::Array(content_parts);
            }
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => {
                msg["content"] =
                    serde_json::Value::String(serde_json::to_string(v).unwrap_or_default());
            }
        }

        // Add tool calls if present (extract from content)
        let tool_calls = message.tool_calls();
        if !tool_calls.is_empty() {
            let tool_calls_json: Vec<serde_json::Value> = tool_calls
                .iter()
                .filter_map(|part| {
                    if let crate::types::ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        ..
                    } = part
                    {
                        Some(serde_json::json!({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments
                            }
                        }))
                    } else {
                        None
                    }
                })
                .collect();
            if !tool_calls_json.is_empty() {
                msg["tool_calls"] = serde_json::Value::Array(tool_calls_json);
            }
        }

        // Add tool call ID if present (for tool response messages - extract from content)
        let tool_results = message.tool_results();
        if let Some(first_result) = tool_results.first()
            && let crate::types::ContentPart::ToolResult { tool_call_id, .. } = first_result
        {
            msg["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
        }

        converted.push(msg);
    }

    Ok(converted)
}

/// Parse finish reason from `xAI` response
pub fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("content_filter") => FinishReason::ContentFilter,
        Some("function_call") => FinishReason::ToolCalls, // Legacy support
        _ => FinishReason::Other(reason.unwrap_or("unknown").to_string()),
    }
}

/// Check if text contains thinking tags
pub fn contains_thinking_tags(text: &str) -> bool {
    text.contains("<think>") && text.contains("</think>")
}

/// Extract thinking content from text
pub fn extract_thinking_content(text: &str) -> Option<String> {
    if let Some(start) = text.find("<think>")
        && let Some(end) = text.find("</think>")
        && start < end
    {
        let thinking_start = start + "<think>".len();
        return Some(text[thinking_start..end].trim().to_string());
    }
    None
}

/// Filter out thinking content from text
pub fn filter_thinking_content(text: &str) -> String {
    let mut result = text.to_string();

    // Remove all <think>...</think> blocks
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            if start < end {
                let end_pos = end + "</think>".len();
                result.replace_range(start..end_pos, "");
            } else {
                break;
            }
        } else {
            break;
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[test]
    fn test_build_headers() {
        let headers = build_headers("test-key", &HashMap::new()).unwrap();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer test-key");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            ChatMessage::user("Hello").build(),
            ChatMessage::assistant("Hi there!").build(),
        ];

        let converted = convert_messages(&messages).unwrap();
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0]["role"], "user");
        assert_eq!(converted[0]["content"], "Hello");
        assert_eq!(converted[1]["role"], "assistant");
        assert_eq!(converted[1]["content"], "Hi there!");
    }

    #[test]
    fn test_thinking_content() {
        let text = "Before I answer, <think>Let me think about this carefully...</think> Here's my response.";

        assert!(contains_thinking_tags(text));
        assert_eq!(
            extract_thinking_content(text),
            Some("Let me think about this carefully...".to_string())
        );
        assert_eq!(
            filter_thinking_content(text),
            "Before I answer,  Here's my response."
        );
    }

    #[test]
    fn test_parse_finish_reason() {
        assert_eq!(parse_finish_reason(Some("stop")), FinishReason::Stop);
        assert_eq!(parse_finish_reason(Some("length")), FinishReason::Length);
        assert_eq!(
            parse_finish_reason(Some("tool_calls")),
            FinishReason::ToolCalls
        );
        assert_eq!(
            parse_finish_reason(Some("content_filter")),
            FinishReason::ContentFilter
        );
        assert_eq!(
            parse_finish_reason(Some("unknown")),
            FinishReason::Other("unknown".to_string())
        );
        assert_eq!(
            parse_finish_reason(None),
            FinishReason::Other("unknown".to_string())
        );
    }
}
