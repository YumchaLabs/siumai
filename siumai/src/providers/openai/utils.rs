//! `OpenAI` Utility Functions
//!
//! Common utility functions for `OpenAI` API interactions.

use super::types::*;
use crate::error::LlmError;
use crate::types::*;
use crate::utils::http_headers::ProviderHeaders;
use base64::Engine;
use reqwest::header::HeaderMap;

/// Infer audio format from media type
pub(crate) fn infer_audio_format(media_type: Option<&str>) -> &'static str {
    match media_type {
        Some("audio/wav") | Some("audio/wave") | Some("audio/x-wav") => "wav",
        Some("audio/mp3") | Some("audio/mpeg") => "mp3",
        _ => "wav", // default to wav
    }
}

/// Build HTTP headers for `OpenAI` API requests
pub fn build_headers(
    api_key: &str,
    organization: Option<&str>,
    project: Option<&str>,
    custom_headers: &std::collections::HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    ProviderHeaders::openai(api_key, organization, project, custom_headers)
}

/// Convert tools to OpenAI Chat API format
pub fn convert_tools_to_openai_format(
    tools: &[crate::types::Tool],
) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut openai_tools = Vec::new();

    for tool in tools {
        match tool {
            crate::types::Tool::Function { function } => {
                openai_tools.push(serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": function.name,
                        "description": function.description,
                        "parameters": function.parameters
                    }
                }));
            }
            crate::types::Tool::ProviderDefined(provider_tool) => {
                // Check if this is an OpenAI provider-defined tool
                if provider_tool.provider() == Some("openai") {
                    // For OpenAI provider-defined tools, construct the tool definition
                    let tool_type = provider_tool.tool_type().unwrap_or("unknown");

                    let mut openai_tool = serde_json::json!({
                        "type": tool_type,
                    });

                    // Merge args into the tool definition
                    if let serde_json::Value::Object(args_map) = &provider_tool.args {
                        if let serde_json::Value::Object(tool_map) = &mut openai_tool {
                            for (k, v) in args_map {
                                tool_map.insert(k.clone(), v.clone());
                            }
                        }
                    }

                    openai_tools.push(openai_tool);
                } else {
                    // Ignore provider-defined tools from other providers
                    continue;
                }
            }
        }
    }

    Ok(openai_tools)
}

/// Convert tools to OpenAI Responses API format (flattened)
pub fn convert_tools_to_responses_format(
    tools: &[crate::types::Tool],
) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut openai_tools = Vec::new();

    for tool in tools {
        match tool {
            crate::types::Tool::Function { function } => {
                openai_tools.push(serde_json::json!({
                    "type": "function",
                    "name": function.name,
                    "description": function.description,
                    "parameters": function.parameters
                }));
            }
            crate::types::Tool::ProviderDefined(provider_tool) => {
                // Check if this is an OpenAI provider-defined tool
                if provider_tool.provider() == Some("openai") {
                    let tool_type = provider_tool.tool_type().unwrap_or("unknown");

                    let mut openai_tool = serde_json::json!({
                        "type": tool_type,
                    });

                    // Merge args into the tool definition
                    if let serde_json::Value::Object(args_map) = &provider_tool.args {
                        if let serde_json::Value::Object(tool_map) = &mut openai_tool {
                            for (k, v) in args_map {
                                tool_map.insert(k.clone(), v.clone());
                            }
                        }
                    }

                    openai_tools.push(openai_tool);
                } else {
                    // Ignore provider-defined tools from other providers
                    continue;
                }
            }
        }
    }

    Ok(openai_tools)
}

/// Convert message content to `OpenAI` format
pub fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(serde_json::Value::String(text.clone())),
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
                        // Extract URL or base64 data
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

                        let mut image_obj = serde_json::json!({
                            "type": "image_url",
                            "image_url": {
                                "url": url
                            }
                        });

                        if let Some(detail) = detail {
                            image_obj["image_url"]["detail"] = serde_json::json!(detail);
                        }

                        content_parts.push(image_obj);
                    }
                    ContentPart::Audio { source, media_type } => {
                        // OpenAI supports base64 audio input
                        match source {
                            crate::types::chat::MediaSource::Base64 { data } => {
                                // Infer format from media_type
                                let format = infer_audio_format(media_type.as_deref());
                                content_parts.push(serde_json::json!({
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": data,
                                        "format": format
                                    }
                                }));
                            }
                            crate::types::chat::MediaSource::Binary { data } => {
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                let format = infer_audio_format(media_type.as_deref());
                                content_parts.push(serde_json::json!({
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": encoded,
                                        "format": format
                                    }
                                }));
                            }
                            crate::types::chat::MediaSource::Url { url } => {
                                // OpenAI doesn't support audio URLs, convert to text placeholder
                                content_parts.push(serde_json::json!({
                                    "type": "text",
                                    "text": format!("[Audio: {}]", url)
                                }));
                            }
                        }
                    }
                    ContentPart::File {
                        source, media_type, ..
                    } => {
                        // OpenAI supports PDF files
                        if media_type == "application/pdf" {
                            let data = match source {
                                crate::types::chat::MediaSource::Base64 { data } => data.clone(),
                                crate::types::chat::MediaSource::Binary { data } => {
                                    base64::engine::general_purpose::STANDARD.encode(data)
                                }
                                crate::types::chat::MediaSource::Url { url } => {
                                    // Convert to text placeholder for URLs
                                    content_parts.push(serde_json::json!({
                                        "type": "text",
                                        "text": format!("[PDF: {}]", url)
                                    }));
                                    continue;
                                }
                            };
                            content_parts.push(serde_json::json!({
                                "type": "file",
                                "file": {
                                    "data": data,
                                    "media_type": media_type
                                }
                            }));
                        } else {
                            // Unsupported file type, convert to text placeholder
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("[Unsupported file type: {}]", media_type)
                            }));
                        }
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

            Ok(serde_json::Value::Array(content_parts))
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(v) => Ok(serde_json::Value::String(
            serde_json::to_string(v).unwrap_or_default(),
        )),
    }
}

/// Convert messages to `OpenAI` format
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<OpenAiMessage>, LlmError> {
    let mut openai_messages = Vec::new();

    for message in messages {
        let openai_message = match message.role {
            MessageRole::System => OpenAiMessage {
                role: "system".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
            },
            MessageRole::User => OpenAiMessage {
                role: "user".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
            },
            MessageRole::Assistant => {
                // Extract tool calls from content
                let tool_calls_vec = message.tool_calls();
                let tool_calls_openai = if !tool_calls_vec.is_empty() {
                    Some(
                        tool_calls_vec
                            .iter()
                            .filter_map(|part| {
                                if let crate::types::ContentPart::ToolCall {
                                    tool_call_id,
                                    tool_name,
                                    arguments,
                                    ..
                                } = part
                                {
                                    Some(OpenAiToolCall {
                                        id: tool_call_id.clone(),
                                        r#type: "function".to_string(),
                                        function: Some(OpenAiFunction {
                                            name: tool_name.clone(),
                                            arguments: serde_json::to_string(arguments)
                                                .unwrap_or_default(),
                                        }),
                                    })
                                } else {
                                    None
                                }
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(convert_message_content(&message.content)?),
                    tool_calls: tool_calls_openai,
                    tool_call_id: None,
                }
            }
            MessageRole::Developer => OpenAiMessage {
                role: "developer".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
            },
            MessageRole::Tool => {
                // Extract tool call ID from tool result in content
                let tool_results = message.tool_results();
                let tool_call_id = tool_results.first().and_then(|part| {
                    if let crate::types::ContentPart::ToolResult { tool_call_id, .. } = part {
                        Some(tool_call_id.clone())
                    } else {
                        None
                    }
                });

                OpenAiMessage {
                    role: "tool".to_string(),
                    content: Some(convert_message_content(&message.content)?),
                    tool_calls: None,
                    tool_call_id,
                }
            }
        };

        openai_messages.push(openai_message);
    }

    Ok(openai_messages)
}

/// Parse `OpenAI` finish reason
pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("stop") => Some(FinishReason::Stop),
        Some("length") => Some(FinishReason::Length),
        Some("tool_calls") => Some(FinishReason::ToolCalls),
        Some("content_filter") => Some(FinishReason::ContentFilter),
        Some("function_call") => Some(FinishReason::ToolCalls), // function_call is deprecated, map to tool_calls
        Some(other) => Some(FinishReason::Other(other.to_string())),
        None => None,
    }
}

/// Get default models for `OpenAI`
pub fn get_default_models() -> Vec<String> {
    use crate::types::models::model_constants::openai;

    let models = vec![
        openai::GPT_5.to_string(),
        openai::GPT_4_1.to_string(),
        openai::GPT_4O.to_string(),
        openai::GPT_4O_MINI.to_string(),
        openai::GPT_4_TURBO.to_string(),
        openai::GPT_4.to_string(),
        openai::O1.to_string(),
        openai::O1_MINI.to_string(),
        openai::O3_MINI.to_string(),
        openai::GPT_3_5_TURBO.to_string(),
    ];

    models
}

/// Check if content contains thinking tags (`<think>` or `</think>`)
/// This is used to detect DeepSeek-style thinking content
pub fn contains_thinking_tags(content: &str) -> bool {
    content.contains("<think>") || content.contains("</think>")
}

/// Extract thinking content from `<think>...</think>` tags
/// Returns the content inside the tags, or None if no valid tags found
/// Uses simple string parsing instead of regex for better performance
pub fn extract_thinking_content(content: &str) -> Option<String> {
    let start_tag = "<think>";
    let end_tag = "</think>";

    let start_pos = content.find(start_tag)?;
    let content_start = start_pos + start_tag.len();
    let end_pos = content[content_start..].find(end_tag)?;

    let thinking = content[content_start..content_start + end_pos].trim();
    if thinking.is_empty() {
        None
    } else {
        Some(thinking.to_string())
    }
}

/// Filter out thinking content from text for display purposes
/// Removes `<think>...</think>` tags and their content
/// Uses simple string parsing instead of regex for better performance
pub fn filter_thinking_content(content: &str) -> String {
    let start_tag = "<think>";
    let end_tag = "</think>";

    let mut result = String::new();
    let mut remaining = content;

    while let Some(start_pos) = remaining.find(start_tag) {
        // Add content before the tag
        result.push_str(&remaining[..start_pos]);

        // Find the end tag
        if let Some(end_pos) = remaining[start_pos..].find(end_tag) {
            // Skip the thinking content and continue after the end tag
            let skip_to = start_pos + end_pos + end_tag.len();
            remaining = &remaining[skip_to..];
        } else {
            // No matching end tag, remove everything from start tag onwards
            remaining = "";
            break;
        }
    }

    // Add any remaining content
    result.push_str(remaining);
    result.trim().to_string()
}

/// Extract content without thinking tags
/// If content contains thinking tags, filter them out; otherwise return as-is
pub fn extract_content_without_thinking(content: &str) -> String {
    if contains_thinking_tags(content) {
        filter_thinking_content(content)
    } else {
        content.to_string()
    }
}

/// Determine if a model should default to Responses API (auto mode)
/// Currently only gpt-5 family triggers auto routing
pub fn is_responses_model(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    m.starts_with("gpt-5")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_responses_model_only_gpt5() {
        assert!(is_responses_model("gpt-5"));
        assert!(is_responses_model("gpt-5-mini"));
        assert!(is_responses_model("GPT-5-VISION"));
        assert!(!is_responses_model("gpt-4o"));
        assert!(!is_responses_model("o1"));
        assert!(!is_responses_model(""));
    }

    #[test]
    fn test_extract_thinking_content() {
        // Test basic extraction
        let content = "Some text <think>This is thinking</think> more text";
        assert_eq!(
            extract_thinking_content(content),
            Some("This is thinking".to_string())
        );

        // Test with newlines
        let content = "Text <think>\nMultiline\nthinking\n</think> end";
        assert_eq!(
            extract_thinking_content(content),
            Some("Multiline\nthinking".to_string())
        );

        // Test with no thinking tags
        assert_eq!(extract_thinking_content("No tags here"), None);

        // Test with empty thinking
        assert_eq!(extract_thinking_content("<think></think>"), None);
        assert_eq!(extract_thinking_content("<think>   </think>"), None);

        // Test with only start tag
        assert_eq!(extract_thinking_content("<think>No end tag"), None);
    }

    #[test]
    fn test_filter_thinking_content() {
        // Test basic filtering
        let content = "Before <think>thinking</think> after";
        assert_eq!(filter_thinking_content(content), "Before  after");

        // Test multiple thinking blocks
        let content = "A <think>t1</think> B <think>t2</think> C";
        assert_eq!(filter_thinking_content(content), "A  B  C");

        // Test with no thinking tags
        let content = "No tags here";
        assert_eq!(filter_thinking_content(content), "No tags here");

        // Test with unclosed tag
        let content = "Before <think>unclosed";
        assert_eq!(filter_thinking_content(content), "Before");

        // Test with nested-like content
        let content = "A <think>B <think>C</think> D";
        // Should remove from first <think> to first </think>
        assert_eq!(filter_thinking_content(content), "A  D");
    }

    #[test]
    fn test_contains_thinking_tags() {
        assert!(contains_thinking_tags("<think>content</think>"));
        assert!(contains_thinking_tags("text <think>"));
        assert!(contains_thinking_tags("text </think>"));
        assert!(!contains_thinking_tags("no tags"));
        assert!(!contains_thinking_tags(""));
    }
}

/// Convert provider-agnostic ToolChoice to OpenAI format
///
/// # OpenAI Format
///
/// - `Auto` → `"auto"`
/// - `Required` → `"required"`
/// - `None` → `"none"`
/// - `Tool { name }` → `{"type": "function", "function": {"name": "..."}}`
///
/// # Example
///
/// ```rust
/// use siumai::types::ToolChoice;
/// use siumai::providers::openai::utils::convert_tool_choice;
///
/// let choice = ToolChoice::tool("weather");
/// let openai_format = convert_tool_choice(&choice);
/// ```
pub fn convert_tool_choice(choice: &crate::types::ToolChoice) -> serde_json::Value {
    use crate::types::ToolChoice;

    match choice {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::Required => serde_json::json!("required"),
        ToolChoice::None => serde_json::json!("none"),
        ToolChoice::Tool { name } => serde_json::json!({
            "type": "function",
            "function": {
                "name": name
            }
        }),
    }
}

#[cfg(test)]
mod tool_choice_tests {
    use super::*;

    #[test]
    fn test_extract_content_without_thinking() {
        let content = "Before <think>thinking</think> after";
        assert_eq!(extract_content_without_thinking(content), "Before  after");

        let content = "No thinking tags";
        assert_eq!(
            extract_content_without_thinking(content),
            "No thinking tags"
        );
    }

    #[test]
    fn test_convert_tool_choice() {
        use crate::types::ToolChoice;

        // Test Auto
        let result = convert_tool_choice(&ToolChoice::Auto);
        assert_eq!(result, serde_json::json!("auto"));

        // Test Required
        let result = convert_tool_choice(&ToolChoice::Required);
        assert_eq!(result, serde_json::json!("required"));

        // Test None
        let result = convert_tool_choice(&ToolChoice::None);
        assert_eq!(result, serde_json::json!("none"));

        // Test Tool
        let result = convert_tool_choice(&ToolChoice::tool("weather"));
        assert_eq!(
            result,
            serde_json::json!({
                "type": "function",
                "function": { "name": "weather" }
            })
        );
    }
}
