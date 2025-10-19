//! `OpenAI` Utility Functions
//!
//! Common utility functions for `OpenAI` API interactions.

use super::types::*;
use crate::error::LlmError;
use crate::types::*;
use crate::utils::http_headers::ProviderHeaders;
use regex::Regex;
use reqwest::header::HeaderMap;

/// Build HTTP headers for `OpenAI` API requests
pub fn build_headers(
    api_key: &str,
    organization: Option<&str>,
    project: Option<&str>,
    custom_headers: &std::collections::HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    ProviderHeaders::openai(api_key, organization, project, custom_headers)
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
                    ContentPart::Image { image_url, detail } => {
                        let mut image_obj = serde_json::json!({
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        });

                        if let Some(detail) = detail {
                            image_obj["image_url"]["detail"] =
                                serde_json::Value::String(detail.clone());
                        }

                        content_parts.push(image_obj);
                    }
                    ContentPart::Audio {
                        audio_url,
                        format: _,
                    } => {
                        // OpenAI does not currently support audio content directly in chat
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": format!("[Audio: {}]", audio_url)
                        }));
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
            MessageRole::Assistant => OpenAiMessage {
                role: "assistant".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: message.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|call| OpenAiToolCall {
                            id: call.id.clone(),
                            r#type: call.r#type.clone(),
                            function: call.function.as_ref().map(|f| OpenAiFunction {
                                name: f.name.clone(),
                                arguments: f.arguments.clone(),
                            }),
                        })
                        .collect()
                }),
                tool_call_id: None,
            },
            MessageRole::Developer => OpenAiMessage {
                role: "developer".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
            },
            MessageRole::Tool => OpenAiMessage {
                role: "tool".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: message.tool_call_id.clone(),
            },
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
pub fn extract_thinking_content(content: &str) -> Option<String> {
    let re = Regex::new(r"(?s)<think>(.*?)</think>").ok()?;
    re.captures(content)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Filter out thinking content from text for display purposes
/// Removes `<think>...</think>` tags and their content
pub fn filter_thinking_content(content: &str) -> String {
    match Regex::new(r"(?s)<think>.*?</think>") {
        Ok(re) => re.replace_all(content, "").trim().to_string(),
        Err(_) => {
            // Fallback to simple string replacement if regex fails
            content.to_string()
        }
    }
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

/// Decide whether to route to Responses API given OpenAI config
/// Rules:
/// - Explicit flag use_responses_api takes precedence
/// - Auto: models matching is_responses_model (currently only gpt-5*)
pub fn should_route_responses(cfg: &super::config::OpenAiConfig) -> bool {
    // 默认仅在显式开启时走 Responses API，不再根据模型名自动切换。
    cfg.use_responses_api
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::config::OpenAiConfig;

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
    fn test_should_route_responses_explicit_only() {
        let cfg = OpenAiConfig::new("test").with_model("gpt-4o");
        assert!(!should_route_responses(&cfg));

        let cfg = OpenAiConfig::new("test").with_model("gpt-5-mini");
        assert!(!should_route_responses(&cfg));

        let cfg = OpenAiConfig::new("test")
            .with_model("gpt-4")
            .with_responses_api(true);
        assert!(should_route_responses(&cfg));
    }
}
