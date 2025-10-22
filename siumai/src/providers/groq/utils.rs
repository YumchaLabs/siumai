//! `Groq` Utility Functions
//!
//! Utility functions for the Groq provider.

use base64::Engine;
use reqwest::header::HeaderMap;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::types::{ChatMessage, FinishReason, MessageContent, MessageRole};
use crate::utils::http_headers::ProviderHeaders;

/// Build HTTP headers for Groq API requests
pub fn build_headers(
    api_key: &str,
    custom_headers: &HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    ProviderHeaders::groq(api_key, custom_headers)
}

/// Convert internal messages to Groq API format
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut groq_messages = Vec::new();

    for message in messages {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Developer => "system", // Map developer to system for Groq
            MessageRole::Tool => "tool",
        };

        let content = match &message.content {
            MessageContent::Text(text) => serde_json::Value::String(text.clone()),
            MessageContent::MultiModal(parts) => {
                let mut content_parts = Vec::new();
                for part in parts {
                    match part {
                        crate::types::ContentPart::Text { text } => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": text
                            }));
                        }
                        crate::types::ContentPart::Image { source, detail } => {
                            // Groq supports image URLs and base64
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
                        crate::types::ContentPart::Audio { source, .. } => {
                            // Groq does not support audio input
                            let placeholder = match source {
                                crate::types::chat::MediaSource::Url { url } => {
                                    format!("[Audio: {}]", url)
                                }
                                _ => "[Audio not supported]".to_string(),
                            };
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": placeholder
                            }));
                        }
                        crate::types::ContentPart::File { source, .. } => {
                            // Groq does not support file input
                            let placeholder = match source {
                                crate::types::chat::MediaSource::Url { url } => {
                                    format!("[File: {}]", url)
                                }
                                _ => "[File not supported]".to_string(),
                            };
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": placeholder
                            }));
                        }
                        crate::types::ContentPart::ToolCall { .. } => {
                            // Tool calls are handled separately
                            // Skip them here as they're not part of content array
                        }
                        crate::types::ContentPart::ToolResult { .. } => {
                            // Tool results are handled separately
                            // Skip them here as they're not part of content array
                        }
                        crate::types::ContentPart::Reasoning { .. } => {
                            // Reasoning/thinking is handled separately
                            // Skip it here as it's not part of content array
                        }
                    }
                }
                serde_json::Value::Array(content_parts)
            }
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => {
                serde_json::Value::String(serde_json::to_string(v).unwrap_or_default())
            }
        };

        let mut groq_message = serde_json::json!({
            "role": role,
            "content": content
        });

        // Extract tool calls from content
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
                                "arguments": arguments,
                            }
                        }))
                    } else {
                        None
                    }
                })
                .collect();
            if !tool_calls_json.is_empty() {
                groq_message["tool_calls"] = serde_json::Value::Array(tool_calls_json);
            }
        }

        // Extract tool call ID from tool result
        let tool_results = message.tool_results();
        if let Some(first_result) = tool_results.first() {
            if let crate::types::ContentPart::ToolResult { tool_call_id, .. } = first_result {
                groq_message["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
            }
        }

        groq_messages.push(groq_message);
    }

    Ok(groq_messages)
}

/// Parse finish reason from Groq API response
pub fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("content_filter") => FinishReason::ContentFilter,
        Some("function_call") => FinishReason::ToolCalls, // Legacy function_call maps to tool_calls
        _ => FinishReason::Other("unknown".to_string()),
    }
}

/// Extract error message from Groq API error response
pub fn extract_error_message(error_text: &str) -> String {
    // Try to parse as JSON error response
    if let Ok(error_response) = serde_json::from_str::<super::types::GroqErrorResponse>(error_text)
    {
        return error_response.error.message;
    }

    // Fallback to raw error text
    error_text.to_string()
}

/// Validate Groq-specific parameters
pub fn validate_groq_params(params: &serde_json::Value) -> Result<(), LlmError> {
    // Validate frequency_penalty
    if let Some(freq_penalty) = params.get("frequency_penalty")
        && let Some(value) = freq_penalty.as_f64()
        && !(-2.0..=2.0).contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "frequency_penalty must be between -2.0 and 2.0".to_string(),
        ));
    }

    // Validate presence_penalty
    if let Some(pres_penalty) = params.get("presence_penalty")
        && let Some(value) = pres_penalty.as_f64()
        && !(-2.0..=2.0).contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "presence_penalty must be between -2.0 and 2.0".to_string(),
        ));
    }

    // Validate temperature (relaxed validation - only check for negative values)
    if let Some(temperature) = params.get("temperature")
        && let Some(value) = temperature.as_f64()
        && value < 0.0
    {
        return Err(LlmError::InvalidParameter(
            "temperature cannot be negative".to_string(),
        ));
    }

    // Validate top_p
    if let Some(top_p) = params.get("top_p")
        && let Some(value) = top_p.as_f64()
        && !(0.0..=1.0).contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "top_p must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Validate n (number of choices)
    if let Some(n) = params.get("n")
        && let Some(value) = n.as_u64()
        && value != 1
    {
        return Err(LlmError::InvalidParameter(
            "Groq only supports n=1".to_string(),
        ));
    }

    // Validate service_tier
    if let Some(service_tier) = params.get("service_tier")
        && let Some(value) = service_tier.as_str()
        && !["auto", "on_demand", "flex"].contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "service_tier must be one of: auto, on_demand, flex".to_string(),
        ));
    }

    // Validate reasoning_effort (for qwen3 models)
    if let Some(reasoning_effort) = params.get("reasoning_effort")
        && let Some(value) = reasoning_effort.as_str()
        && !["none", "default"].contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "reasoning_effort must be one of: none, default".to_string(),
        ));
    }

    // Validate reasoning_format
    if let Some(reasoning_format) = params.get("reasoning_format")
        && let Some(value) = reasoning_format.as_str()
        && !["hidden", "raw", "parsed"].contains(&value)
    {
        return Err(LlmError::InvalidParameter(
            "reasoning_format must be one of: hidden, raw, parsed".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;
    use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, USER_AGENT};

    #[test]
    fn test_build_headers() {
        let custom_headers = HashMap::new();
        let headers = build_headers("test-api-key", &custom_headers).unwrap();

        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer test-api-key");
        assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
        assert!(headers.get(USER_AGENT).is_some());
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant").build(),
            ChatMessage::user("Hello, world!").build(),
        ];

        let groq_messages = convert_messages(&messages).unwrap();
        assert_eq!(groq_messages.len(), 2);
        assert_eq!(groq_messages[0]["role"], "system");
        assert_eq!(groq_messages[1]["role"], "user");
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
            parse_finish_reason(Some("unknown")),
            FinishReason::Other("unknown".to_string())
        );
        assert_eq!(
            parse_finish_reason(None),
            FinishReason::Other("unknown".to_string())
        );
    }

    #[test]
    fn test_validate_groq_params() {
        // Valid parameters
        let valid_params = serde_json::json!({
            "temperature": 0.7,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "service_tier": "auto"
        });
        assert!(validate_groq_params(&valid_params).is_ok());

        // High temperature (now allowed with relaxed validation)
        let high_temp = serde_json::json!({
            "temperature": 3.0
        });
        assert!(validate_groq_params(&high_temp).is_ok());

        // Negative temperature (still invalid)
        let invalid_temp = serde_json::json!({
            "temperature": -1.0
        });
        assert!(validate_groq_params(&invalid_temp).is_err());

        // Invalid service_tier
        let invalid_tier = serde_json::json!({
            "service_tier": "invalid"
        });
        assert!(validate_groq_params(&invalid_tier).is_err());
    }
}
