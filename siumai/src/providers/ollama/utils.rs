//! Ollama utility functions
//!
//! Common utility functions for Ollama provider implementation.

use super::types::*;
use crate::error::LlmError;
use crate::execution::http::headers::ProviderHeaders;
#[allow(deprecated)]
use crate::types::ToolCall;
use crate::types::{ChatMessage, Tool};
use base64::Engine;
use reqwest::header::HeaderMap;
use std::collections::HashMap;

/// Build HTTP headers for Ollama requests
pub fn build_headers(additional_headers: &HashMap<String, String>) -> Result<HeaderMap, LlmError> {
    ProviderHeaders::ollama(additional_headers)
}

/// Convert common `ChatMessage` to Ollama format
pub fn convert_chat_message(message: &ChatMessage) -> OllamaChatMessage {
    let role_str = match message.role {
        crate::types::MessageRole::System => "system",
        crate::types::MessageRole::User => "user",
        crate::types::MessageRole::Assistant => "assistant",
        crate::types::MessageRole::Developer => "system", // Map developer to system
        crate::types::MessageRole::Tool => "tool",
    }
    .to_string();

    let content_str = match &message.content {
        crate::types::MessageContent::Text(text) => text.clone(),
        crate::types::MessageContent::MultiModal(parts) => {
            // Extract text from multimodal content
            parts
                .iter()
                .filter_map(|part| {
                    if let crate::types::ContentPart::Text { text } = part {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        #[cfg(feature = "structured-messages")]
        crate::types::MessageContent::Json(v) => serde_json::to_string(v).unwrap_or_default(),
    };

    let mut ollama_message = OllamaChatMessage {
        role: role_str,
        content: content_str,
        images: None,
        tool_calls: None,
        thinking: None,
    };

    // Extract images from multimodal content
    if let crate::types::MessageContent::MultiModal(parts) = &message.content {
        let images: Vec<String> = parts
            .iter()
            .filter_map(|part| {
                if let crate::types::ContentPart::Image { source, .. } = part {
                    match source {
                        crate::types::chat::MediaSource::Url { url } => Some(url.clone()),
                        crate::types::chat::MediaSource::Base64 { data } => {
                            Some(format!("data:image/jpeg;base64,{}", data))
                        }
                        crate::types::chat::MediaSource::Binary { data } => {
                            let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                            Some(format!("data:image/jpeg;base64,{}", encoded))
                        }
                    }
                } else {
                    None
                }
            })
            .collect();

        if !images.is_empty() {
            ollama_message.images = Some(images);
        }
    }

    // Convert tool calls if present
    let tool_calls = message.tool_calls();
    if !tool_calls.is_empty() {
        ollama_message.tool_calls = Some(
            tool_calls
                .iter()
                .filter_map(|part| {
                    if let crate::types::ContentPart::ToolCall {
                        tool_name,
                        arguments,
                        ..
                    } = part
                    {
                        Some(OllamaToolCall {
                            function: OllamaFunctionCall {
                                name: tool_name.clone(),
                                arguments: arguments.clone(),
                            },
                        })
                    } else {
                        None
                    }
                })
                .collect(),
        );
    }

    ollama_message
}

/// Convert common Tool to Ollama format
pub fn convert_tool(tool: &Tool) -> Option<OllamaTool> {
    match tool {
        Tool::Function { function } => Some(OllamaTool {
            tool_type: "function".to_string(),
            function: OllamaFunction {
                name: function.name.clone(),
                description: function.description.clone(),
                parameters: function.parameters.clone(),
            },
        }),
        Tool::ProviderDefined(_) => {
            // Ollama doesn't support provider-defined tools
            // Return None to skip them
            None
        }
    }
}

/// Convert common `ToolCall` to Ollama format
#[deprecated(note = "Use ContentPart::ToolCall instead")]
#[allow(deprecated)]
pub fn convert_tool_call(tool_call: &crate::types::ToolCall) -> OllamaToolCall {
    OllamaToolCall {
        function: OllamaFunctionCall {
            name: tool_call
                .function
                .as_ref()
                .map(|f| f.name.clone())
                .unwrap_or_default(),
            arguments: tool_call
                .function
                .as_ref()
                .map(|f| {
                    serde_json::from_str(&f.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
                })
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
        },
    }
}

/// Convert Ollama chat message to common format
pub fn convert_from_ollama_message(message: &OllamaChatMessage) -> ChatMessage {
    let role = match message.role.as_str() {
        "system" => crate::types::MessageRole::System,
        "user" => crate::types::MessageRole::User,
        "assistant" => crate::types::MessageRole::Assistant,
        "tool" => crate::types::MessageRole::Tool,
        _ => crate::types::MessageRole::Assistant, // Default fallback
    };

    let mut parts = vec![crate::types::ContentPart::Text {
        text: message.content.clone(),
    }];

    // Add images if present
    if let Some(images) = &message.images {
        for image_url in images {
            parts.push(crate::types::ContentPart::Image {
                source: crate::types::chat::MediaSource::Url {
                    url: image_url.clone(),
                },
                detail: None,
            });
        }
    }

    // Add tool calls if present
    if let Some(tool_calls) = &message.tool_calls {
        for tc in tool_calls {
            parts.push(crate::types::ContentPart::tool_call(
                format!("call_{}", chrono::Utc::now().timestamp_millis()),
                tc.function.name.clone(),
                tc.function.arguments.clone(),
                None,
            ));
        }
    }

    // Add thinking content if present
    if let Some(thinking) = &message.thinking {
        parts.push(crate::types::ContentPart::reasoning(thinking));
    }

    // Determine final content
    let content = if parts.len() == 1 && parts[0].is_text() {
        crate::types::MessageContent::Text(message.content.clone())
    } else {
        crate::types::MessageContent::MultiModal(parts)
    };

    ChatMessage {
        role,
        content,
        metadata: crate::types::MessageMetadata::default(),
    }
}

/// Convert Ollama tool call to common format
#[allow(deprecated)]
pub fn convert_from_ollama_tool_call(tool_call: &OllamaToolCall) -> ToolCall {
    ToolCall {
        id: format!("call_{}", chrono::Utc::now().timestamp_millis()), // Generate ID since Ollama doesn't provide one
        r#type: "function".to_string(),
        function: Some(crate::types::FunctionCall {
            name: tool_call.function.name.clone(),
            arguments: tool_call.function.arguments.to_string(),
        }),
    }
}

/// Parse streaming response line
pub fn parse_streaming_line(line: &str) -> Result<Option<serde_json::Value>, LlmError> {
    let line = line.trim();

    // Skip empty lines and comments
    if line.is_empty() || line.starts_with(':') {
        return Ok(None);
    }

    // Remove "data: " prefix if present
    let json_str = if let Some(stripped) = line.strip_prefix("data: ") {
        stripped
    } else {
        line
    };

    // Skip [DONE] marker
    if json_str == "[DONE]" {
        return Ok(None);
    }

    // Parse JSON
    serde_json::from_str(json_str)
        .map(Some)
        .map_err(|e| LlmError::ParseError(format!("Failed to parse streaming response: {e}")))
}

/// Extract model name from model string (handles model:tag format)
pub fn extract_model_name(model: &str) -> String {
    // Ollama models can be in format "model:tag" or just "model"
    // We keep the full format as Ollama expects it
    model.to_string()
}

/// Validate model name format
pub fn validate_model_name(model: &str) -> Result<(), LlmError> {
    if model.is_empty() {
        return Err(LlmError::ConfigurationError(
            "Model name cannot be empty".to_string(),
        ));
    }

    // Basic validation - model names should not contain invalid characters
    if model.contains(' ') || model.contains('\n') || model.contains('\t') {
        return Err(LlmError::ConfigurationError(
            "Model name contains invalid characters".to_string(),
        ));
    }

    Ok(())
}

/// Build model options from common parameters
pub fn build_model_options(
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    additional_options: Option<&HashMap<String, serde_json::Value>>,
) -> HashMap<String, serde_json::Value> {
    let mut options = HashMap::new();

    if let Some(temp) = temperature {
        options.insert(
            "temperature".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(temp as f64)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
    }

    if let Some(max_tokens) = max_tokens {
        options.insert(
            "num_predict".to_string(),
            serde_json::Value::Number(serde_json::Number::from(max_tokens)),
        );
    }

    if let Some(top_p) = top_p {
        options.insert(
            "top_p".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(top_p as f64)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
    }

    if let Some(freq_penalty) = frequency_penalty {
        options.insert(
            "frequency_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(freq_penalty as f64)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
    }

    if let Some(pres_penalty) = presence_penalty {
        options.insert(
            "presence_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(pres_penalty as f64)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
    }

    // Add additional options
    if let Some(additional) = additional_options {
        for (key, value) in additional {
            options.insert(key.clone(), value.clone());
        }
    }

    options
}

/// Calculate tokens per second from Ollama response metrics
pub fn calculate_tokens_per_second(
    eval_count: Option<u32>,
    eval_duration: Option<u64>,
) -> Option<f64> {
    match (eval_count, eval_duration) {
        (Some(count), Some(duration)) if duration > 0 => {
            // Convert nanoseconds to seconds and calculate tokens/second
            let duration_seconds = duration as f64 / 1_000_000_000.0;
            Some(count as f64 / duration_seconds)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{CONTENT_TYPE, USER_AGENT};

    #[test]
    fn test_build_headers() {
        let additional = HashMap::new();
        let headers = build_headers(&additional).unwrap();

        assert!(headers.contains_key(CONTENT_TYPE));
        assert!(headers.contains_key(USER_AGENT));
    }

    #[test]
    fn test_convert_chat_message() {
        let message = ChatMessage {
            role: crate::types::MessageRole::User,
            content: crate::types::MessageContent::MultiModal(vec![
                crate::types::ContentPart::Text {
                    text: "Hello".to_string(),
                },
                crate::types::ContentPart::Image {
                    source: crate::types::chat::MediaSource::Url {
                        url: "image1".to_string(),
                    },
                    detail: None,
                },
            ]),
            metadata: crate::types::MessageMetadata::default(),
        };

        let ollama_message = convert_chat_message(&message);
        assert_eq!(ollama_message.role, "user");
        assert_eq!(ollama_message.content, "Hello");
        assert_eq!(ollama_message.images, Some(vec!["image1".to_string()]));
    }

    #[test]
    fn test_validate_model_name() {
        assert!(validate_model_name("llama3.2").is_ok());
        assert!(validate_model_name("llama3.2:latest").is_ok());
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name("model with spaces").is_err());
    }

    #[test]
    fn test_calculate_tokens_per_second() {
        assert_eq!(
            calculate_tokens_per_second(Some(100), Some(1_000_000_000)),
            Some(100.0)
        );
        assert_eq!(
            calculate_tokens_per_second(Some(50), Some(500_000_000)),
            Some(100.0)
        );
        assert_eq!(calculate_tokens_per_second(None, Some(1_000_000_000)), None);
        assert_eq!(calculate_tokens_per_second(Some(100), None), None);
        assert_eq!(calculate_tokens_per_second(Some(100), Some(0)), None);
    }
}
