//! OpenAI(-compatible) protocol utilities.
//!
//! This module contains wire-format conversion helpers that are shared across
//! OpenAI-compatible provider implementations (e.g. vendor adapters).
#![deny(unsafe_code)]

use crate::error::LlmError;
use crate::standards::openai::types::{OpenAiFunction, OpenAiMessage, OpenAiToolCall};
use crate::types::{ChatMessage, ContentPart, FinishReason, MessageContent, MessageRole};
use base64::Engine;

fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
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
                        let url = match source {
                            crate::types::chat::MediaSource::Url { url } => url.clone(),
                            crate::types::chat::MediaSource::Base64 { data } => {
                                if data.starts_with("data:") {
                                    data.clone()
                                } else {
                                    format!("data:image/jpeg;base64,{}", data)
                                }
                            }
                            crate::types::chat::MediaSource::Binary { data } => {
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                format!("data:image/jpeg;base64,{}", encoded)
                            }
                        };

                        let mut image_obj = serde_json::json!({
                            "type": "image_url",
                            "image_url": { "url": url }
                        });

                        if let Some(detail) = detail {
                            image_obj["image_url"]["detail"] = serde_json::json!(detail);
                        }

                        content_parts.push(image_obj);
                    }
                    ContentPart::Audio { source, media_type } => match source {
                        crate::types::chat::MediaSource::Base64 { data } => {
                            let format = infer_audio_format(media_type.as_deref());
                            content_parts.push(serde_json::json!({
                                "type": "input_audio",
                                "input_audio": { "data": data, "format": format }
                            }));
                        }
                        crate::types::chat::MediaSource::Binary { data } => {
                            let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                            let format = infer_audio_format(media_type.as_deref());
                            content_parts.push(serde_json::json!({
                                "type": "input_audio",
                                "input_audio": { "data": encoded, "format": format }
                            }));
                        }
                        crate::types::chat::MediaSource::Url { url } => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("[Audio: {}]", url)
                            }));
                        }
                    },
                    ContentPart::File {
                        source, media_type, ..
                    } => {
                        if media_type == "application/pdf" {
                            let data = match source {
                                crate::types::chat::MediaSource::Base64 { data } => data.clone(),
                                crate::types::chat::MediaSource::Binary { data } => {
                                    base64::engine::general_purpose::STANDARD.encode(data)
                                }
                                crate::types::chat::MediaSource::Url { url } => {
                                    content_parts.push(serde_json::json!({
                                        "type": "text",
                                        "text": format!("[PDF: {}]", url)
                                    }));
                                    continue;
                                }
                            };

                            content_parts.push(serde_json::json!({
                                "type": "file",
                                "file": { "data": data, "media_type": media_type }
                            }));
                        } else {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("[Unsupported file type: {}]", media_type)
                            }));
                        }
                    }
                    ContentPart::ToolCall { .. } => {}
                    ContentPart::ToolResult { .. } => {}
                    ContentPart::Reasoning { .. } => {}
                    ContentPart::ToolApprovalResponse { .. } => {}
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

/// Convert a message content value to the OpenAI(-compatible) wire format.
pub fn convert_message_content_to_openai_value(
    content: &MessageContent,
) -> Result<serde_json::Value, LlmError> {
    convert_message_content(content)
}

/// Convert Siumai messages into OpenAI(-compatible) wire format.
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
            MessageRole::Tool => {
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
            MessageRole::Developer => OpenAiMessage {
                role: "developer".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
            },
        };

        openai_messages.push(openai_message);
    }

    Ok(openai_messages)
}

/// Infer audio format from media type.
pub(crate) fn infer_audio_format(media_type: Option<&str>) -> &'static str {
    match media_type {
        Some("audio/wav") | Some("audio/wave") | Some("audio/x-wav") => "wav",
        Some("audio/mp3") | Some("audio/mpeg") => "mp3",
        _ => "wav",
    }
}

/// Parse OpenAI(-compatible) finish reason to unified `FinishReason`.
pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("stop") => Some(FinishReason::Stop),
        Some("length") => Some(FinishReason::Length),
        Some("tool_calls") => Some(FinishReason::ToolCalls),
        Some("content_filter") => Some(FinishReason::ContentFilter),
        Some("function_call") => Some(FinishReason::ToolCalls),
        Some(other) => Some(FinishReason::Other(other.to_string())),
        None => None,
    }
}
