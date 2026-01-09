//! OpenAI(-compatible) protocol utilities.
//!
//! This module contains wire-format conversion helpers that are shared across
//! OpenAI-compatible provider implementations (e.g. vendor adapters).
#![deny(unsafe_code)]

use crate::error::LlmError;
use crate::standards::openai::types::{OpenAiFunction, OpenAiMessage, OpenAiToolCall};
use crate::types::{
    ChatMessage, ContentPart, FinishReason, MessageContent, MessageRole, ToolResultOutput,
};
use base64::Engine;
use std::collections::HashMap;

fn merge_openai_compatible_extra(
    extra: &mut HashMap<String, serde_json::Value>,
    provider_map: &HashMap<String, serde_json::Value>,
) {
    let Some(serde_json::Value::Object(obj)) = provider_map.get("openaiCompatible") else {
        return;
    };

    for (k, v) in obj {
        extra.insert(k.clone(), v.clone());
    }
}

fn merge_openai_compatible_json(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    provider_metadata: Option<&HashMap<String, serde_json::Value>>,
) {
    let Some(provider_metadata) = provider_metadata else {
        return;
    };

    let Some(serde_json::Value::Object(extra)) = provider_metadata.get("openaiCompatible") else {
        return;
    };

    for (k, v) in extra {
        obj.insert(k.clone(), v.clone());
    }
}

fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(serde_json::Value::String(text.clone())),
        MessageContent::MultiModal(parts) => {
            if parts.len() == 1
                && let Some(ContentPart::Text { text, .. }) = parts.first()
            {
                return Ok(serde_json::Value::String(text.clone()));
            }

            let mut content_parts = Vec::new();

            for part in parts {
                match part {
                    ContentPart::Text {
                        text,
                        provider_metadata,
                    } => {
                        let mut obj = serde_json::Map::new();
                        obj.insert(
                            "type".to_string(),
                            serde_json::Value::String("text".to_string()),
                        );
                        obj.insert("text".to_string(), serde_json::Value::String(text.clone()));
                        merge_openai_compatible_json(&mut obj, provider_metadata.as_ref());
                        content_parts.push(serde_json::Value::Object(obj));
                    }
                    ContentPart::Image {
                        source,
                        detail,
                        provider_metadata,
                    } => {
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

                        if let serde_json::Value::Object(ref mut obj) = image_obj {
                            merge_openai_compatible_json(obj, provider_metadata.as_ref());
                        }

                        content_parts.push(image_obj);
                    }
                    ContentPart::Audio {
                        source, media_type, ..
                    } => match source {
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
                        source,
                        media_type,
                        provider_metadata,
                        ..
                    } => {
                        if media_type.starts_with("image/") {
                            let normalized_media_type = if media_type == "image/*" {
                                "image/jpeg"
                            } else {
                                media_type.as_str()
                            };

                            let url = match source {
                                crate::types::chat::MediaSource::Url { url } => url.clone(),
                                crate::types::chat::MediaSource::Base64 { data } => {
                                    if data.starts_with("data:") {
                                        data.clone()
                                    } else {
                                        format!("data:{};base64,{}", normalized_media_type, data)
                                    }
                                }
                                crate::types::chat::MediaSource::Binary { data } => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    format!("data:{};base64,{}", normalized_media_type, encoded)
                                }
                            };

                            let mut image_obj = serde_json::json!({
                                "type": "image_url",
                                "image_url": { "url": url }
                            });

                            if let serde_json::Value::Object(ref mut obj) = image_obj {
                                merge_openai_compatible_json(obj, provider_metadata.as_ref());
                            }

                            content_parts.push(image_obj);
                        } else {
                            return Err(LlmError::UnsupportedOperation(format!(
                                "OpenAI-compatible chat does not support file part media type {media_type}"
                            )));
                        }
                    }
                    ContentPart::ToolCall { .. } => {}
                    ContentPart::ToolResult { .. } => {}
                    ContentPart::Reasoning { .. } => {}
                    ContentPart::ToolApprovalResponse { .. } => {}
                    ContentPart::ToolApprovalRequest { .. } => {}
                    ContentPart::Source { .. } => {}
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
        let user_single_text_part_metadata = if message.role == MessageRole::User {
            match &message.content {
                MessageContent::MultiModal(parts) if parts.len() == 1 => match parts.first() {
                    Some(ContentPart::Text {
                        provider_metadata: Some(provider_metadata),
                        ..
                    }) => Some(provider_metadata),
                    _ => None,
                },
                _ => None,
            }
        } else {
            None
        };

        let mut openai_message = match message.role {
            MessageRole::System => OpenAiMessage {
                role: "system".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
                extra: HashMap::new(),
            },
            MessageRole::User => OpenAiMessage {
                role: "user".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
                extra: HashMap::new(),
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
                                    provider_metadata,
                                    ..
                                } = part
                                {
                                    let mut tool_call = OpenAiToolCall {
                                        id: tool_call_id.clone(),
                                        r#type: "function".to_string(),
                                        function: Some(OpenAiFunction {
                                            name: tool_name.clone(),
                                            arguments: serde_json::to_string(arguments)
                                                .unwrap_or_default(),
                                        }),
                                        extra: HashMap::new(),
                                    };

                                    if let Some(provider_metadata) = provider_metadata {
                                        merge_openai_compatible_extra(
                                            &mut tool_call.extra,
                                            provider_metadata,
                                        );
                                    }

                                    Some(tool_call)
                                } else {
                                    None
                                }
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                // Vercel AI SDK parity: assistant content is a plain string, formed by
                // concatenating text parts without separators. Tool calls live in `tool_calls`.
                let mut text = String::new();
                match &message.content {
                    MessageContent::Text(t) => text.push_str(t),
                    MessageContent::MultiModal(parts) => {
                        for p in parts {
                            if let ContentPart::Text { text: t, .. } = p {
                                text.push_str(t);
                            }
                        }
                    }
                    #[cfg(feature = "structured-messages")]
                    MessageContent::Json(v) => {
                        text.push_str(&serde_json::to_string(v).unwrap_or_default());
                    }
                }

                OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(serde_json::Value::String(text)),
                    tool_calls: tool_calls_openai,
                    tool_call_id: None,
                    extra: HashMap::new(),
                }
            }
            MessageRole::Tool => {
                // Vercel AI SDK parity: emit one OpenAI "tool" message per tool result.
                // Tool approvals are not represented as tool messages.
                match &message.content {
                    MessageContent::MultiModal(parts) => {
                        let mut emitted = false;
                        for part in parts {
                            let ContentPart::ToolResult {
                                tool_call_id,
                                output,
                                provider_metadata,
                                ..
                            } = part
                            else {
                                continue;
                            };

                            emitted = true;

                            let content_value = match output {
                                ToolResultOutput::Text { value }
                                | ToolResultOutput::ErrorText { value } => value.clone(),
                                ToolResultOutput::ExecutionDenied { reason } => reason
                                    .clone()
                                    .unwrap_or_else(|| "Tool execution denied.".to_string()),
                                ToolResultOutput::Json { value }
                                | ToolResultOutput::ErrorJson { value } => {
                                    serde_json::to_string(value).unwrap_or_default()
                                }
                                ToolResultOutput::Content { value } => {
                                    serde_json::to_string(value).unwrap_or_default()
                                }
                            };

                            openai_messages.push(OpenAiMessage {
                                role: "tool".to_string(),
                                content: Some(serde_json::Value::String(content_value)),
                                tool_calls: None,
                                tool_call_id: Some(tool_call_id.clone()),
                                extra: provider_metadata
                                    .as_ref()
                                    .map(|m| {
                                        let mut extra = HashMap::new();
                                        merge_openai_compatible_extra(&mut extra, m);
                                        extra
                                    })
                                    .unwrap_or_default(),
                            });
                        }

                        if emitted {
                            continue;
                        }

                        // Tool-only messages without results (e.g. approval responses) are omitted.
                        continue;
                    }
                    MessageContent::Text(t) => OpenAiMessage {
                        role: "tool".to_string(),
                        content: Some(serde_json::Value::String(t.clone())),
                        tool_calls: None,
                        tool_call_id: None,
                        extra: HashMap::new(),
                    },
                    #[cfg(feature = "structured-messages")]
                    MessageContent::Json(v) => OpenAiMessage {
                        role: "tool".to_string(),
                        content: Some(serde_json::Value::String(
                            serde_json::to_string(v).unwrap_or_default(),
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                        extra: HashMap::new(),
                    },
                }
            }
            MessageRole::Developer => OpenAiMessage {
                role: "developer".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
                extra: HashMap::new(),
            },
        };

        merge_openai_compatible_extra(&mut openai_message.extra, &message.metadata.custom);
        if let Some(provider_metadata) = user_single_text_part_metadata {
            merge_openai_compatible_extra(&mut openai_message.extra, provider_metadata);
        }
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
