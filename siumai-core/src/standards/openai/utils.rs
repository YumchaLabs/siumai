//! OpenAI(-compatible) protocol utilities.
//!
//! This module contains wire-format conversion helpers that are shared across
//! OpenAI-compatible provider implementations (e.g. vendor adapters).
#![deny(unsafe_code)]

use crate::error::LlmError;
use crate::standards::openai::types::{OpenAiFunction, OpenAiMessage, OpenAiToolCall};
use crate::types::{
    ChatMessage, ContentPart, FilePartSource, FinishReason, MediaSource, MessageContent,
    MessageRole, ProviderOptionsMap, ProviderReference, ToolResultOutput, Usage, UsageInputTokens,
    UsageOutputTokens,
};
use base64::Engine;
use serde_json::{Map, Value};
use std::borrow::Cow;
use std::collections::HashMap;

fn openai_compatible_options_object(
    provider_options: Option<&ProviderOptionsMap>,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    provider_options
        .and_then(|provider_options| provider_options.get("openaiCompatible"))
        .and_then(|value| value.as_object())
}

fn merge_openai_compatible_extra(
    extra: &mut HashMap<String, serde_json::Value>,
    provider_options: Option<&ProviderOptionsMap>,
) {
    let Some(obj) = openai_compatible_options_object(provider_options) else {
        return;
    };

    for (k, v) in obj {
        extra.insert(k.clone(), v.clone());
    }
}

fn merge_openai_compatible_json(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    provider_options: Option<&ProviderOptionsMap>,
) {
    let Some(extra) = openai_compatible_options_object(provider_options) else {
        return;
    };

    for (k, v) in extra {
        obj.insert(k.clone(), v.clone());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageConversionTarget {
    /// Match Vercel `@ai-sdk/openai-compatible` behavior.
    OpenAiCompatible,
    /// Match Vercel `@ai-sdk/openai` chat message conversion behavior.
    OpenAiChat,
}

fn openai_chat_audio_format(media_type: &str) -> Result<&'static str, LlmError> {
    match media_type {
        "audio/wav" | "audio/wave" | "audio/x-wav" => Ok("wav"),
        "audio/mp3" | "audio/mpeg" => Ok("mp3"),
        _ => Err(LlmError::UnsupportedOperation(format!(
            "audio content parts with media type {media_type}"
        ))),
    }
}

fn extract_openai_chat_image_detail(
    provider_options: Option<&ProviderOptionsMap>,
) -> Option<String> {
    provider_options
        .and_then(|provider_options| {
            provider_options
                .get_object("openai")
                .or_else(|| provider_options.get_object("azure"))
        })
        .and_then(|openai| {
            openai
                .get("imageDetail")
                .or_else(|| openai.get("image_detail"))
        })
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn resolve_openai_chat_provider_reference<'a>(
    provider_reference: &'a ProviderReference,
) -> Result<&'a str, LlmError> {
    for provider_id in ["openai", "azure"] {
        if let Some(reference) = provider_reference.get(provider_id) {
            return Ok(reference);
        }
    }

    let available = provider_reference.available_providers();
    let available = if available.is_empty() {
        "none".to_string()
    } else {
        available.join(", ")
    };

    Err(LlmError::InvalidParameter(format!(
        "No provider reference found for OpenAI chat. Available providers: {available}"
    )))
}

fn unsupported_openai_compatible_provider_reference(label: &str) -> LlmError {
    LlmError::UnsupportedOperation(format!(
        "OpenAI-compatible chat does not support {label} with provider references"
    ))
}

fn convert_message_content_with_target(
    content: &MessageContent,
    target: MessageConversionTarget,
) -> Result<serde_json::Value, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(serde_json::Value::String(text.clone())),
        MessageContent::MultiModal(parts) => {
            if parts.len() == 1
                && let Some(ContentPart::Text { text, .. }) = parts.first()
            {
                return Ok(serde_json::Value::String(text.clone()));
            }

            let mut content_parts = Vec::new();

            for (index, part) in parts.iter().enumerate() {
                match part {
                    ContentPart::Text {
                        text,
                        provider_options,
                        ..
                    } => {
                        let mut obj = serde_json::Map::new();
                        obj.insert(
                            "type".to_string(),
                            serde_json::Value::String("text".to_string()),
                        );
                        obj.insert("text".to_string(), serde_json::Value::String(text.clone()));
                        if target == MessageConversionTarget::OpenAiCompatible {
                            merge_openai_compatible_json(&mut obj, Some(provider_options));
                        }
                        content_parts.push(serde_json::Value::Object(obj));
                    }
                    ContentPart::Image {
                        source,
                        detail,
                        provider_options,
                        ..
                    } => {
                        let url = match source {
                            FilePartSource::Media(MediaSource::Url { url }) => url.clone(),
                            FilePartSource::Media(MediaSource::Base64 { data }) => {
                                if data.starts_with("data:") {
                                    data.clone()
                                } else {
                                    format!("data:image/jpeg;base64,{}", data)
                                }
                            }
                            FilePartSource::Media(MediaSource::Binary { data }) => {
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                format!("data:image/jpeg;base64,{}", encoded)
                            }
                            FilePartSource::ProviderReference { provider_reference } => {
                                match target {
                                    MessageConversionTarget::OpenAiChat => {
                                        content_parts.push(serde_json::json!({
                                            "type": "file",
                                            "file": {
                                                "file_id": resolve_openai_chat_provider_reference(provider_reference)?,
                                            }
                                        }));
                                        continue;
                                    }
                                    MessageConversionTarget::OpenAiCompatible => {
                                        return Err(
                                            unsupported_openai_compatible_provider_reference(
                                                "image parts",
                                            ),
                                        );
                                    }
                                }
                            }
                        };

                        let mut image_obj = serde_json::json!({
                            "type": "image_url",
                            "image_url": { "url": url }
                        });

                        if let Some(detail) = detail {
                            image_obj["image_url"]["detail"] = serde_json::json!(detail);
                        }

                        if target == MessageConversionTarget::OpenAiCompatible
                            && let serde_json::Value::Object(ref mut obj) = image_obj
                        {
                            merge_openai_compatible_json(obj, Some(provider_options));
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
                        provider_options,
                        filename,
                        ..
                    } => {
                        if media_type.starts_with("image/") {
                            let normalized_media_type = if media_type == "image/*" {
                                "image/jpeg"
                            } else {
                                media_type.as_str()
                            };

                            let url = match source {
                                FilePartSource::Media(MediaSource::Url { url }) => url.clone(),
                                FilePartSource::Media(MediaSource::Base64 { data }) => {
                                    if data.starts_with("data:") {
                                        data.clone()
                                    } else {
                                        format!("data:{};base64,{}", normalized_media_type, data)
                                    }
                                }
                                FilePartSource::Media(MediaSource::Binary { data }) => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    format!("data:{};base64,{}", normalized_media_type, encoded)
                                }
                                FilePartSource::ProviderReference { provider_reference } => {
                                    match target {
                                        MessageConversionTarget::OpenAiChat => {
                                            content_parts.push(serde_json::json!({
                                                "type": "file",
                                                "file": {
                                                    "file_id": resolve_openai_chat_provider_reference(provider_reference)?,
                                                }
                                            }));
                                            continue;
                                        }
                                        MessageConversionTarget::OpenAiCompatible => {
                                            return Err(
                                                unsupported_openai_compatible_provider_reference(
                                                    "file image parts",
                                                ),
                                            );
                                        }
                                    }
                                }
                            };

                            let mut image_obj = serde_json::json!({
                                "type": "image_url",
                                "image_url": { "url": url }
                            });

                            if target == MessageConversionTarget::OpenAiChat
                                && let Some(detail) =
                                    extract_openai_chat_image_detail(Some(provider_options))
                            {
                                image_obj["image_url"]["detail"] = serde_json::json!(detail);
                            }

                            if target == MessageConversionTarget::OpenAiCompatible
                                && let serde_json::Value::Object(ref mut obj) = image_obj
                            {
                                merge_openai_compatible_json(obj, Some(provider_options));
                            }

                            content_parts.push(image_obj);
                        } else if target == MessageConversionTarget::OpenAiChat
                            && media_type.starts_with("audio/")
                        {
                            let format = openai_chat_audio_format(media_type)?;
                            match source {
                                FilePartSource::Media(MediaSource::Url { .. }) => {
                                    return Err(LlmError::UnsupportedOperation(
                                        "audio file parts with URLs".to_string(),
                                    ));
                                }
                                FilePartSource::Media(MediaSource::Base64 { data }) => {
                                    content_parts.push(serde_json::json!({
                                        "type": "input_audio",
                                        "input_audio": { "data": data, "format": format }
                                    }));
                                }
                                FilePartSource::Media(MediaSource::Binary { data }) => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    content_parts.push(serde_json::json!({
                                        "type": "input_audio",
                                        "input_audio": { "data": encoded, "format": format }
                                    }));
                                }
                                FilePartSource::ProviderReference { .. } => {
                                    return Err(LlmError::UnsupportedOperation(
                                        "audio file parts with provider references".to_string(),
                                    ));
                                }
                            }
                        } else if target == MessageConversionTarget::OpenAiChat
                            && media_type == "application/pdf"
                        {
                            match source {
                                FilePartSource::Media(MediaSource::Url { .. }) => {
                                    return Err(LlmError::UnsupportedOperation(
                                        "PDF file parts with URLs".to_string(),
                                    ));
                                }
                                FilePartSource::Media(MediaSource::Base64 { data }) => {
                                    let file = serde_json::json!({
                                        "filename": filename.clone().unwrap_or_else(|| format!("part-{}.pdf", index)),
                                        "file_data": format!("data:application/pdf;base64,{}", data),
                                    });
                                    content_parts
                                        .push(serde_json::json!({ "type": "file", "file": file }));
                                }
                                FilePartSource::Media(MediaSource::Binary { data }) => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    let file = serde_json::json!({
                                        "filename": filename.clone().unwrap_or_else(|| format!("part-{}.pdf", index)),
                                        "file_data": format!("data:application/pdf;base64,{}", encoded),
                                    });
                                    content_parts
                                        .push(serde_json::json!({ "type": "file", "file": file }));
                                }
                                FilePartSource::ProviderReference { provider_reference } => {
                                    content_parts.push(serde_json::json!({
                                        "type": "file",
                                        "file": {
                                            "file_id": resolve_openai_chat_provider_reference(provider_reference)?,
                                        }
                                    }));
                                }
                            }
                        } else {
                            if target == MessageConversionTarget::OpenAiChat {
                                return Err(LlmError::UnsupportedOperation(format!(
                                    "file part media type {media_type}"
                                )));
                            }
                            return Err(LlmError::UnsupportedOperation(format!(
                                "OpenAI-compatible chat does not support file part media type {media_type}"
                            )));
                        }
                    }
                    ContentPart::ToolCall { .. } => {}
                    ContentPart::ToolResult { .. } => {}
                    ContentPart::Reasoning { .. } => {}
                    ContentPart::ReasoningFile { .. } => {}
                    ContentPart::Custom { .. } => {}
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

fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    convert_message_content_with_target(content, MessageConversionTarget::OpenAiCompatible)
}

/// Convert a message content value to the OpenAI(-compatible) wire format.
pub fn convert_message_content_to_openai_value(
    content: &MessageContent,
) -> Result<serde_json::Value, LlmError> {
    convert_message_content(content)
}

/// Convert a message content value to the OpenAI Chat Completions wire format.
///
/// This is aligned with Vercel `@ai-sdk/openai` behavior (PDF/audio file parts).
pub fn convert_message_content_to_openai_chat_value(
    content: &MessageContent,
) -> Result<serde_json::Value, LlmError> {
    convert_message_content_with_target(content, MessageConversionTarget::OpenAiChat)
}

/// Convert Siumai messages into OpenAI(-compatible) wire format.
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<OpenAiMessage>, LlmError> {
    convert_messages_with_target(messages, MessageConversionTarget::OpenAiCompatible)
}

/// Convert Siumai messages into OpenAI Chat Completions wire format.
///
/// This is aligned with Vercel `@ai-sdk/openai` behavior (PDF/audio file parts).
pub fn convert_messages_openai_chat(
    messages: &[ChatMessage],
) -> Result<Vec<OpenAiMessage>, LlmError> {
    convert_messages_with_target(messages, MessageConversionTarget::OpenAiChat)
}

fn convert_messages_with_target(
    messages: &[ChatMessage],
    target: MessageConversionTarget,
) -> Result<Vec<OpenAiMessage>, LlmError> {
    let mut openai_messages = Vec::new();

    for message in messages {
        let user_single_text_part_options = if target == MessageConversionTarget::OpenAiCompatible
            && message.role == MessageRole::User
        {
            match &message.content {
                MessageContent::MultiModal(parts) if parts.len() == 1 => match parts.first() {
                    Some(ContentPart::Text {
                        provider_options, ..
                    }) if !provider_options.is_empty() => Some(provider_options),
                    _ => None,
                },
                _ => None,
            }
        } else {
            None
        };

        let openai_message = match message.role {
            MessageRole::System => {
                let mut extra = HashMap::new();
                if target == MessageConversionTarget::OpenAiCompatible {
                    merge_openai_compatible_extra(&mut extra, Some(&message.provider_options));
                }

                OpenAiMessage {
                    role: "system".to_string(),
                    content: Some(convert_message_content_with_target(
                        &message.content,
                        target,
                    )?),
                    tool_calls: None,
                    tool_call_id: None,
                    extra,
                }
            }
            MessageRole::User => {
                let mut extra = HashMap::new();
                if let Some(provider_options) = user_single_text_part_options {
                    merge_openai_compatible_extra(&mut extra, Some(provider_options));
                } else if target == MessageConversionTarget::OpenAiCompatible {
                    merge_openai_compatible_extra(&mut extra, Some(&message.provider_options));
                }

                OpenAiMessage {
                    role: "user".to_string(),
                    content: Some(convert_message_content_with_target(
                        &message.content,
                        target,
                    )?),
                    tool_calls: None,
                    tool_call_id: None,
                    extra,
                }
            }
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
                                    provider_options,
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

                                    if target == MessageConversionTarget::OpenAiCompatible {
                                        merge_openai_compatible_extra(
                                            &mut tool_call.extra,
                                            Some(provider_options),
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
                let mut reasoning = String::new();
                match &message.content {
                    MessageContent::Text(t) => text.push_str(t),
                    MessageContent::MultiModal(parts) => {
                        for p in parts {
                            match p {
                                ContentPart::Text { text: t, .. } => text.push_str(t),
                                ContentPart::Reasoning { text: t, .. }
                                    if target == MessageConversionTarget::OpenAiCompatible =>
                                {
                                    reasoning.push_str(t);
                                }
                                _ => {}
                            }
                        }
                    }
                    #[cfg(feature = "structured-messages")]
                    MessageContent::Json(v) => {
                        text.push_str(&serde_json::to_string(v).unwrap_or_default());
                    }
                }

                let mut extra = HashMap::new();
                if target == MessageConversionTarget::OpenAiCompatible {
                    merge_openai_compatible_extra(&mut extra, Some(&message.provider_options));
                    if !reasoning.is_empty() {
                        extra.insert(
                            "reasoning_content".to_string(),
                            serde_json::Value::String(reasoning),
                        );
                    }
                }

                OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(serde_json::Value::String(text)),
                    tool_calls: tool_calls_openai,
                    tool_call_id: None,
                    extra,
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
                                provider_options,
                                ..
                            } = part
                            else {
                                continue;
                            };

                            emitted = true;

                            let content_value = match output {
                                ToolResultOutput::Text { value, .. }
                                | ToolResultOutput::ErrorText { value, .. } => value.clone(),
                                ToolResultOutput::ExecutionDenied { reason, .. } => reason
                                    .clone()
                                    .unwrap_or_else(|| "Tool execution denied.".to_string()),
                                ToolResultOutput::Json { value, .. }
                                | ToolResultOutput::ErrorJson { value, .. } => {
                                    serde_json::to_string(value).unwrap_or_default()
                                }
                                ToolResultOutput::Content { value, .. } => {
                                    serde_json::to_string(value).unwrap_or_default()
                                }
                            };

                            let mut extra = HashMap::new();
                            if target == MessageConversionTarget::OpenAiCompatible {
                                merge_openai_compatible_extra(&mut extra, Some(provider_options));
                            }

                            openai_messages.push(OpenAiMessage {
                                role: "tool".to_string(),
                                content: Some(serde_json::Value::String(content_value)),
                                tool_calls: None,
                                tool_call_id: Some(tool_call_id.clone()),
                                extra,
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
            MessageRole::Developer => {
                let mut extra = HashMap::new();
                if target == MessageConversionTarget::OpenAiCompatible {
                    merge_openai_compatible_extra(&mut extra, Some(&message.provider_options));
                }

                OpenAiMessage {
                    role: "developer".to_string(),
                    content: Some(convert_message_content_with_target(
                        &message.content,
                        target,
                    )?),
                    tool_calls: None,
                    tool_call_id: None,
                    extra,
                }
            }
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

/// Parse provider-specific OpenAI-compatible finish reasons to unified `FinishReason`.
pub fn parse_provider_openai_finish_reason(
    provider_id: &str,
    reason: Option<&str>,
) -> Option<FinishReason> {
    let reason = reason?;

    Some(match provider_id {
        "deepseek" => match reason {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "content_filter" => FinishReason::ContentFilter,
            "tool_calls" => FinishReason::ToolCalls,
            "insufficient_system_resource" => FinishReason::Error,
            other => FinishReason::Other(other.to_string()),
        },
        "mistral" => match reason {
            "stop" => FinishReason::Stop,
            "length" | "model_length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            other => FinishReason::Other(other.to_string()),
        },
        "perplexity" => match reason {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            other => FinishReason::Other(other.to_string()),
        },
        "cohere" => match reason {
            "COMPLETE" | "STOP_SEQUENCE" => FinishReason::Stop,
            "MAX_TOKENS" => FinishReason::Length,
            "ERROR" => FinishReason::Error,
            "TOOL_CALL" => FinishReason::ToolCalls,
            other => FinishReason::Other(other.to_string()),
        },
        _ => parse_finish_reason(Some(reason))?,
    })
}

fn usage_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_u64)
        .map(|value| value.min(u32::MAX as u64) as u32)
}

fn usage_value<'a>(object: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| object.get(*key))
}

fn usage_object<'a>(
    object: &'a Map<String, Value>,
    keys: &[&str],
) -> Option<&'a Map<String, Value>> {
    usage_value(object, keys).and_then(Value::as_object)
}

fn parse_input_tokens_value(
    tokens: Option<&Value>,
) -> (Option<u32>, Option<u32>, Option<u32>, Option<u32>) {
    let Some(tokens) = tokens else {
        return (None, None, None, None);
    };

    if let Some(total) = usage_u32(Some(tokens)) {
        return (Some(total), Some(total), None, None);
    }

    let Some(object) = tokens.as_object() else {
        return (None, None, None, None);
    };

    let no_cache = usage_u32(usage_value(object, &["noCache", "no_cache"]));
    let cache_read = usage_u32(usage_value(object, &["cacheRead", "cache_read"]));
    let cache_write = usage_u32(usage_value(object, &["cacheWrite", "cache_write"]));
    let total = usage_u32(usage_value(object, &["total", "totalTokens"])).or_else(|| {
        if no_cache.is_none() && cache_read.is_none() && cache_write.is_none() {
            None
        } else {
            Some(
                no_cache
                    .unwrap_or(0)
                    .saturating_add(cache_read.unwrap_or(0))
                    .saturating_add(cache_write.unwrap_or(0)),
            )
        }
    });

    let no_cache = no_cache.or_else(|| {
        total.map(|total| {
            total
                .saturating_sub(cache_read.unwrap_or(0))
                .saturating_sub(cache_write.unwrap_or(0))
        })
    });

    (total, no_cache, cache_read, cache_write)
}

fn parse_output_tokens_value(tokens: Option<&Value>) -> (Option<u32>, Option<u32>, Option<u32>) {
    let Some(tokens) = tokens else {
        return (None, None, None);
    };

    if let Some(total) = usage_u32(Some(tokens)) {
        return (Some(total), Some(total), None);
    }

    let Some(object) = tokens.as_object() else {
        return (None, None, None);
    };

    let reasoning = usage_u32(usage_value(object, &["reasoning", "reasoningTokens"]));
    let text = usage_u32(usage_value(object, &["text", "textTokens"]));
    let total = usage_u32(usage_value(object, &["total", "totalTokens"])).or_else(|| {
        if text.is_none() && reasoning.is_none() {
            None
        } else {
            Some(text.unwrap_or(0).saturating_add(reasoning.unwrap_or(0)))
        }
    });
    let text = text.or_else(|| total.map(|total| total.saturating_sub(reasoning.unwrap_or(0))));

    (total, text, reasoning)
}

fn stripped_raw_usage_object(usage: &Usage, removed_keys: &[&str]) -> Map<String, Value> {
    let mut object = usage.raw.clone().unwrap_or_default();
    for key in removed_keys {
        object.remove(*key);
    }
    object
}

fn ensure_object_entry<'a>(
    object: &'a mut Map<String, Value>,
    key: &str,
) -> &'a mut Map<String, Value> {
    if !object.get(key).is_some_and(Value::is_object) {
        object.insert(key.to_string(), Value::Object(Map::new()));
    }

    object
        .get_mut(key)
        .and_then(Value::as_object_mut)
        .expect("usage detail entry must be an object")
}

fn set_usage_detail_number(
    object: &mut Map<String, Value>,
    detail_key: &str,
    token_key: &str,
    value: u32,
) {
    ensure_object_entry(object, detail_key).insert(token_key.to_string(), serde_json::json!(value));
}

fn parse_normalized_openai_usage_with_raw(
    mut normalized: Map<String, Value>,
    raw: Map<String, Value>,
) -> Option<Usage> {
    normalized.insert("raw".to_string(), Value::Object(raw));
    parse_openai_usage_value(&Value::Object(normalized))
}

/// Parse OpenAI chat/responses/AI SDK usage payloads into the unified `Usage` shape.
pub fn parse_openai_usage_value(value: &Value) -> Option<Usage> {
    let object = value.as_object()?;

    let (input_total, input_no_cache, input_cache_read, input_cache_write) =
        parse_input_tokens_value(object.get("inputTokens"));
    let (output_total, output_text, output_reasoning) =
        parse_output_tokens_value(object.get("outputTokens"));

    let prompt_tokens = usage_u32(usage_value(object, &["prompt_tokens", "input_tokens"]))
        .or(input_total)
        .or_else(|| {
            if input_no_cache.is_none() && input_cache_read.is_none() && input_cache_write.is_none()
            {
                None
            } else {
                Some(
                    input_no_cache
                        .unwrap_or(0)
                        .saturating_add(input_cache_read.unwrap_or(0))
                        .saturating_add(input_cache_write.unwrap_or(0)),
                )
            }
        });

    let completion_tokens = usage_u32(usage_value(object, &["completion_tokens", "output_tokens"]))
        .or(output_total)
        .or_else(|| {
            if output_text.is_none() && output_reasoning.is_none() {
                None
            } else {
                Some(
                    output_text
                        .unwrap_or(0)
                        .saturating_add(output_reasoning.unwrap_or(0)),
                )
            }
        });

    let total_tokens = usage_u32(usage_value(object, &["total_tokens", "totalTokens"]));

    let prompt_details = usage_object(object, &["prompt_tokens_details", "input_tokens_details"]);
    let completion_details = usage_object(
        object,
        &["completion_tokens_details", "output_tokens_details"],
    );

    let cached_tokens = usage_u32(
        prompt_details.and_then(|details| usage_value(details, &["cached_tokens", "cachedTokens"])),
    )
    .or(input_cache_read);
    let reasoning_tokens = usage_u32(usage_value(
        object,
        &["reasoning_tokens", "reasoningTokens"],
    ))
    .or_else(|| {
        usage_u32(
            completion_details
                .and_then(|details| usage_value(details, &["reasoning_tokens", "reasoningTokens"])),
        )
    })
    .or(output_reasoning);

    let prompt_audio_tokens = usage_u32(
        prompt_details.and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let completion_audio_tokens = usage_u32(
        completion_details
            .and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let accepted_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["accepted_prediction_tokens", "acceptedPredictionTokens"],
        )
    }));
    let rejected_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["rejected_prediction_tokens", "rejectedPredictionTokens"],
        )
    }));

    let mut builder = Usage::builder();

    if let Some(prompt_tokens) = prompt_tokens {
        builder = builder.prompt_tokens(prompt_tokens);
    }
    if let Some(completion_tokens) = completion_tokens {
        builder = builder.completion_tokens(completion_tokens);
    }
    if let Some(total_tokens) = total_tokens {
        builder = builder.total_tokens(total_tokens);
    }
    if let Some(cached_tokens) = cached_tokens {
        builder = builder.with_cached_tokens(cached_tokens);
    }
    if let Some(reasoning_tokens) = reasoning_tokens {
        builder = builder.with_reasoning_tokens(reasoning_tokens);
    }
    if let Some(prompt_audio_tokens) = prompt_audio_tokens {
        builder = builder.with_prompt_audio_tokens(prompt_audio_tokens);
    }
    if let Some(completion_audio_tokens) = completion_audio_tokens {
        builder = builder.with_completion_audio_tokens(completion_audio_tokens);
    }
    if let Some(accepted_prediction_tokens) = accepted_prediction_tokens {
        builder = builder.with_accepted_prediction_tokens(accepted_prediction_tokens);
    }
    if let Some(rejected_prediction_tokens) = rejected_prediction_tokens {
        builder = builder.with_rejected_prediction_tokens(rejected_prediction_tokens);
    }
    if let Some(input_total) = input_total {
        builder = builder.with_input_total_tokens(input_total);
    }
    if let Some(input_no_cache) = input_no_cache {
        builder = builder.with_input_no_cache_tokens(input_no_cache);
    }
    if let Some(input_cache_read) = input_cache_read {
        builder = builder.with_input_cache_read_tokens(input_cache_read);
    }
    if let Some(input_cache_write) = input_cache_write {
        builder = builder.with_input_cache_write_tokens(input_cache_write);
    }
    if let Some(output_total) = output_total {
        builder = builder.with_output_total_tokens(output_total);
    }
    if let Some(output_text) = output_text {
        builder = builder.with_output_text_tokens(output_text);
    }
    if let Some(output_reasoning) = output_reasoning {
        builder = builder.with_output_reasoning_tokens(output_reasoning);
    }

    let raw_usage = object
        .get("raw")
        .and_then(Value::as_object)
        .cloned()
        .or_else(|| {
            if object.contains_key("raw") {
                None
            } else {
                Some(object.clone())
            }
        });
    if let Some(raw_usage) = raw_usage {
        builder = builder.with_raw_usage(raw_usage);
    }

    Some(builder.build())
}

struct OpenAiCompatibleChatUsageFields {
    raw: Map<String, Value>,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: Option<u32>,
    cache_read_tokens: u32,
    cache_write_tokens: Option<u32>,
    no_cache_tokens: Option<u32>,
    reasoning_tokens: u32,
    output_text_tokens: Option<u32>,
    prompt_detail_cached_tokens: Option<u32>,
    completion_detail_reasoning_tokens: Option<u32>,
    prompt_audio_tokens: Option<u32>,
    completion_audio_tokens: Option<u32>,
    accepted_prediction_tokens: Option<u32>,
    rejected_prediction_tokens: Option<u32>,
}

fn build_openai_compatible_chat_usage(fields: OpenAiCompatibleChatUsageFields) -> Usage {
    let cache_write_for_no_cache = fields.cache_write_tokens.unwrap_or(0);
    let no_cache_tokens = fields.no_cache_tokens.unwrap_or_else(|| {
        fields
            .prompt_tokens
            .saturating_sub(fields.cache_read_tokens)
            .saturating_sub(cache_write_for_no_cache)
    });
    let output_text_tokens = fields.output_text_tokens.unwrap_or_else(|| {
        fields
            .completion_tokens
            .saturating_sub(fields.reasoning_tokens)
    });

    let mut builder = Usage::builder()
        .prompt_tokens(fields.prompt_tokens)
        .completion_tokens(fields.completion_tokens)
        .with_input_total_tokens(fields.prompt_tokens)
        .with_input_no_cache_tokens(no_cache_tokens)
        .with_input_cache_read_tokens(fields.cache_read_tokens)
        .with_output_total_tokens(fields.completion_tokens)
        .with_output_text_tokens(output_text_tokens)
        .with_output_reasoning_tokens(fields.reasoning_tokens)
        .with_raw_usage(fields.raw);

    if let Some(total_tokens) = fields.total_tokens {
        builder = builder.total_tokens(total_tokens);
    }
    if let Some(cache_write_tokens) = fields.cache_write_tokens {
        builder = builder.with_input_cache_write_tokens(cache_write_tokens);
    }
    if let Some(prompt_detail_cached_tokens) = fields.prompt_detail_cached_tokens {
        builder = builder.with_cached_tokens(prompt_detail_cached_tokens);
    }
    if let Some(completion_detail_reasoning_tokens) = fields.completion_detail_reasoning_tokens {
        builder = builder.with_reasoning_tokens(completion_detail_reasoning_tokens);
    }
    if let Some(prompt_audio_tokens) = fields.prompt_audio_tokens {
        builder = builder.with_prompt_audio_tokens(prompt_audio_tokens);
    }
    if let Some(completion_audio_tokens) = fields.completion_audio_tokens {
        builder = builder.with_completion_audio_tokens(completion_audio_tokens);
    }
    if let Some(accepted_prediction_tokens) = fields.accepted_prediction_tokens {
        builder = builder.with_accepted_prediction_tokens(accepted_prediction_tokens);
    }
    if let Some(rejected_prediction_tokens) = fields.rejected_prediction_tokens {
        builder = builder.with_rejected_prediction_tokens(rejected_prediction_tokens);
    }

    builder.build()
}

fn parse_openai_compatible_chat_usage_value(value: &Value) -> Option<Usage> {
    let object = value.as_object()?;
    let raw = object
        .get("raw")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_else(|| object.clone());

    let (input_total, input_no_cache, input_cache_read, input_cache_write) =
        parse_input_tokens_value(object.get("inputTokens"));
    let (output_total, output_text, output_reasoning) =
        parse_output_tokens_value(object.get("outputTokens"));

    let prompt_tokens = usage_u32(usage_value(object, &["prompt_tokens", "input_tokens"]))
        .or(input_total)
        .unwrap_or(0);
    let completion_tokens = usage_u32(usage_value(object, &["completion_tokens", "output_tokens"]))
        .or(output_total)
        .unwrap_or(0);
    let total_tokens = usage_u32(usage_value(object, &["total_tokens", "totalTokens"]));

    let prompt_details = usage_object(object, &["prompt_tokens_details", "input_tokens_details"]);
    let completion_details = usage_object(
        object,
        &["completion_tokens_details", "output_tokens_details"],
    );
    let cache_read_tokens = usage_u32(
        prompt_details.and_then(|details| usage_value(details, &["cached_tokens", "cachedTokens"])),
    )
    .or(input_cache_read);
    let reasoning_tokens = usage_u32(usage_value(
        object,
        &["reasoning_tokens", "reasoningTokens"],
    ))
    .or_else(|| {
        usage_u32(
            completion_details
                .and_then(|details| usage_value(details, &["reasoning_tokens", "reasoningTokens"])),
        )
    })
    .or(output_reasoning);
    let prompt_audio_tokens = usage_u32(
        prompt_details.and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let completion_audio_tokens = usage_u32(
        completion_details
            .and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let accepted_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["accepted_prediction_tokens", "acceptedPredictionTokens"],
        )
    }));
    let rejected_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["rejected_prediction_tokens", "rejectedPredictionTokens"],
        )
    }));

    Some(build_openai_compatible_chat_usage(
        OpenAiCompatibleChatUsageFields {
            raw,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cache_read_tokens: cache_read_tokens.unwrap_or(0),
            cache_write_tokens: input_cache_write,
            no_cache_tokens: input_no_cache,
            reasoning_tokens: reasoning_tokens.unwrap_or(0),
            output_text_tokens: output_text,
            prompt_detail_cached_tokens: cache_read_tokens,
            completion_detail_reasoning_tokens: reasoning_tokens,
            prompt_audio_tokens,
            completion_audio_tokens,
            accepted_prediction_tokens,
            rejected_prediction_tokens,
        },
    ))
}

fn parse_deepseek_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();
    let mut normalized = raw.clone();

    if let Some(cache_read_tokens) = usage_u32(usage_value(&raw, &["prompt_cache_hit_tokens"])) {
        set_usage_detail_number(
            &mut normalized,
            "prompt_tokens_details",
            "cached_tokens",
            cache_read_tokens,
        );
    }

    parse_normalized_openai_usage_with_raw(normalized, raw)
}

fn parse_groq_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();
    let mut normalized = raw.clone();

    if let Some(details) = normalized
        .get_mut("prompt_tokens_details")
        .and_then(Value::as_object_mut)
    {
        details.remove("cached_tokens");
        details.remove("cachedTokens");
        if details.is_empty() {
            normalized.remove("prompt_tokens_details");
        }
    }

    parse_normalized_openai_usage_with_raw(normalized, raw)
}

fn parse_moonshotai_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();

    let prompt_tokens =
        usage_u32(usage_value(&raw, &["prompt_tokens", "input_tokens"])).unwrap_or(0);
    let completion_tokens =
        usage_u32(usage_value(&raw, &["completion_tokens", "output_tokens"])).unwrap_or(0);
    let total_tokens = usage_u32(usage_value(&raw, &["total_tokens", "totalTokens"]));
    let prompt_details = usage_object(&raw, &["prompt_tokens_details", "input_tokens_details"]);
    let completion_details = usage_object(
        &raw,
        &["completion_tokens_details", "output_tokens_details"],
    );
    let cache_read_tokens = usage_u32(usage_value(&raw, &["cached_tokens", "cachedTokens"]))
        .or_else(|| {
            usage_u32(
                prompt_details
                    .and_then(|details| usage_value(details, &["cached_tokens", "cachedTokens"])),
            )
        });
    let reasoning_tokens = usage_u32(
        completion_details
            .and_then(|details| usage_value(details, &["reasoning_tokens", "reasoningTokens"])),
    );
    let prompt_audio_tokens = usage_u32(
        prompt_details.and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let completion_audio_tokens = usage_u32(
        completion_details
            .and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let accepted_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["accepted_prediction_tokens", "acceptedPredictionTokens"],
        )
    }));
    let rejected_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["rejected_prediction_tokens", "rejectedPredictionTokens"],
        )
    }));

    Some(build_openai_compatible_chat_usage(
        OpenAiCompatibleChatUsageFields {
            raw,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cache_read_tokens: cache_read_tokens.unwrap_or(0),
            cache_write_tokens: None,
            no_cache_tokens: None,
            reasoning_tokens: reasoning_tokens.unwrap_or(0),
            output_text_tokens: None,
            prompt_detail_cached_tokens: cache_read_tokens,
            completion_detail_reasoning_tokens: reasoning_tokens,
            prompt_audio_tokens,
            completion_audio_tokens,
            accepted_prediction_tokens,
            rejected_prediction_tokens,
        },
    ))
}

fn parse_alibaba_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();

    let prompt_tokens =
        usage_u32(usage_value(&raw, &["prompt_tokens", "input_tokens"])).unwrap_or(0);
    let completion_tokens =
        usage_u32(usage_value(&raw, &["completion_tokens", "output_tokens"])).unwrap_or(0);
    let total_tokens = usage_u32(usage_value(&raw, &["total_tokens", "totalTokens"]));
    let prompt_details = usage_object(&raw, &["prompt_tokens_details", "input_tokens_details"]);
    let completion_details = usage_object(
        &raw,
        &["completion_tokens_details", "output_tokens_details"],
    );
    let cache_read_tokens = usage_u32(
        prompt_details.and_then(|details| usage_value(details, &["cached_tokens", "cachedTokens"])),
    );
    let cache_write_tokens = usage_u32(prompt_details.and_then(|details| {
        usage_value(
            details,
            &["cache_creation_input_tokens", "cacheCreationInputTokens"],
        )
    }))
    .unwrap_or(0);
    let reasoning_tokens = usage_u32(
        completion_details
            .and_then(|details| usage_value(details, &["reasoning_tokens", "reasoningTokens"])),
    );
    let prompt_audio_tokens = usage_u32(
        prompt_details.and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let completion_audio_tokens = usage_u32(
        completion_details
            .and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let accepted_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["accepted_prediction_tokens", "acceptedPredictionTokens"],
        )
    }));
    let rejected_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["rejected_prediction_tokens", "rejectedPredictionTokens"],
        )
    }));

    Some(build_openai_compatible_chat_usage(
        OpenAiCompatibleChatUsageFields {
            raw,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cache_read_tokens: cache_read_tokens.unwrap_or(0),
            cache_write_tokens: Some(cache_write_tokens),
            no_cache_tokens: None,
            reasoning_tokens: reasoning_tokens.unwrap_or(0),
            output_text_tokens: None,
            prompt_detail_cached_tokens: cache_read_tokens,
            completion_detail_reasoning_tokens: reasoning_tokens,
            prompt_audio_tokens,
            completion_audio_tokens,
            accepted_prediction_tokens,
            rejected_prediction_tokens,
        },
    ))
}

fn parse_xai_chat_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();
    let mut normalized = raw.clone();

    let prompt_tokens = usage_u32(usage_value(&raw, &["prompt_tokens", "input_tokens"]));
    let cache_read_tokens = usage_object(&raw, &["prompt_tokens_details", "input_tokens_details"])
        .and_then(|details| usage_u32(usage_value(details, &["cached_tokens", "cachedTokens"])))
        .unwrap_or(0);

    set_usage_detail_number(
        &mut normalized,
        "prompt_tokens_details",
        "cached_tokens",
        cache_read_tokens,
    );

    let normalized_input_total = prompt_tokens.map(|prompt_tokens| {
        let input_total = if cache_read_tokens > prompt_tokens {
            prompt_tokens.saturating_add(cache_read_tokens)
        } else {
            prompt_tokens
        };
        normalized.insert("prompt_tokens".to_string(), serde_json::json!(input_total));
        input_total
    });

    let completion_tokens = usage_u32(usage_value(&raw, &["completion_tokens", "output_tokens"]));
    let reasoning_tokens = usage_object(
        &raw,
        &["completion_tokens_details", "output_tokens_details"],
    )
    .and_then(|details| {
        usage_u32(usage_value(
            details,
            &["reasoning_tokens", "reasoningTokens"],
        ))
    })
    .unwrap_or(0);

    set_usage_detail_number(
        &mut normalized,
        "completion_tokens_details",
        "reasoning_tokens",
        reasoning_tokens,
    );

    let normalized_output_total = completion_tokens.map(|completion_tokens| {
        let output_total = completion_tokens.saturating_add(reasoning_tokens);
        normalized.insert(
            "completion_tokens".to_string(),
            serde_json::json!(output_total),
        );
        output_total
    });

    if let Some(total_tokens) = normalized_input_total
        .zip(normalized_output_total)
        .map(|(input, output)| input.saturating_add(output))
    {
        normalized.insert("total_tokens".to_string(), serde_json::json!(total_tokens));
    }

    parse_normalized_openai_usage_with_raw(normalized, raw)
}

/// Parse xAI Responses usage payloads into the unified `Usage` shape.
pub fn parse_xai_responses_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();
    let mut normalized = raw.clone();

    let input_tokens = usage_u32(usage_value(&raw, &["input_tokens", "prompt_tokens"]));
    let cache_read_tokens = usage_object(&raw, &["input_tokens_details", "prompt_tokens_details"])
        .and_then(|details| usage_u32(usage_value(details, &["cached_tokens", "cachedTokens"])))
        .unwrap_or(0);

    set_usage_detail_number(
        &mut normalized,
        "input_tokens_details",
        "cached_tokens",
        cache_read_tokens,
    );

    let normalized_input_total = input_tokens.map(|input_tokens| {
        let input_total = if cache_read_tokens > input_tokens {
            input_tokens.saturating_add(cache_read_tokens)
        } else {
            input_tokens
        };
        normalized.insert("input_tokens".to_string(), serde_json::json!(input_total));
        input_total
    });

    let reasoning_tokens = usage_object(&raw, &["output_tokens_details"])
        .and_then(|details| {
            usage_u32(usage_value(
                details,
                &["reasoning_tokens", "reasoningTokens"],
            ))
        })
        .unwrap_or(0);

    set_usage_detail_number(
        &mut normalized,
        "output_tokens_details",
        "reasoning_tokens",
        reasoning_tokens,
    );

    if let Some(total_tokens) = normalized_input_total
        .zip(usage_u32(usage_value(&raw, &["output_tokens"])))
        .map(|(input, output)| input.saturating_add(output))
    {
        normalized.insert("total_tokens".to_string(), serde_json::json!(total_tokens));
    }

    parse_normalized_openai_usage_with_raw(normalized, raw)
}

/// Return the xAI Responses fallback usage when the provider omits `usage`.
///
/// AI SDK treats missing xAI Responses usage as explicit zero counts instead of unknown counts.
pub fn xai_responses_zero_usage() -> Usage {
    Usage::builder()
        .with_input_tokens(UsageInputTokens {
            total: Some(0),
            no_cache: Some(0),
            cache_read: Some(0),
            cache_write: Some(0),
        })
        .with_output_tokens(UsageOutputTokens {
            total: Some(0),
            text: Some(0),
            reasoning: Some(0),
        })
        .build()
}

/// Extract xAI Responses usage-owned provider metadata.
///
/// AI SDK exposes `usage.cost_in_usd_ticks` as
/// `providerMetadata.xai.costInUsdTicks` instead of folding it into unified usage.
pub fn xai_responses_usage_provider_metadata_value(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    let cost = usage_value(object, &["cost_in_usd_ticks", "costInUsdTicks"])
        .filter(|value| !value.is_null())?;

    Some(serde_json::json!({ "costInUsdTicks": cost.clone() }))
}

fn normalize_deepinfra_usage_value<'a>(value: &'a Value) -> Cow<'a, Value> {
    let Some(object) = value.as_object() else {
        return Cow::Borrowed(value);
    };

    let completion_tokens = usage_u32(usage_value(object, &["completion_tokens", "output_tokens"]));
    let completion_details = usage_object(
        object,
        &["completion_tokens_details", "output_tokens_details"],
    );
    let reasoning_tokens = usage_u32(usage_value(
        object,
        &["reasoning_tokens", "reasoningTokens"],
    ))
    .or_else(|| {
        usage_u32(
            completion_details
                .and_then(|details| usage_value(details, &["reasoning_tokens", "reasoningTokens"])),
        )
    });

    let (Some(completion_tokens), Some(reasoning_tokens)) = (completion_tokens, reasoning_tokens)
    else {
        return Cow::Borrowed(value);
    };

    if reasoning_tokens <= completion_tokens {
        return Cow::Borrowed(value);
    }

    let corrected_completion_tokens = completion_tokens.saturating_add(reasoning_tokens);
    let mut fixed = object.clone();

    if fixed.contains_key("completion_tokens") {
        fixed.insert(
            "completion_tokens".to_string(),
            serde_json::json!(corrected_completion_tokens),
        );
    }
    if fixed.contains_key("output_tokens") {
        fixed.insert(
            "output_tokens".to_string(),
            serde_json::json!(corrected_completion_tokens),
        );
    }

    if let Some(total_tokens) = usage_u32(usage_value(object, &["total_tokens", "totalTokens"])) {
        let corrected_total_tokens = total_tokens.saturating_add(reasoning_tokens);
        if fixed.contains_key("total_tokens") {
            fixed.insert(
                "total_tokens".to_string(),
                serde_json::json!(corrected_total_tokens),
            );
        }
        if fixed.contains_key("totalTokens") {
            fixed.insert(
                "totalTokens".to_string(),
                serde_json::json!(corrected_total_tokens),
            );
        }
    }

    if let Some(output_tokens) = fixed.get_mut("outputTokens").and_then(Value::as_object_mut) {
        if !output_tokens.contains_key("text") && !output_tokens.contains_key("textTokens") {
            output_tokens.insert("text".to_string(), serde_json::json!(completion_tokens));
        }
        output_tokens.insert(
            if output_tokens.contains_key("totalTokens") {
                "totalTokens".to_string()
            } else {
                "total".to_string()
            },
            serde_json::json!(corrected_completion_tokens),
        );
    }

    Cow::Owned(Value::Object(fixed))
}

pub fn normalize_openai_usage_value_for_provider<'a>(
    provider_id: &str,
    value: &'a Value,
) -> Cow<'a, Value> {
    match provider_id {
        "deepinfra" => normalize_deepinfra_usage_value(value),
        _ => Cow::Borrowed(value),
    }
}

pub fn parse_provider_openai_usage_value(provider_id: &str, value: &Value) -> Option<Usage> {
    match provider_id {
        "alibaba" | "qwen" => parse_alibaba_usage_value(value),
        "deepseek" => parse_deepseek_usage_value(value),
        "groq" => parse_groq_usage_value(value),
        "moonshot" | "moonshotai" => parse_moonshotai_usage_value(value),
        "xai" => parse_xai_chat_usage_value(value),
        _ => {
            let normalized = normalize_openai_usage_value_for_provider(provider_id, value);
            parse_openai_compatible_chat_usage_value(normalized.as_ref())
        }
    }
}

pub fn extract_provider_openai_usage_value(provider_id: &str, raw: &Value) -> Option<Usage> {
    fn non_null_usage<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
        let mut cursor = value;
        for key in keys {
            cursor = cursor.get(*key)?;
        }
        (!cursor.is_null()).then_some(cursor)
    }

    if provider_id == "groq" {
        for usage in [
            non_null_usage(raw, &["x_groq", "usage"]),
            non_null_usage(raw, &["usage"]),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(parsed) = parse_provider_openai_usage_value(provider_id, usage) {
                return Some(parsed);
            }
        }

        return None;
    }

    non_null_usage(raw, &["usage"])
        .and_then(|usage| parse_provider_openai_usage_value(provider_id, usage))
}

/// Convert unified `Usage` into OpenAI Chat Completions usage JSON.
pub fn openai_chat_usage_value(usage: &Usage) -> Value {
    let normalized_input = usage.normalized_input_tokens();
    let normalized_output = usage.normalized_output_tokens();
    let prompt_tokens = normalized_input
        .total
        .or_else(|| usage.prompt_tokens_value());
    let completion_tokens = normalized_output
        .total
        .or_else(|| usage.completion_tokens_value());
    let total_tokens = usage.total_tokens_value().or_else(|| {
        prompt_tokens
            .zip(completion_tokens)
            .map(|(prompt, completion)| prompt.saturating_add(completion))
    });
    let mut object = stripped_raw_usage_object(
        usage,
        &[
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
            "output_tokens_details",
            "inputTokens",
            "outputTokens",
            "raw",
        ],
    );

    object.insert(
        "prompt_tokens".to_string(),
        prompt_tokens
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "completion_tokens".to_string(),
        completion_tokens
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "total_tokens".to_string(),
        total_tokens
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );

    if normalized_input.cache_read.is_some()
        || usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
            .is_some()
        || object
            .get("prompt_tokens_details")
            .and_then(Value::as_object)
            .is_some()
    {
        let details = ensure_object_entry(&mut object, "prompt_tokens_details");
        if let Some(cached_tokens) = normalized_input.cache_read {
            details.insert(
                "cached_tokens".to_string(),
                serde_json::json!(cached_tokens),
            );
        }
        if let Some(audio_tokens) = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
        {
            details.insert("audio_tokens".to_string(), serde_json::json!(audio_tokens));
        }
        if details.is_empty() {
            object.remove("prompt_tokens_details");
        }
    }

    if normalized_output.reasoning.is_some()
        || usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
            .is_some()
        || usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.accepted_prediction_tokens)
            .is_some()
        || usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.rejected_prediction_tokens)
            .is_some()
        || object
            .get("completion_tokens_details")
            .and_then(Value::as_object)
            .is_some()
    {
        let details = ensure_object_entry(&mut object, "completion_tokens_details");
        if let Some(reasoning_tokens) = normalized_output.reasoning {
            details.insert(
                "reasoning_tokens".to_string(),
                serde_json::json!(reasoning_tokens),
            );
        }
        if let Some(audio_tokens) = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
        {
            details.insert("audio_tokens".to_string(), serde_json::json!(audio_tokens));
        }
        if let Some(accepted_prediction_tokens) = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.accepted_prediction_tokens)
        {
            details.insert(
                "accepted_prediction_tokens".to_string(),
                serde_json::json!(accepted_prediction_tokens),
            );
        }
        if let Some(rejected_prediction_tokens) = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.rejected_prediction_tokens)
        {
            details.insert(
                "rejected_prediction_tokens".to_string(),
                serde_json::json!(rejected_prediction_tokens),
            );
        }
        if details.is_empty() {
            object.remove("completion_tokens_details");
        }
    }

    Value::Object(object)
}

/// Convert unified `Usage` into OpenAI Responses usage JSON.
pub fn openai_responses_usage_value(usage: &Usage) -> Value {
    let normalized_input = usage.normalized_input_tokens();
    let normalized_output = usage.normalized_output_tokens();
    let input_total = normalized_input
        .total
        .or_else(|| usage.prompt_tokens_value());
    let output_total = normalized_output
        .total
        .or_else(|| usage.completion_tokens_value());
    let total_tokens = usage.total_tokens_value().or_else(|| {
        input_total
            .zip(output_total)
            .map(|(input, output)| input.saturating_add(output))
    });
    let mut object = stripped_raw_usage_object(
        usage,
        &[
            "prompt_tokens",
            "prompt_tokens_details",
            "completion_tokens",
            "completion_tokens_details",
            "reasoning_tokens",
            "reasoningTokens",
            "inputTokens",
            "outputTokens",
            "raw",
        ],
    );

    object.insert(
        "input_tokens".to_string(),
        input_total
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "output_tokens".to_string(),
        output_total
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "total_tokens".to_string(),
        total_tokens
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );

    if normalized_input.cache_read.is_some()
        || object
            .get("input_tokens_details")
            .and_then(Value::as_object)
            .is_some()
    {
        let details = ensure_object_entry(&mut object, "input_tokens_details");
        if let Some(cached_tokens) = normalized_input.cache_read {
            details.insert(
                "cached_tokens".to_string(),
                serde_json::json!(cached_tokens),
            );
        }
        if details.is_empty() {
            object.remove("input_tokens_details");
        }
    }

    if normalized_output.reasoning.is_some()
        || object
            .get("output_tokens_details")
            .and_then(Value::as_object)
            .is_some()
    {
        let details = ensure_object_entry(&mut object, "output_tokens_details");
        if let Some(reasoning_tokens) = normalized_output.reasoning {
            details.insert(
                "reasoning_tokens".to_string(),
                serde_json::json!(reasoning_tokens),
            );
        }
        if details.is_empty() {
            object.remove("output_tokens_details");
        }
    }

    Value::Object(object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatMessage, FilePartSource, MessageMetadata, ProviderOptionsMap, ProviderReference,
        ToolResultContentPart,
    };
    use std::collections::HashMap;

    #[test]
    fn openai_chat_pdf_file_part_maps_to_file_content_part() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::base64("Zm9v"),
                media_type: "application/pdf".to_string(),
                filename: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };

        let out = convert_messages_openai_chat(&[msg]).expect("convert messages");
        assert_eq!(out.len(), 1);
        let content = out[0].content.clone().expect("content");
        let parts = content.as_array().expect("array");
        assert_eq!(parts[0]["type"], "file");
        assert_eq!(parts[0]["file"]["filename"], "part-0.pdf");
        assert_eq!(
            parts[0]["file"]["file_data"],
            "data:application/pdf;base64,Zm9v"
        );
    }

    #[test]
    fn openai_chat_pdf_provider_reference_maps_to_file_id() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::provider_reference(ProviderReference::single(
                    "openai", "file-abc",
                )),
                media_type: "application/pdf".to_string(),
                filename: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };

        let out = convert_messages_openai_chat(&[msg]).expect("convert messages");
        let content = out[0].content.clone().expect("content");
        let parts = content.as_array().expect("array");
        assert_eq!(parts[0]["type"], "file");
        assert_eq!(parts[0]["file"]["file_id"], "file-abc");
    }

    #[test]
    fn openai_chat_image_provider_reference_maps_to_file_id() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::Image {
                source: FilePartSource::provider_reference(ProviderReference::single(
                    "openai",
                    "file-image",
                )),
                media_type: None,
                detail: Some(crate::types::ImageDetail::High),
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let out = convert_messages_openai_chat(&[msg]).expect("convert messages");
        let content = out[0].content.clone().expect("content");
        let parts = content.as_array().expect("array");
        assert_eq!(parts[0]["type"], "file");
        assert_eq!(parts[0]["file"]["file_id"], "file-image");
        assert!(parts[0]["image_url"].is_null());
    }

    #[test]
    fn openai_chat_audio_file_part_maps_to_input_audio() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::base64("AAEC"),
                media_type: "audio/mpeg".to_string(),
                filename: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };

        let out = convert_messages_openai_chat(&[msg]).expect("convert messages");
        let content = out[0].content.clone().expect("content");
        let parts = content.as_array().expect("array");
        assert_eq!(parts[0]["type"], "input_audio");
        assert_eq!(parts[0]["input_audio"]["data"], "AAEC");
        assert_eq!(parts[0]["input_audio"]["format"], "mp3");
    }

    #[test]
    fn openai_chat_ignores_legacy_image_detail_provider_metadata() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::base64("AAEC"),
                media_type: "image/png".to_string(),
                filename: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    serde_json::json!({
                        "imageDetail": "low"
                    }),
                )])),
            }]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let out = convert_messages_openai_chat(&[msg]).expect("convert messages");
        let content = out[0].content.clone().expect("content");
        let parts = content.as_array().expect("array");
        assert!(parts[0]["image_url"].get("detail").is_none());
    }

    #[test]
    fn openai_compatible_single_text_ignores_legacy_request_metadata_channels() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::Text {
                text: "Hello".to_string(),
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: Some(HashMap::from([(
                    "openaiCompatible".to_string(),
                    serde_json::json!({
                        "sharedKey": "legacy-part"
                    }),
                )])),
            }]),
            metadata: MessageMetadata {
                id: None,
                timestamp: None,
                cache_control: None,
                custom: HashMap::from([(
                    "openaiCompatible".to_string(),
                    serde_json::json!({
                        "sharedKey": "legacy-message"
                    }),
                )]),
            },
            provider_options: ProviderOptionsMap::default(),
        };

        let out = convert_messages(&[msg]).expect("convert messages");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "user");
        assert_eq!(
            out[0].content,
            Some(serde_json::Value::String("Hello".to_string()))
        );
        assert!(out[0].extra.is_empty());
    }

    #[test]
    fn openai_compatible_assistant_reasoning_parts_map_to_reasoning_content() {
        let msg = ChatMessage {
            role: MessageRole::Assistant,
            content: MessageContent::MultiModal(vec![
                ContentPart::Text {
                    text: "Final answer.".to_string(),
                    provider_options: ProviderOptionsMap::default(),
                    provider_metadata: None,
                },
                ContentPart::Reasoning {
                    text: "step-1".to_string(),
                    provider_options: ProviderOptionsMap::default(),
                    provider_metadata: None,
                },
                ContentPart::Reasoning {
                    text: "step-2".to_string(),
                    provider_options: ProviderOptionsMap::default(),
                    provider_metadata: None,
                },
            ]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let out = convert_messages(&[msg]).expect("convert messages");
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].content,
            Some(serde_json::Value::String("Final answer.".to_string()))
        );
        assert_eq!(
            out[0].extra.get("reasoning_content"),
            Some(&serde_json::json!("step-1step-2"))
        );
    }

    #[test]
    fn openai_tool_messages_keep_explicit_tool_result_content_variants_as_json_strings() {
        let message = ChatMessage {
            role: MessageRole::Tool,
            content: MessageContent::MultiModal(vec![ContentPart::tool_result_content(
                "call_1",
                "render_asset",
                vec![
                    ToolResultContentPart::text("done"),
                    ToolResultContentPart::file_data(
                        "JVBERi0x",
                        "application/pdf",
                        Some("report.pdf".to_string()),
                    ),
                    ToolResultContentPart::file_url("https://example.com/report.pdf"),
                    ToolResultContentPart::file_id(HashMap::from([(
                        "openai".to_string(),
                        "file_openai".to_string(),
                    )])),
                    ToolResultContentPart::image_data("aGVsbG8=", "image/png"),
                    ToolResultContentPart::image_url("https://example.com/image.png"),
                    ToolResultContentPart::image_file_id(HashMap::from([(
                        "openai".to_string(),
                        "image_openai".to_string(),
                    )])),
                    ToolResultContentPart::custom().with_provider_option(
                        "anthropic",
                        serde_json::json!({
                            "type": "tool-reference",
                        }),
                    ),
                ],
            )]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };

        for converted in [
            convert_messages(&[message.clone()]).expect("convert openai-compatible messages"),
            convert_messages_openai_chat(&[message.clone()]).expect("convert openai chat messages"),
        ] {
            assert_eq!(converted.len(), 1);
            assert_eq!(converted[0].role, "tool");
            assert_eq!(converted[0].tool_call_id.as_deref(), Some("call_1"));

            let content = converted[0]
                .content
                .as_ref()
                .and_then(|value| value.as_str())
                .expect("tool message content should be string");
            let parsed: serde_json::Value =
                serde_json::from_str(content).expect("parse tool content json string");
            let parts = parsed.as_array().expect("tool content array");

            assert_eq!(parts[0]["type"], serde_json::json!("text"));
            assert_eq!(parts[0]["text"], serde_json::json!("done"));
            assert_eq!(parts[1]["type"], serde_json::json!("file-data"));
            assert_eq!(parts[1]["mediaType"], serde_json::json!("application/pdf"));
            assert_eq!(parts[1]["filename"], serde_json::json!("report.pdf"));
            assert_eq!(parts[2]["type"], serde_json::json!("file-url"));
            assert_eq!(
                parts[2]["url"],
                serde_json::json!("https://example.com/report.pdf")
            );
            assert_eq!(parts[3]["type"], serde_json::json!("file-id"));
            assert_eq!(
                parts[3]["fileId"],
                serde_json::json!({ "openai": "file_openai" })
            );
            assert_eq!(parts[4]["type"], serde_json::json!("image-data"));
            assert_eq!(parts[4]["mediaType"], serde_json::json!("image/png"));
            assert_eq!(parts[5]["type"], serde_json::json!("image-url"));
            assert_eq!(
                parts[5]["url"],
                serde_json::json!("https://example.com/image.png")
            );
            assert_eq!(parts[6]["type"], serde_json::json!("image-file-id"));
            assert_eq!(
                parts[6]["fileId"],
                serde_json::json!({ "openai": "image_openai" })
            );
            assert_eq!(parts[7]["type"], serde_json::json!("custom"));
            assert_eq!(
                parts[7]["providerOptions"]["anthropic"]["type"],
                serde_json::json!("tool-reference")
            );
        }
    }

    #[test]
    fn parse_openai_usage_value_supports_ai_sdk_wrapper_with_raw_usage() {
        let usage = parse_openai_usage_value(&serde_json::json!({
            "inputTokens": {
                "total": 12,
                "noCache": 9,
                "cacheRead": 3
            },
            "outputTokens": {
                "total": 8,
                "text": 5,
                "reasoning": 3
            },
            "raw": {
                "input_tokens": 12,
                "input_tokens_details": {
                    "cached_tokens": 3
                },
                "output_tokens": 8,
                "output_tokens_details": {
                    "reasoning_tokens": 3
                },
                "total_tokens": 20
            }
        }))
        .expect("parse usage");

        assert_eq!(usage.prompt_tokens(), Some(12));
        assert_eq!(usage.completion_tokens(), Some(8));
        assert_eq!(usage.total_tokens(), Some(20));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(9));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(3));
        assert_eq!(usage.normalized_output_tokens().text, Some(5));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(3));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["input_tokens"],
            serde_json::json!(12)
        );
    }

    #[test]
    fn parse_provider_openai_usage_value_defaults_generic_chat_usage_like_ai_sdk() {
        let usage = parse_provider_openai_usage_value(
            "openrouter",
            &serde_json::json!({
                "prompt_tokens": 100,
                "completion_tokens": 50
            }),
        )
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(100));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(100));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_write, None);
        assert_eq!(usage.normalized_output_tokens().total, Some(50));
        assert_eq!(usage.normalized_output_tokens().text, Some(50));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(0));
    }

    #[test]
    fn extract_provider_openai_usage_value_reads_groq_streaming_x_groq_usage() {
        let usage = extract_provider_openai_usage_value(
            "groq",
            &serde_json::json!({
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                "x_groq": {
                    "usage": {
                        "queue_time": 0.061348671,
                        "prompt_tokens": 18,
                        "completion_tokens": 439,
                        "total_tokens": 457
                    }
                }
            }),
        )
        .expect("parse x_groq usage");

        assert_eq!(usage.prompt_tokens(), Some(18));
        assert_eq!(usage.completion_tokens(), Some(439));
        assert_eq!(usage.total_tokens(), Some(457));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["queue_time"],
            serde_json::json!(0.061348671)
        );
    }

    #[test]
    fn parse_provider_openai_finish_reason_matches_ai_sdk_vendor_mappings() {
        assert_eq!(
            parse_provider_openai_finish_reason("deepseek", Some("insufficient_system_resource")),
            Some(FinishReason::Error)
        );
        assert_eq!(
            parse_provider_openai_finish_reason("mistral", Some("model_length")),
            Some(FinishReason::Length)
        );
        assert_eq!(
            parse_provider_openai_finish_reason("perplexity", Some("content_filter")),
            Some(FinishReason::Other("content_filter".to_string()))
        );
        assert_eq!(
            parse_provider_openai_finish_reason("cohere", Some("STOP_SEQUENCE")),
            Some(FinishReason::Stop)
        );
        assert_eq!(
            parse_provider_openai_finish_reason("groq", Some("function_call")),
            Some(FinishReason::ToolCalls)
        );
    }

    #[test]
    fn parse_provider_openai_usage_value_fixes_deepinfra_reasoning_totals() {
        let usage = parse_provider_openai_usage_value(
            "deepinfra",
            &serde_json::json!({
                "prompt_tokens": 21,
                "completion_tokens": 84,
                "total_tokens": 105,
                "completion_tokens_details": {
                    "reasoning_tokens": 1081
                }
            }),
        )
        .expect("parse usage");

        assert_eq!(usage.prompt_tokens(), Some(21));
        assert_eq!(usage.completion_tokens(), Some(1165));
        assert_eq!(usage.total_tokens(), Some(1186));
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens),
            Some(1081)
        );
        assert_eq!(usage.normalized_output_tokens().total, Some(1165));
        assert_eq!(usage.normalized_output_tokens().text, Some(84));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(1081));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["completion_tokens"],
            serde_json::json!(1165)
        );
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["total_tokens"],
            serde_json::json!(1186)
        );
    }

    #[test]
    fn parse_provider_openai_usage_value_maps_alibaba_cache_creation_tokens() {
        let usage = parse_provider_openai_usage_value(
            "qwen",
            &serde_json::json!({
                "prompt_tokens": 200,
                "completion_tokens": 75,
                "total_tokens": 275,
                "prompt_tokens_details": {
                    "cached_tokens": 120,
                    "cache_creation_input_tokens": 50
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 25
                }
            }),
        )
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(200));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(30));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(120));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(50));
        assert_eq!(usage.normalized_output_tokens().total, Some(75));
        assert_eq!(usage.normalized_output_tokens().text, Some(50));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(25));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["prompt_tokens_details"]["cache_creation_input_tokens"],
            serde_json::json!(50)
        );
    }

    #[test]
    fn parse_provider_openai_usage_value_maps_alibaba_missing_cache_write_to_zero() {
        let usage = parse_provider_openai_usage_value(
            "alibaba",
            &serde_json::json!({
                "prompt_tokens": 200,
                "completion_tokens": 75,
                "prompt_tokens_details": {
                    "cached_tokens": 120
                }
            }),
        )
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(200));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(80));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(120));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(0));
        assert_eq!(usage.normalized_output_tokens().total, Some(75));
        assert_eq!(usage.normalized_output_tokens().text, Some(75));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(0));
    }

    #[test]
    fn parse_provider_openai_usage_value_maps_deepseek_prompt_cache_hits() {
        let usage = parse_provider_openai_usage_value(
            "deepseek",
            &serde_json::json!({
                "prompt_tokens": 495,
                "completion_tokens": 157,
                "total_tokens": 652,
                "prompt_cache_hit_tokens": 320,
                "prompt_cache_miss_tokens": 175,
                "completion_tokens_details": {
                    "reasoning_tokens": 118
                }
            }),
        )
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(495));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(175));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(320));
        assert_eq!(usage.normalized_output_tokens().total, Some(157));
        assert_eq!(usage.normalized_output_tokens().text, Some(39));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(118));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["prompt_cache_hit_tokens"],
            serde_json::json!(320)
        );
    }

    #[test]
    fn parse_provider_openai_usage_value_prefers_moonshot_top_level_cached_tokens() {
        let usage = parse_provider_openai_usage_value(
            "moonshotai",
            &serde_json::json!({
                "prompt_tokens": 100,
                "completion_tokens": 80,
                "cached_tokens": 35,
                "prompt_tokens_details": {
                    "cached_tokens": 25
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 30
                }
            }),
        )
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(100));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(65));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(35));
        assert_eq!(usage.normalized_output_tokens().total, Some(80));
        assert_eq!(usage.normalized_output_tokens().text, Some(50));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(30));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["prompt_tokens_details"]["cached_tokens"],
            serde_json::json!(25)
        );
    }

    #[test]
    fn parse_provider_openai_usage_value_maps_moonshot_null_fields_to_zero() {
        let usage = parse_provider_openai_usage_value(
            "moonshotai",
            &serde_json::json!({
                "prompt_tokens": null,
                "completion_tokens": null,
                "cached_tokens": null
            }),
        )
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(0));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_write, None);
        assert_eq!(usage.normalized_output_tokens().total, Some(0));
        assert_eq!(usage.normalized_output_tokens().text, Some(0));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(0));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["prompt_tokens"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn parse_provider_openai_usage_value_maps_xai_chat_reasoning_and_cache_semantics() {
        let usage = parse_provider_openai_usage_value(
            "xai",
            &serde_json::json!({
                "prompt_tokens": 4142,
                "completion_tokens": 254,
                "total_tokens": 8734,
                "prompt_tokens_details": {
                    "cached_tokens": 4328
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 10
                }
            }),
        )
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(8470));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(4142));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(4328));
        assert_eq!(usage.normalized_output_tokens().total, Some(264));
        assert_eq!(usage.normalized_output_tokens().text, Some(254));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(10));
        assert_eq!(usage.total_tokens(), Some(8734));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["completion_tokens"],
            serde_json::json!(254)
        );
    }

    #[test]
    fn parse_provider_openai_usage_value_keeps_groq_prompt_cache_details_raw_only() {
        let usage = parse_provider_openai_usage_value(
            "groq",
            &serde_json::json!({
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "prompt_tokens_details": {
                    "cached_tokens": 5
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 3
                }
            }),
        )
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(20));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(20));
        assert_eq!(usage.normalized_input_tokens().cache_read, None);
        assert_eq!(usage.normalized_output_tokens().total, Some(10));
        assert_eq!(usage.normalized_output_tokens().text, Some(7));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(3));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["prompt_tokens_details"]["cached_tokens"],
            serde_json::json!(5)
        );
    }

    #[test]
    fn parse_xai_responses_usage_value_handles_noninclusive_cache_read_tokens() {
        let usage = parse_xai_responses_usage_value(&serde_json::json!({
            "input_tokens": 4142,
            "output_tokens": 254,
            "total_tokens": 4396,
            "input_tokens_details": {
                "cached_tokens": 4328
            },
            "output_tokens_details": {
                "reasoning_tokens": 10
            }
        }))
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(8470));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(4142));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(4328));
        assert_eq!(usage.normalized_output_tokens().total, Some(254));
        assert_eq!(usage.normalized_output_tokens().text, Some(244));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(10));
        assert_eq!(usage.total_tokens(), Some(8724));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["input_tokens"],
            serde_json::json!(4142)
        );
    }

    #[test]
    fn xai_responses_zero_usage_matches_ai_sdk_missing_usage_fallback() {
        let usage = xai_responses_zero_usage();

        assert_eq!(usage.normalized_input_tokens().total, Some(0));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(0));
        assert_eq!(usage.normalized_output_tokens().total, Some(0));
        assert_eq!(usage.normalized_output_tokens().text, Some(0));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(0));
        assert!(usage.raw_usage_value().is_none());
    }

    #[test]
    fn xai_responses_usage_provider_metadata_maps_cost_ticks() {
        let metadata = xai_responses_usage_provider_metadata_value(&serde_json::json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_in_usd_ticks": 113500
        }))
        .expect("provider metadata");

        assert_eq!(metadata["costInUsdTicks"], serde_json::json!(113500));
    }

    #[test]
    fn openai_chat_usage_value_preserves_vendor_extensions_and_strips_responses_keys() {
        let usage = Usage::builder()
            .prompt_tokens(10)
            .completion_tokens(8)
            .total_tokens(18)
            .with_input_total_tokens(10)
            .with_input_no_cache_tokens(6)
            .with_input_cache_read_tokens(4)
            .with_output_total_tokens(8)
            .with_output_text_tokens(5)
            .with_output_reasoning_tokens(3)
            .with_raw_usage_value(serde_json::json!({
                "citation_tokens": 7,
                "input_tokens": 99,
                "output_tokens": 42
            }))
            .build();

        let value = openai_chat_usage_value(&usage);
        assert_eq!(value["prompt_tokens"], serde_json::json!(10));
        assert_eq!(value["completion_tokens"], serde_json::json!(8));
        assert_eq!(
            value["prompt_tokens_details"]["cached_tokens"],
            serde_json::json!(4)
        );
        assert_eq!(
            value["completion_tokens_details"]["reasoning_tokens"],
            serde_json::json!(3)
        );
        assert_eq!(value["citation_tokens"], serde_json::json!(7));
        assert!(value.get("input_tokens").is_none());
        assert!(value.get("output_tokens").is_none());
    }

    #[test]
    fn openai_responses_usage_value_preserves_vendor_extensions_and_strips_chat_keys() {
        let usage = Usage::builder()
            .prompt_tokens(11)
            .completion_tokens(9)
            .total_tokens(20)
            .with_cached_tokens(2)
            .with_reasoning_tokens(4)
            .with_raw_usage_value(serde_json::json!({
                "citation_tokens": 5,
                "prompt_tokens": 111,
                "completion_tokens": 222
            }))
            .build();

        let value = openai_responses_usage_value(&usage);
        assert_eq!(value["input_tokens"], serde_json::json!(11));
        assert_eq!(value["output_tokens"], serde_json::json!(9));
        assert_eq!(
            value["input_tokens_details"]["cached_tokens"],
            serde_json::json!(2)
        );
        assert_eq!(
            value["output_tokens_details"]["reasoning_tokens"],
            serde_json::json!(4)
        );
        assert_eq!(value["citation_tokens"], serde_json::json!(5));
        assert!(value.get("prompt_tokens").is_none());
        assert!(value.get("completion_tokens").is_none());
    }

    #[test]
    fn openai_responses_usage_value_preserves_unknown_totals_as_null() {
        let usage = Usage::builder()
            .with_raw_usage_value(serde_json::json!({
                "input_tokens": null,
                "output_tokens": null,
                "total_tokens": null
            }))
            .build();

        let value = openai_responses_usage_value(&usage);
        assert_eq!(value["input_tokens"], serde_json::Value::Null);
        assert_eq!(value["output_tokens"], serde_json::Value::Null);
        assert_eq!(value["total_tokens"], serde_json::Value::Null);
    }

    #[test]
    fn openai_compatible_pdf_file_part_is_still_unsupported() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::base64("Zm9v"),
                media_type: "application/pdf".to_string(),
                filename: None,
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: crate::types::ProviderOptionsMap::default(),
        };

        let err = convert_messages(&[msg]).unwrap_err();
        assert!(matches!(err, LlmError::UnsupportedOperation(_)));
    }

    #[test]
    fn openai_compatible_image_provider_reference_is_unsupported() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::Image {
                source: FilePartSource::provider_reference(ProviderReference::single(
                    "openai",
                    "file-image",
                )),
                media_type: None,
                detail: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let err = convert_messages(&[msg]).expect_err("expected unsupported provider reference");
        assert!(matches!(
            err,
            LlmError::UnsupportedOperation(message)
                if message.contains("provider references")
        ));
    }
}
