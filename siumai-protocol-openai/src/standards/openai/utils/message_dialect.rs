//! OpenAI-compatible chat message dialect conversion.
//!
//! This private module keeps provider-specific request message projection separate from the
//! broader utility surface for tools, response formats, finish reasons, and usage.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageConversionTarget {
    /// Match Vercel `@ai-sdk/openai-compatible` behavior.
    OpenAiCompatible,
    /// Match Vercel `@ai-sdk/openai` chat message conversion behavior.
    OpenAiChat,
    /// Match Vercel `@ai-sdk/xai` chat message conversion behavior.
    XaiChat,
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

fn resolve_openai_chat_provider_reference(
    provider_reference: &ProviderReference,
) -> Result<&str, LlmError> {
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

fn resolve_provider_reference<'a>(
    provider_reference: &'a ProviderReference,
    provider_id: &str,
) -> Result<&'a str, LlmError> {
    provider_reference.get(provider_id).ok_or_else(|| {
        let available = provider_reference.available_providers();
        let available = if available.is_empty() {
            "none".to_string()
        } else {
            available.join(", ")
        };

        LlmError::InvalidParameter(format!(
            "No provider reference found for provider '{provider_id}'. Available providers: {available}"
        ))
    })
}

fn media_source_as_data_or_url(
    source: &FilePartSource,
    media_type: &str,
) -> Result<String, LlmError> {
    match source {
        FilePartSource::Media(MediaSource::Url { url }) => Ok(url.clone()),
        FilePartSource::Media(MediaSource::Base64 { data }) => {
            if data.starts_with("data:") {
                Ok(data.clone())
            } else {
                Ok(format!("data:{media_type};base64,{data}"))
            }
        }
        FilePartSource::Media(MediaSource::Binary { data }) => {
            let encoded = base64::engine::general_purpose::STANDARD.encode(data);
            Ok(format!("data:{media_type};base64,{encoded}"))
        }
        FilePartSource::ProviderReference { .. } => Err(LlmError::UnsupportedOperation(
            "file parts with provider references".to_string(),
        )),
    }
}

fn media_source_as_base64(
    source: &FilePartSource,
    unsupported_url: &str,
) -> Result<String, LlmError> {
    match source {
        FilePartSource::Media(MediaSource::Url { .. }) => {
            Err(LlmError::UnsupportedOperation(unsupported_url.to_string()))
        }
        FilePartSource::Media(MediaSource::Base64 { data }) => Ok(data.clone()),
        FilePartSource::Media(MediaSource::Binary { data }) => {
            Ok(base64::engine::general_purpose::STANDARD.encode(data))
        }
        FilePartSource::ProviderReference { .. } => Err(LlmError::UnsupportedOperation(
            "file parts with provider references".to_string(),
        )),
    }
}

fn decode_text_file_source(source: &FilePartSource) -> Result<String, LlmError> {
    match source {
        FilePartSource::Media(MediaSource::Url { url }) => Ok(url.clone()),
        FilePartSource::Media(MediaSource::Base64 { data }) => {
            let raw = data
                .split_once(',')
                .map_or(data.as_str(), |(_, payload)| payload);
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(raw)
                .map_err(|err| {
                    LlmError::InvalidParameter(format!("Invalid base64 text file data: {err}"))
                })?;
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
        FilePartSource::Media(MediaSource::Binary { data }) => {
            Ok(String::from_utf8_lossy(data).into_owned())
        }
        FilePartSource::ProviderReference { .. } => Err(LlmError::UnsupportedOperation(
            "file parts with provider references".to_string(),
        )),
    }
}

fn openai_compatible_audio_format(media_type: &str) -> Result<&'static str, LlmError> {
    match media_type {
        "audio/wav" => Ok("wav"),
        "audio/mp3" | "audio/mpeg" => Ok("mp3"),
        _ => Err(LlmError::UnsupportedOperation(format!(
            "audio media type {media_type}"
        ))),
    }
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
                        media_type,
                        detail,
                        provider_options,
                        ..
                    } => {
                        let normalized_media_type = media_type.as_deref().unwrap_or("image/jpeg");
                        let url = match source {
                            FilePartSource::Media(MediaSource::Url { url }) => url.clone(),
                            FilePartSource::Media(MediaSource::Base64 { data }) => {
                                if data.starts_with("data:") {
                                    data.clone()
                                } else {
                                    format!("data:{normalized_media_type};base64,{data}")
                                }
                            }
                            FilePartSource::Media(MediaSource::Binary { data }) => {
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                format!("data:{normalized_media_type};base64,{encoded}")
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
                                    MessageConversionTarget::XaiChat => {
                                        content_parts.push(serde_json::json!({
                                            "type": "file",
                                            "file": {
                                                "file_id": resolve_provider_reference(provider_reference, "xai")?,
                                            }
                                        }));
                                        continue;
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
                        source,
                        media_type,
                        provider_options,
                        ..
                    } => {
                        let media_type = media_type.as_deref().unwrap_or("audio/wav");
                        let format = match target {
                            MessageConversionTarget::OpenAiCompatible => {
                                openai_compatible_audio_format(media_type)?
                            }
                            MessageConversionTarget::OpenAiChat => {
                                openai_chat_audio_format(media_type)?
                            }
                            _ => {
                                return Err(LlmError::UnsupportedOperation(format!(
                                    "file part media type {media_type}"
                                )));
                            }
                        };

                        let data = match source {
                            MediaSource::Url { .. } => {
                                return Err(LlmError::UnsupportedOperation(
                                    "audio file parts with URLs".to_string(),
                                ));
                            }
                            MediaSource::Base64 { data } => data.clone(),
                            MediaSource::Binary { data } => {
                                base64::engine::general_purpose::STANDARD.encode(data)
                            }
                        };

                        let mut audio_obj = serde_json::json!({
                            "type": "input_audio",
                            "input_audio": { "data": data, "format": format }
                        });

                        if target == MessageConversionTarget::OpenAiCompatible
                            && let serde_json::Value::Object(ref mut obj) = audio_obj
                        {
                            merge_openai_compatible_json(obj, Some(provider_options));
                        }

                        content_parts.push(audio_obj);
                    }
                    ContentPart::File {
                        source,
                        media_type,
                        provider_options,
                        filename,
                        ..
                    } => {
                        if target == MessageConversionTarget::XaiChat
                            && let FilePartSource::ProviderReference { provider_reference } = source
                        {
                            content_parts.push(serde_json::json!({
                                "type": "file",
                                "file": {
                                    "file_id": resolve_provider_reference(provider_reference, "xai")?,
                                }
                            }));
                            continue;
                        }

                        if media_type.starts_with("image/") {
                            let normalized_media_type = if media_type == "image/*" {
                                "image/jpeg"
                            } else {
                                media_type.as_str()
                            };

                            let url = match source {
                                FilePartSource::Media(_) => {
                                    media_source_as_data_or_url(source, normalized_media_type)?
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
                                        MessageConversionTarget::XaiChat => {
                                            content_parts.push(serde_json::json!({
                                                "type": "file",
                                                "file": {
                                                    "file_id": resolve_provider_reference(provider_reference, "xai")?,
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
                        } else if matches!(
                            target,
                            MessageConversionTarget::OpenAiChat
                                | MessageConversionTarget::OpenAiCompatible
                        ) && media_type.starts_with("audio/")
                        {
                            let format = match target {
                                MessageConversionTarget::OpenAiCompatible => {
                                    openai_compatible_audio_format(media_type)?
                                }
                                MessageConversionTarget::OpenAiChat => {
                                    openai_chat_audio_format(media_type)?
                                }
                                _ => unreachable!(),
                            };

                            let data = match source {
                                FilePartSource::ProviderReference { .. } => {
                                    return Err(LlmError::UnsupportedOperation(
                                        "audio file parts with provider references".to_string(),
                                    ));
                                }
                                _ => media_source_as_base64(source, "audio file parts with URLs")?,
                            };

                            let mut audio_obj = serde_json::json!({
                                "type": "input_audio",
                                "input_audio": { "data": data, "format": format }
                            });

                            if target == MessageConversionTarget::OpenAiCompatible
                                && let serde_json::Value::Object(ref mut obj) = audio_obj
                            {
                                merge_openai_compatible_json(obj, Some(provider_options));
                            }

                            content_parts.push(audio_obj);
                        } else if matches!(
                            target,
                            MessageConversionTarget::OpenAiChat
                                | MessageConversionTarget::OpenAiCompatible
                        ) && media_type == "application/pdf"
                        {
                            match source {
                                FilePartSource::Media(MediaSource::Url { .. }) => {
                                    return Err(LlmError::UnsupportedOperation(
                                        "PDF file parts with URLs".to_string(),
                                    ));
                                }
                                FilePartSource::Media(MediaSource::Base64 { data }) => {
                                    let default_filename = match target {
                                        MessageConversionTarget::OpenAiCompatible => {
                                            "document.pdf".to_string()
                                        }
                                        _ => format!("part-{index}.pdf"),
                                    };
                                    let file = serde_json::json!({
                                        "filename": filename.clone().unwrap_or(default_filename),
                                        "file_data": format!("data:application/pdf;base64,{data}"),
                                    });
                                    let mut file_obj =
                                        serde_json::json!({ "type": "file", "file": file });
                                    if target == MessageConversionTarget::OpenAiCompatible
                                        && let serde_json::Value::Object(ref mut obj) = file_obj
                                    {
                                        merge_openai_compatible_json(obj, Some(provider_options));
                                    }
                                    content_parts.push(file_obj);
                                }
                                FilePartSource::Media(MediaSource::Binary { data }) => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    let default_filename = match target {
                                        MessageConversionTarget::OpenAiCompatible => {
                                            "document.pdf".to_string()
                                        }
                                        _ => format!("part-{index}.pdf"),
                                    };
                                    let file = serde_json::json!({
                                        "filename": filename.clone().unwrap_or(default_filename),
                                        "file_data": format!("data:application/pdf;base64,{encoded}"),
                                    });
                                    let mut file_obj =
                                        serde_json::json!({ "type": "file", "file": file });
                                    if target == MessageConversionTarget::OpenAiCompatible
                                        && let serde_json::Value::Object(ref mut obj) = file_obj
                                    {
                                        merge_openai_compatible_json(obj, Some(provider_options));
                                    }
                                    content_parts.push(file_obj);
                                }
                                FilePartSource::ProviderReference { provider_reference } => {
                                    if target == MessageConversionTarget::OpenAiChat {
                                        content_parts.push(serde_json::json!({
                                            "type": "file",
                                            "file": {
                                                "file_id": resolve_openai_chat_provider_reference(provider_reference)?,
                                            }
                                        }));
                                    } else {
                                        return Err(LlmError::UnsupportedOperation(
                                            "file parts with provider references".to_string(),
                                        ));
                                    }
                                }
                            }
                        } else if target == MessageConversionTarget::OpenAiCompatible
                            && media_type.starts_with("text/")
                        {
                            let text = decode_text_file_source(source)?;
                            let mut text_obj = serde_json::json!({
                                "type": "text",
                                "text": text,
                            });
                            if let serde_json::Value::Object(ref mut obj) = text_obj {
                                merge_openai_compatible_json(obj, Some(provider_options));
                            }
                            content_parts.push(text_obj);
                        } else {
                            if target == MessageConversionTarget::OpenAiChat {
                                return Err(LlmError::UnsupportedOperation(format!(
                                    "file part media type {media_type}"
                                )));
                            }
                            return Err(LlmError::UnsupportedOperation(format!(
                                "file part media type {media_type}"
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

/// Convert messages for the Perplexity chat-completions shape used by the AI SDK provider.
///
/// Perplexity differs from the generic OpenAI-compatible conversion in two important ways:
/// text-only user/assistant content is collapsed into a single string, while image/PDF file
/// content uses Perplexity's `image_url` / `file_url` content-part shape.
pub fn convert_messages_perplexity_chat(
    messages: &[ChatMessage],
) -> Result<Vec<OpenAiMessage>, LlmError> {
    let mut out = Vec::new();

    for message in messages {
        match message.role {
            MessageRole::System | MessageRole::Developer => {
                out.push(OpenAiMessage {
                    role: match message.role {
                        MessageRole::Developer => "developer",
                        _ => "system",
                    }
                    .to_string(),
                    content: Some(serde_json::Value::String(perplexity_text_content(
                        &message.content,
                    ))),
                    tool_calls: None,
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::User => {
                out.push(OpenAiMessage {
                    role: "user".to_string(),
                    content: Some(perplexity_message_content(&message.content)?),
                    tool_calls: None,
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::Assistant => {
                out.push(OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(perplexity_message_content(&message.content)?),
                    tool_calls: plain_assistant_tool_calls(message),
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::Tool => {
                out.extend(convert_messages_with_target(
                    std::slice::from_ref(message),
                    MessageConversionTarget::OpenAiCompatible,
                )?);
            }
        }
    }

    Ok(out)
}

/// Convert messages for the DeepSeek chat-completions shape used by the AI SDK provider.
///
/// DeepSeek is OpenAI-compatible at the transport layer, but the AI SDK provider intentionally
/// narrows prompt conversion: user messages are text-only, assistant reasoning is replayed only
/// for the assistant turns after the last user message, and tool-only assistant messages keep an
/// empty-string `content`.
pub fn convert_messages_deepseek_chat(
    messages: &[ChatMessage],
) -> Result<Vec<OpenAiMessage>, LlmError> {
    let last_user_message_index = messages
        .iter()
        .rposition(|message| message.role == MessageRole::User);
    let mut out = Vec::new();

    for (index, message) in messages.iter().enumerate() {
        match message.role {
            MessageRole::System | MessageRole::Developer => {
                out.push(OpenAiMessage {
                    role: match message.role {
                        MessageRole::Developer => "developer",
                        _ => "system",
                    }
                    .to_string(),
                    content: Some(serde_json::Value::String(deepseek_text_content(
                        &message.content,
                    ))),
                    tool_calls: None,
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::User => {
                out.push(OpenAiMessage {
                    role: "user".to_string(),
                    content: Some(serde_json::Value::String(deepseek_text_content(
                        &message.content,
                    ))),
                    tool_calls: None,
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::Assistant => {
                let mut text = String::new();
                let mut reasoning = String::new();

                match &message.content {
                    MessageContent::Text(value) => text.push_str(value),
                    MessageContent::MultiModal(parts) => {
                        for part in parts {
                            match part {
                                ContentPart::Text { text: value, .. } => text.push_str(value),
                                ContentPart::Reasoning { text: value, .. }
                                    if last_user_message_index
                                        .is_none_or(|last_user| index > last_user) =>
                                {
                                    reasoning.push_str(value);
                                }
                                _ => {}
                            }
                        }
                    }
                    #[cfg(feature = "structured-messages")]
                    MessageContent::Json(value) => {
                        text.push_str(&serde_json::to_string(value).unwrap_or_default());
                    }
                }

                let mut extra = HashMap::new();
                if !reasoning.is_empty() {
                    extra.insert(
                        "reasoning_content".to_string(),
                        serde_json::Value::String(reasoning),
                    );
                }

                out.push(OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(serde_json::Value::String(text)),
                    tool_calls: plain_assistant_tool_calls(message),
                    tool_call_id: None,
                    extra,
                });
            }
            MessageRole::Tool => {
                out.extend(convert_messages_with_target(
                    std::slice::from_ref(message),
                    MessageConversionTarget::OpenAiCompatible,
                )?);
            }
        }
    }

    Ok(out)
}

/// Convert messages for the xAI chat-completions shape used by the AI SDK provider.
///
/// xAI is OpenAI-compatible at the transport layer, but provider-owned file references are
/// represented as `{ type: "file", file: { file_id } }`, assistant reasoning is not replayed as
/// `reasoning_content`, and tool-only assistant messages keep an empty-string `content`.
pub fn convert_messages_xai_chat(messages: &[ChatMessage]) -> Result<Vec<OpenAiMessage>, LlmError> {
    let mut out = Vec::new();

    for message in messages {
        match message.role {
            MessageRole::System | MessageRole::Developer => {
                out.push(OpenAiMessage {
                    role: match message.role {
                        MessageRole::Developer => "developer",
                        _ => "system",
                    }
                    .to_string(),
                    content: Some(serde_json::Value::String(deepseek_text_content(
                        &message.content,
                    ))),
                    tool_calls: None,
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::User => {
                out.push(OpenAiMessage {
                    role: "user".to_string(),
                    content: Some(convert_message_content_with_target(
                        &message.content,
                        MessageConversionTarget::XaiChat,
                    )?),
                    tool_calls: None,
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::Assistant => {
                let mut text = String::new();
                match &message.content {
                    MessageContent::Text(value) => text.push_str(value),
                    MessageContent::MultiModal(parts) => {
                        for part in parts {
                            if let ContentPart::Text { text: value, .. } = part {
                                text.push_str(value);
                            }
                        }
                    }
                    #[cfg(feature = "structured-messages")]
                    MessageContent::Json(value) => {
                        text.push_str(&serde_json::to_string(value).unwrap_or_default());
                    }
                }

                out.push(OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(serde_json::Value::String(text)),
                    tool_calls: plain_assistant_tool_calls(message),
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::Tool => {
                emit_tool_result_messages(&mut out, message, false, false);
            }
        }
    }

    Ok(out)
}

/// Convert messages for the Mistral chat-completions shape used by the AI SDK provider.
///
/// Mistral uses string-valued `image_url` / `document_url` content parts, replays reasoning by
/// concatenating it into assistant text, and marks the trailing assistant turn with `prefix: true`.
pub fn convert_messages_mistral_chat(
    messages: &[ChatMessage],
) -> Result<Vec<OpenAiMessage>, LlmError> {
    let mut out = Vec::new();

    for (index, message) in messages.iter().enumerate() {
        match message.role {
            MessageRole::System | MessageRole::Developer => {
                out.push(OpenAiMessage {
                    role: match message.role {
                        MessageRole::Developer => "developer",
                        _ => "system",
                    }
                    .to_string(),
                    content: Some(serde_json::Value::String(deepseek_text_content(
                        &message.content,
                    ))),
                    tool_calls: None,
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::User => {
                out.push(OpenAiMessage {
                    role: "user".to_string(),
                    content: Some(mistral_user_content(&message.content)?),
                    tool_calls: None,
                    tool_call_id: None,
                    extra: HashMap::new(),
                });
            }
            MessageRole::Assistant => {
                let mut text = String::new();
                match &message.content {
                    MessageContent::Text(value) => text.push_str(value),
                    MessageContent::MultiModal(parts) => {
                        for part in parts {
                            match part {
                                ContentPart::Text { text: value, .. }
                                | ContentPart::Reasoning { text: value, .. } => {
                                    text.push_str(value);
                                }
                                _ => {}
                            }
                        }
                    }
                    #[cfg(feature = "structured-messages")]
                    MessageContent::Json(value) => {
                        text.push_str(&serde_json::to_string(value).unwrap_or_default());
                    }
                }

                let mut extra = HashMap::new();
                if index == messages.len().saturating_sub(1) {
                    extra.insert("prefix".to_string(), serde_json::Value::Bool(true));
                }

                out.push(OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(serde_json::Value::String(text)),
                    tool_calls: plain_assistant_tool_calls(message),
                    tool_call_id: None,
                    extra,
                });
            }
            MessageRole::Tool => {
                emit_tool_result_messages(&mut out, message, true, false);
            }
        }
    }

    Ok(out)
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
                    content: Some(
                        if target == MessageConversionTarget::OpenAiCompatible && text.is_empty() {
                            serde_json::Value::Null
                        } else {
                            serde_json::Value::String(text)
                        },
                    ),
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

                            let content_value = tool_result_content_value(output);

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

fn emit_tool_result_messages(
    out: &mut Vec<OpenAiMessage>,
    message: &ChatMessage,
    include_tool_name: bool,
    include_openai_compatible_metadata: bool,
) {
    match &message.content {
        MessageContent::MultiModal(parts) => {
            for part in parts {
                let ContentPart::ToolResult {
                    tool_call_id,
                    tool_name,
                    output,
                    provider_options,
                    ..
                } = part
                else {
                    continue;
                };

                let mut extra = HashMap::new();
                if include_tool_name {
                    extra.insert(
                        "name".to_string(),
                        serde_json::Value::String(tool_name.clone()),
                    );
                }
                if include_openai_compatible_metadata {
                    merge_openai_compatible_extra(&mut extra, Some(provider_options));
                }

                out.push(OpenAiMessage {
                    role: "tool".to_string(),
                    content: Some(serde_json::Value::String(tool_result_content_value(output))),
                    tool_calls: None,
                    tool_call_id: Some(tool_call_id.clone()),
                    extra,
                });
            }
        }
        MessageContent::Text(value) => {
            out.push(OpenAiMessage {
                role: "tool".to_string(),
                content: Some(serde_json::Value::String(value.clone())),
                tool_calls: None,
                tool_call_id: None,
                extra: HashMap::new(),
            });
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => {
            out.push(OpenAiMessage {
                role: "tool".to_string(),
                content: Some(serde_json::Value::String(
                    serde_json::to_string(value).unwrap_or_default(),
                )),
                tool_calls: None,
                tool_call_id: None,
                extra: HashMap::new(),
            });
        }
    }
}

fn mistral_user_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    let mut content_parts = Vec::new();

    match content {
        MessageContent::Text(value) => {
            content_parts.push(serde_json::json!({
                "type": "text",
                "text": value,
            }));
        }
        MessageContent::MultiModal(parts) => {
            for part in parts {
                match part {
                    ContentPart::Text { text, .. } => {
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": text,
                        }));
                    }
                    ContentPart::Image {
                        source, media_type, ..
                    } => {
                        let media_type = media_type.as_deref().unwrap_or("image/jpeg");
                        content_parts.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": media_source_as_data_or_url(source, media_type)?,
                        }));
                    }
                    ContentPart::File {
                        source, media_type, ..
                    } if media_type.starts_with("image/") => {
                        let media_type = if media_type == "image/*" {
                            "image/jpeg"
                        } else {
                            media_type.as_str()
                        };
                        content_parts.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": media_source_as_data_or_url(source, media_type)?,
                        }));
                    }
                    ContentPart::File {
                        source, media_type, ..
                    } if media_type == "application/pdf" => {
                        content_parts.push(serde_json::json!({
                            "type": "document_url",
                            "document_url": media_source_as_data_or_url(source, "application/pdf")?,
                        }));
                    }
                    ContentPart::File {
                        source: FilePartSource::ProviderReference { .. },
                        ..
                    } => {
                        return Err(LlmError::UnsupportedOperation(
                            "file parts with provider references".to_string(),
                        ));
                    }
                    ContentPart::File { .. } | ContentPart::Audio { .. } => {
                        return Err(LlmError::UnsupportedOperation(
                            "Only images and PDF file parts are supported".to_string(),
                        ));
                    }
                    _ => {}
                }
            }
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => {
            content_parts.push(serde_json::json!({
                "type": "text",
                "text": serde_json::to_string(value).unwrap_or_default(),
            }));
        }
    }

    Ok(serde_json::Value::Array(content_parts))
}

fn deepseek_text_content(content: &MessageContent) -> String {
    let mut text = String::new();
    match content {
        MessageContent::Text(value) => text.push_str(value),
        MessageContent::MultiModal(parts) => {
            for part in parts {
                if let ContentPart::Text { text: value, .. } = part {
                    text.push_str(value);
                }
            }
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => {
            text.push_str(&serde_json::to_string(value).unwrap_or_default());
        }
    }
    text
}

fn perplexity_text_content(content: &MessageContent) -> String {
    let mut text = String::new();
    match content {
        MessageContent::Text(value) => text.push_str(value),
        MessageContent::MultiModal(parts) => {
            for part in parts {
                if let ContentPart::Text { text: value, .. } = part {
                    text.push_str(value);
                }
            }
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => {
            text.push_str(&serde_json::to_string(value).unwrap_or_default());
        }
    }
    text
}

fn perplexity_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    let MessageContent::MultiModal(parts) = content else {
        return Ok(serde_json::Value::String(perplexity_text_content(content)));
    };

    let has_multipart_content = parts.iter().any(|part| match part {
        ContentPart::Image { .. } => true,
        ContentPart::File { media_type, .. } => {
            media_type.starts_with("image/") || media_type == "application/pdf"
        }
        _ => false,
    });

    if !has_multipart_content {
        return Ok(serde_json::Value::String(perplexity_text_content(content)));
    }

    let mut content_parts = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        match part {
            ContentPart::Text { text, .. } => {
                content_parts.push(serde_json::json!({
                    "type": "text",
                    "text": text,
                }));
            }
            ContentPart::Image {
                source, media_type, ..
            } => {
                content_parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": {
                        "url": perplexity_image_url(source, media_type.as_deref())?,
                    },
                }));
            }
            ContentPart::File {
                source, media_type, ..
            } if media_type.starts_with("image/") => {
                content_parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": {
                        "url": perplexity_image_url(source, Some(media_type))?,
                    },
                }));
            }
            ContentPart::File {
                source,
                media_type,
                filename,
                ..
            } if media_type == "application/pdf" => {
                let mut file_part = serde_json::Map::new();
                file_part.insert(
                    "type".to_string(),
                    serde_json::Value::String("file_url".to_string()),
                );
                file_part.insert(
                    "file_url".to_string(),
                    serde_json::json!({
                        "url": perplexity_pdf_url(source)?,
                    }),
                );

                if let Some(file_name) = filename
                    .clone()
                    .filter(|value| !value.trim().is_empty())
                    .or_else(|| {
                        (!matches!(source, FilePartSource::Media(MediaSource::Url { .. })))
                            .then(|| format!("document-{index}.pdf"))
                    })
                {
                    file_part.insert("file_name".to_string(), serde_json::json!(file_name));
                }

                content_parts.push(serde_json::Value::Object(file_part));
            }
            _ => {}
        }
    }

    Ok(serde_json::Value::Array(content_parts))
}

fn perplexity_image_url(
    source: &FilePartSource,
    media_type: Option<&str>,
) -> Result<String, LlmError> {
    let media_type = media_type.unwrap_or("image/jpeg");
    match source {
        FilePartSource::Media(MediaSource::Url { url }) => Ok(url.clone()),
        FilePartSource::Media(MediaSource::Base64 { data }) => {
            if data.starts_with("data:") {
                Ok(data.clone())
            } else {
                Ok(format!("data:{media_type};base64,{data}"))
            }
        }
        FilePartSource::Media(MediaSource::Binary { data }) => {
            let encoded = base64::engine::general_purpose::STANDARD.encode(data);
            Ok(format!("data:{media_type};base64,{encoded}"))
        }
        FilePartSource::ProviderReference { .. } => Err(LlmError::UnsupportedOperation(
            "file parts with provider references".to_string(),
        )),
    }
}

fn perplexity_pdf_url(source: &FilePartSource) -> Result<String, LlmError> {
    match source {
        FilePartSource::Media(MediaSource::Url { url }) => Ok(url.clone()),
        FilePartSource::Media(MediaSource::Base64 { data }) => Ok(data.clone()),
        FilePartSource::Media(MediaSource::Binary { data }) => {
            Ok(base64::engine::general_purpose::STANDARD.encode(data))
        }
        FilePartSource::ProviderReference { .. } => Err(LlmError::UnsupportedOperation(
            "file parts with provider references".to_string(),
        )),
    }
}

fn plain_assistant_tool_calls(message: &ChatMessage) -> Option<Vec<OpenAiToolCall>> {
    let tool_calls = message.tool_calls();
    if tool_calls.is_empty() {
        return None;
    }

    Some(
        tool_calls
            .iter()
            .filter_map(|part| {
                let ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    ..
                } = part
                else {
                    return None;
                };

                Some(OpenAiToolCall {
                    id: tool_call_id.clone(),
                    r#type: "function".to_string(),
                    function: Some(OpenAiFunction {
                        name: tool_name.clone(),
                        arguments: serde_json::to_string(arguments).unwrap_or_default(),
                    }),
                    extra: HashMap::new(),
                })
            })
            .collect(),
    )
}

fn tool_result_content_value(output: &ToolResultOutput) -> String {
    match output {
        ToolResultOutput::Text { value, .. } | ToolResultOutput::ErrorText { value, .. } => {
            value.clone()
        }
        ToolResultOutput::ExecutionDenied { reason, .. } => reason
            .clone()
            .unwrap_or_else(|| "Tool call execution denied.".to_string()),
        ToolResultOutput::Json { value, .. } | ToolResultOutput::ErrorJson { value, .. } => {
            serde_json::to_string(value).unwrap_or_default()
        }
        ToolResultOutput::Content { value, .. } => serde_json::to_string(value).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatMessage, FilePartSource, MessageMetadata, ProviderOptionsMap, ProviderReference,
        ToolResultContentPart,
    };
    use std::collections::HashMap;
    fn request_conversion_source() -> &'static str {
        let source = include_str!("message_dialect.rs");
        let (section, _) = source
            .split_once("\n#[cfg(test)]")
            .expect("test module marker should exist");
        section
    }

    #[test]
    fn openai_chat_request_conversion_source_does_not_read_legacy_provider_metadata_fields() {
        let source = request_conversion_source();

        assert!(
            !source.contains("providerMetadata"),
            "OpenAI chat request conversion must not read legacy response-side providerMetadata"
        );
        assert!(
            !source.contains("provider_metadata"),
            "OpenAI chat request conversion must not read legacy response-side provider_metadata"
        );
    }

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
    fn perplexity_text_only_parts_collapse_to_string() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![
                ContentPart::text("Hello "),
                ContentPart::text("World"),
            ]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let out = convert_messages_perplexity_chat(&[msg]).expect("convert messages");
        assert_eq!(
            out[0].content.as_ref(),
            Some(&serde_json::json!("Hello World"))
        );
    }

    #[test]
    fn perplexity_pdf_file_part_maps_to_file_url_content_part() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::base64("Zm9v"),
                media_type: "application/pdf".to_string(),
                filename: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let out = convert_messages_perplexity_chat(&[msg]).expect("convert messages");
        let content = out[0].content.clone().expect("content");
        let parts = content.as_array().expect("array");
        assert_eq!(parts[0]["type"], "file_url");
        assert_eq!(parts[0]["file_url"]["url"], "Zm9v");
        assert_eq!(parts[0]["file_name"], "document-0.pdf");
    }

    #[test]
    fn perplexity_file_provider_reference_is_rejected() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::provider_reference(ProviderReference::single(
                    "perplexity",
                    "file-abc",
                )),
                media_type: "application/pdf".to_string(),
                filename: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let err = convert_messages_perplexity_chat(&[msg]).expect_err("provider reference error");
        assert!(
            matches!(err, LlmError::UnsupportedOperation(message) if message == "file parts with provider references")
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
    fn openai_compatible_tool_only_assistant_content_is_null() {
        let msg = ChatMessage::assistant_with_content(vec![ContentPart::tool_call(
            "call_1",
            "lookup",
            serde_json::json!({ "query": "rust" }),
            None,
        )])
        .build();

        let out = convert_messages(&[msg]).expect("convert messages");

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, Some(serde_json::Value::Null));
        assert!(out[0].tool_calls.is_some());
    }

    #[test]
    fn openai_compatible_user_file_parts_match_ai_sdk_shapes() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![
                ContentPart::file_base64("AAECAw==", "application/pdf", None),
                ContentPart::file_base64("SGVsbG8=", "text/plain", None),
                ContentPart::file_base64("AAEC", "audio/mpeg", None),
            ]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let out = convert_messages(&[msg]).expect("convert messages");
        let parts = out[0]
            .content
            .as_ref()
            .and_then(|value| value.as_array())
            .expect("multipart user content");

        assert_eq!(parts[0]["type"], "file");
        assert_eq!(parts[0]["file"]["filename"], "document.pdf");
        assert_eq!(
            parts[0]["file"]["file_data"],
            "data:application/pdf;base64,AAECAw=="
        );
        assert_eq!(
            parts[1],
            serde_json::json!({ "type": "text", "text": "Hello" })
        );
        assert_eq!(parts[2]["type"], "input_audio");
        assert_eq!(parts[2]["input_audio"]["data"], "AAEC");
        assert_eq!(parts[2]["input_audio"]["format"], "mp3");
    }

    #[test]
    fn xai_provider_reference_file_parts_map_to_file_ids() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::provider_reference(ProviderReference::from([
                    ("openai", "file-openai"),
                    ("xai", "file-xai"),
                ])),
                media_type: "application/pdf".to_string(),
                filename: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let out = convert_messages_xai_chat(&[msg]).expect("convert messages");
        let parts = out[0]
            .content
            .as_ref()
            .and_then(|value| value.as_array())
            .expect("multipart xai user content");

        assert_eq!(
            parts[0],
            serde_json::json!({
                "type": "file",
                "file": { "file_id": "file-xai" }
            })
        );
    }

    #[test]
    fn xai_missing_provider_reference_reports_available_providers() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![ContentPart::File {
                source: FilePartSource::provider_reference(ProviderReference::single(
                    "openai",
                    "file-openai",
                )),
                media_type: "image/png".to_string(),
                filename: None,
                provider_options: ProviderOptionsMap::default(),
                provider_metadata: None,
            }]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let err = convert_messages_xai_chat(&[msg]).expect_err("missing xai reference");
        assert!(
            matches!(err, LlmError::InvalidParameter(message) if message == "No provider reference found for provider 'xai'. Available providers: openai")
        );
    }

    #[test]
    fn xai_assistant_reasoning_is_not_sent_as_reasoning_content() {
        let msg = ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("private reasoning"),
            ContentPart::tool_call(
                "call_1",
                "lookup",
                serde_json::json!({ "query": "rust" }),
                None,
            ),
        ])
        .build();

        let out = convert_messages_xai_chat(&[msg]).expect("convert messages");

        assert_eq!(out[0].content, Some(serde_json::json!("")));
        assert!(!out[0].extra.contains_key("reasoning_content"));
        assert!(out[0].tool_calls.is_some());
    }

    #[test]
    fn mistral_user_pdf_and_assistant_reasoning_match_ai_sdk_shape() {
        let user = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![
                ContentPart::text("Analyze this PDF"),
                ContentPart::file_url("https://example.com/report.pdf", "application/pdf"),
            ]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };
        let assistant = ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("thinking"),
            ContentPart::text("answer"),
        ])
        .build();

        let out = convert_messages_mistral_chat(&[user, assistant]).expect("convert messages");
        let user_parts = out[0]
            .content
            .as_ref()
            .and_then(|value| value.as_array())
            .expect("mistral user content array");

        assert_eq!(
            user_parts[1],
            serde_json::json!({
                "type": "document_url",
                "document_url": "https://example.com/report.pdf"
            })
        );
        assert_eq!(out[1].content, Some(serde_json::json!("thinkinganswer")));
        assert_eq!(out[1].extra.get("prefix"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn mistral_tool_result_messages_include_tool_name() {
        let assistant = ChatMessage::assistant_with_content(vec![ContentPart::tool_call(
            "call_1",
            "lookup",
            serde_json::json!({ "query": "rust" }),
            None,
        )])
        .build();
        let tool_result =
            ChatMessage::tool_result_json("call_1", "lookup", serde_json::json!({ "ok": true }))
                .build();

        let out =
            convert_messages_mistral_chat(&[assistant, tool_result]).expect("convert messages");

        assert_eq!(out[0].content, Some(serde_json::json!("")));
        assert!(!out[0].extra.contains_key("prefix"));
        assert_eq!(out[1].role, "tool");
        assert_eq!(out[1].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(out[1].extra.get("name"), Some(&serde_json::json!("lookup")));
        assert_eq!(out[1].content, Some(serde_json::json!("{\"ok\":true}")));
    }

    #[test]
    fn deepseek_user_message_conversion_keeps_only_text_parts() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::MultiModal(vec![
                ContentPart::text("Hello "),
                ContentPart::File {
                    source: FilePartSource::base64("AAECAw=="),
                    media_type: "image/png".to_string(),
                    filename: None,
                    provider_options: ProviderOptionsMap::default(),
                    provider_metadata: None,
                },
                ContentPart::text("World"),
            ]),
            metadata: MessageMetadata::default(),
            provider_options: ProviderOptionsMap::default(),
        };

        let out = convert_messages_deepseek_chat(&[msg]).expect("convert messages");

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "user");
        assert_eq!(
            out[0].content,
            Some(serde_json::Value::String("Hello World".to_string()))
        );
    }

    #[test]
    fn deepseek_assistant_reasoning_is_kept_only_after_last_user_message() {
        let assistant = ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("private old reasoning"),
            ContentPart::tool_call(
                "call_1",
                "lookup",
                serde_json::json!({ "query": "rust" }),
                None,
            ),
        ])
        .build();
        let tool_result =
            ChatMessage::tool_result_json("call_1", "lookup", serde_json::json!({ "ok": true }))
                .build();

        let before_last_user = convert_messages_deepseek_chat(&[
            ChatMessage::user("first").build(),
            assistant.clone(),
            tool_result,
            ChatMessage::user("second").build(),
        ])
        .expect("convert messages");

        assert_eq!(
            before_last_user[1].content,
            Some(serde_json::Value::String(String::new()))
        );
        assert!(
            !before_last_user[1].extra.contains_key("reasoning_content"),
            "reasoning before the last user turn must not be replayed"
        );

        let after_last_user =
            convert_messages_deepseek_chat(&[ChatMessage::user("first").build(), assistant])
                .expect("convert messages");

        assert_eq!(
            after_last_user[1].extra.get("reasoning_content"),
            Some(&serde_json::json!("private old reasoning"))
        );
        assert!(after_last_user[1].tool_calls.is_some());
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
            convert_messages(std::slice::from_ref(&message))
                .expect("convert openai-compatible messages"),
            convert_messages_openai_chat(std::slice::from_ref(&message))
                .expect("convert openai chat messages"),
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
    fn openai_compatible_pdf_file_part_maps_to_file_content_part() {
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

        let out = convert_messages(&[msg]).expect("convert messages");
        let content = out[0].content.clone().expect("content");
        let parts = content.as_array().expect("array");
        assert_eq!(parts[0]["type"], "file");
        assert_eq!(parts[0]["file"]["filename"], "document.pdf");
        assert_eq!(
            parts[0]["file"]["file_data"],
            "data:application/pdf;base64,Zm9v"
        );
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
