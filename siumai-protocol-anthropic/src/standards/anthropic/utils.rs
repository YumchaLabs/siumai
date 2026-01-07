//! Anthropic Utility Functions
//!
//! Common utility functions for Anthropic Claude API interactions.

use super::types::*;
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::types::*;
use base64::Engine;
use reqwest::header::HeaderMap;

/// Build HTTP headers for Anthropic API requests according to official documentation
/// <https://docs.anthropic.com/en/api/messages>
pub fn build_headers(
    api_key: &str,
    custom_headers: &std::collections::HashMap<String, String>,
) -> Result<HeaderMap, LlmError> {
    let mut builder = HttpHeaderBuilder::new()
        .with_custom_auth("x-api-key", api_key)?
        .with_json_content_type()
        .with_header("anthropic-version", "2023-06-01")?;

    if let Some(beta_features) = custom_headers.get("anthropic-beta") {
        builder = builder.with_header("anthropic-beta", beta_features)?;
    }

    let filtered_headers: std::collections::HashMap<String, String> = custom_headers
        .iter()
        .filter(|(k, _)| k.as_str() != "anthropic-beta")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    builder = builder.with_custom_headers(&filtered_headers)?;
    Ok(builder.build())
}

#[cfg(test)]
mod header_tests {
    use super::*;

    #[test]
    fn build_headers_includes_required_anthropic_headers() {
        let headers = build_headers("k", &std::collections::HashMap::new()).unwrap();
        assert_eq!(
            headers.get("x-api-key").and_then(|v| v.to_str().ok()),
            Some("k")
        );
        assert!(headers.contains_key("anthropic-version"));
        assert_eq!(
            headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }

    #[test]
    fn build_headers_preserves_anthropic_beta_header() {
        let mut custom = std::collections::HashMap::new();
        custom.insert(
            "anthropic-beta".to_string(),
            "feature-a,feature-b".to_string(),
        );
        let headers = build_headers("k", &custom).unwrap();
        assert_eq!(
            headers.get("anthropic-beta").and_then(|v| v.to_str().ok()),
            Some("feature-a,feature-b")
        );
    }
}

/// Convert message content to Anthropic format
pub fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(serde_json::Value::Array(vec![serde_json::json!({
            "type": "text",
            "text": text
        })])),
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
                    ContentPart::Image { source, .. } => {
                        // Anthropic requires base64-encoded images
                        let (media_type, data) = match source {
                            crate::types::chat::MediaSource::Base64 { data } => {
                                ("image/jpeg", data.clone())
                            }
                            crate::types::chat::MediaSource::Binary { data } => {
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                ("image/jpeg", encoded)
                            }
                            crate::types::chat::MediaSource::Url { url } => {
                                // Anthropic doesn't support URLs, convert to text
                                content_parts.push(serde_json::json!({
                                    "type": "text",
                                    "text": format!("[Image: {}]", url)
                                }));
                                continue;
                            }
                        };

                        content_parts.push(serde_json::json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data
                            }
                        }));
                    }
                    ContentPart::Audio { source, .. } => {
                        // Anthropic does not support audio input
                        let placeholder = match source {
                            crate::types::chat::MediaSource::Url { url } => {
                                format!("[Audio: {}]", url)
                            }
                            _ => "[Audio input not supported by Anthropic]".to_string(),
                        };
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": placeholder
                        }));
                    }
                    ContentPart::Source {
                        source_type: _,
                        url,
                        title,
                        ..
                    } => {
                        // Anthropic does not support `source` parts in request input.
                        // Convert them into a best-effort text placeholder to preserve context.
                        let text = if !title.is_empty() && title != url {
                            format!("[Source: {title} ({url})]")
                        } else {
                            format!("[Source: {url}]")
                        };
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                    ContentPart::File {
                        source,
                        media_type,
                        filename,
                        ..
                    } => {
                        // Anthropic supports document parts for PDFs and plain text.
                        if media_type == "application/pdf" {
                            let source_json = match source {
                                crate::types::chat::MediaSource::Base64 { data } => {
                                    serde_json::json!({
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": data
                                    })
                                }
                                crate::types::chat::MediaSource::Binary { data } => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    serde_json::json!({
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": encoded
                                    })
                                }
                                crate::types::chat::MediaSource::Url { url } => {
                                    serde_json::json!({
                                        "type": "url",
                                        "url": url
                                    })
                                }
                            };

                            let mut doc = serde_json::json!({
                                "type": "document",
                                "source": source_json,
                            });
                            if let Some(name) = filename.as_ref() {
                                doc["title"] = serde_json::json!(name);
                            }
                            content_parts.push(doc);
                        } else if media_type == "text/plain" {
                            let source_json = match source {
                                crate::types::chat::MediaSource::Url { url } => {
                                    serde_json::json!({
                                        "type": "url",
                                        "url": url
                                    })
                                }
                                crate::types::chat::MediaSource::Binary { data } => {
                                    let text =
                                        String::from_utf8(data.clone()).unwrap_or_else(|_| {
                                            "[Invalid UTF-8 text/plain]".to_string()
                                        });
                                    serde_json::json!({
                                        "type": "text",
                                        "media_type": "text/plain",
                                        "data": text
                                    })
                                }
                                crate::types::chat::MediaSource::Base64 { data } => {
                                    let decoded = base64::engine::general_purpose::STANDARD
                                        .decode(data.as_bytes())
                                        .ok()
                                        .and_then(|bytes| String::from_utf8(bytes).ok())
                                        .unwrap_or_else(|| {
                                            "[Invalid base64 text/plain]".to_string()
                                        });
                                    serde_json::json!({
                                        "type": "text",
                                        "media_type": "text/plain",
                                        "data": decoded
                                    })
                                }
                            };

                            let mut doc = serde_json::json!({
                                "type": "document",
                                "source": source_json,
                            });
                            if let Some(name) = filename.as_ref() {
                                doc["title"] = serde_json::json!(name);
                            }
                            content_parts.push(doc);
                        } else {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("[Unsupported file type: {}]", media_type)
                            }));
                        }
                    }
                    ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        ..
                    } => {
                        // Tool calls in Anthropic format
                        content_parts.push(serde_json::json!({
                            "type": "tool_use",
                            "id": tool_call_id,
                            "name": tool_name,
                            "input": arguments
                        }));
                    }
                    ContentPart::ToolResult {
                        tool_call_id,
                        output,
                        ..
                    } => {
                        // Tool results in Anthropic format
                        use crate::types::ToolResultOutput;

                        let (content_value, is_error) = match output {
                            ToolResultOutput::Text { value } => (serde_json::json!(value), false),
                            ToolResultOutput::Json { value } => (value.clone(), false),
                            ToolResultOutput::ErrorText { value } => {
                                (serde_json::json!(value), true)
                            }
                            ToolResultOutput::ErrorJson { value } => (value.clone(), true),
                            ToolResultOutput::ExecutionDenied { reason } => {
                                let msg = reason
                                    .as_ref()
                                    .map(|r| format!("Execution denied: {}", r))
                                    .unwrap_or_else(|| "Execution denied".to_string());
                                (serde_json::json!(msg), true)
                            }
                            ToolResultOutput::Content { value } => {
                                // Convert multimodal content to Anthropic format
                                let content_array: Vec<serde_json::Value> = value.iter().map(|part| {
                                    use crate::types::ToolResultContentPart;
                                    match part {
                                        ToolResultContentPart::Text { text } => {
                                            serde_json::json!({"type": "text", "text": text})
                                        }
                                        ToolResultContentPart::Image { source, .. } => {
                                            use crate::types::MediaSource;
                                            match source {
                                                MediaSource::Url { url } => {
                                                    serde_json::json!({"type": "image", "source": {"type": "url", "url": url}})
                                                }
                                                MediaSource::Base64 { data } => {
                                                    serde_json::json!({"type": "image", "source": {"type": "base64", "data": data}})
                                                }
                                                MediaSource::Binary { .. } => {
                                                    serde_json::json!({"type": "text", "text": "[Binary image data]"})
                                                }
                                            }
                                        }
                                        ToolResultContentPart::File { .. } => {
                                            serde_json::json!({"type": "text", "text": "[File attachment]"})
                                        }
                                    }
                                }).collect();
                                (serde_json::Value::Array(content_array), false)
                            }
                        };

                        content_parts.push(serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content_value,
                            "is_error": is_error
                        }));
                    }
                    ContentPart::ToolApprovalResponse { .. } => {}
                    ContentPart::Reasoning { text, .. } => {
                        // Emit as a thinking block (Anthropic format). If the caller does not provide
                        // a valid signature (required for replaying thinking blocks), we will
                        // degrade it to a plain text `<thinking>...</thinking>` wrapper later in
                        // `convert_messages`.
                        content_parts.push(serde_json::json!({
                            "type": "thinking",
                            "thinking": text
                        }));
                    }
                }
            }

            Ok(serde_json::Value::Array(content_parts))
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(v) => {
            let s = serde_json::to_string(v).unwrap_or_default();
            Ok(serde_json::Value::Array(vec![serde_json::json!({
                "type": "text",
                "text": s
            })]))
        }
    }
}

/// Convert messages to Anthropic format
pub fn convert_messages(
    messages: &[ChatMessage],
) -> Result<(Vec<AnthropicMessage>, Option<serde_json::Value>), LlmError> {
    let mut anthropic_messages = Vec::new();
    let mut system_blocks: Vec<serde_json::Value> = Vec::new();
    let mut system_phase = true;
    let mut cache_breakpoints: usize = 0;
    const MAX_CACHE_BREAKPOINTS: usize = 4;

    // Helper: map generic message metadata cache control to Anthropic JSON
    fn map_cache_control(meta: &MessageMetadata) -> Option<serde_json::Value> {
        if let Some(ref cc) = meta.cache_control {
            match cc {
                CacheControl::Ephemeral => Some(serde_json::json!({
                    "type": "ephemeral"
                })),
                CacheControl::Persistent { ttl } => {
                    // Anthropic currently documents ephemeral cache; when persistent is used
                    // map to ephemeral and forward TTL seconds if provided for best effort.
                    let mut obj = serde_json::json!({ "type": "ephemeral" });
                    if let Some(dur) = ttl {
                        obj["ttl"] = serde_json::json!(dur.as_secs());
                    }
                    Some(obj)
                }
            }
        } else {
            None
        }
    }

    fn maybe_attach_cache_control(
        target: &mut serde_json::Value,
        cache_control: &serde_json::Value,
        cache_breakpoints: &mut usize,
    ) {
        if cache_control.is_null() {
            return;
        }
        if *cache_breakpoints >= MAX_CACHE_BREAKPOINTS {
            return;
        }
        let Some(obj) = target.as_object_mut() else {
            return;
        };
        if obj
            .get("type")
            .and_then(|v| v.as_str())
            .is_some_and(|t| t == "thinking" || t == "redacted_thinking")
        {
            // Vercel-aligned: thinking blocks cannot have cache_control directly.
            return;
        }
        if obj.contains_key("cache_control") {
            return;
        }
        obj.insert("cache_control".to_string(), cache_control.clone());
        *cache_breakpoints += 1;
    }

    fn normalize_reasoning_blocks(
        role: &MessageRole,
        message_custom: &std::collections::HashMap<String, serde_json::Value>,
        content_json: &mut serde_json::Value,
    ) {
        let Some(parts) = content_json.as_array_mut() else {
            return;
        };

        let signature_global = message_custom
            .get("anthropic_thinking_signature")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let signatures_by_index = message_custom
            .get("anthropic_thinking_signatures")
            .and_then(|v| v.as_object());
        let redacted_data = message_custom
            .get("anthropic_redacted_thinking_data")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        for (idx, part) in parts.iter_mut().enumerate() {
            let Some(kind) = part.get("type").and_then(|v| v.as_str()) else {
                continue;
            };

            match kind {
                "thinking" => {
                    // Remove any accidental cache_control on thinking blocks.
                    if let Some(obj) = part.as_object_mut() {
                        obj.remove("cache_control");
                    }

                    // Only assistant messages can safely replay thinking blocks, and they require a signature.
                    if !matches!(role, MessageRole::Assistant) {
                        let thinking = part
                            .get("thinking")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        *part = serde_json::json!({
                            "type": "text",
                            "text": format!("<thinking>{thinking}</thinking>")
                        });
                        continue;
                    }

                    // If caller explicitly requested redacted thinking, convert the block.
                    if signature_global.is_none()
                        && signatures_by_index.is_none()
                        && let Some(data) = &redacted_data
                    {
                        *part = serde_json::json!({
                            "type": "redacted_thinking",
                            "data": data
                        });
                        continue;
                    }

                    // Attach signature if available; otherwise degrade to text wrapper.
                    let sig = signatures_by_index
                        .and_then(|m| m.get(&idx.to_string()))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| signature_global.clone());

                    if let Some(sig) = sig {
                        part["signature"] = serde_json::json!(sig);
                    } else {
                        let thinking = part
                            .get("thinking")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        *part = serde_json::json!({
                            "type": "text",
                            "text": format!("<thinking>{thinking}</thinking>")
                        });
                    }
                }
                "redacted_thinking" => {
                    // Remove any accidental cache_control on redacted thinking blocks.
                    if let Some(obj) = part.as_object_mut() {
                        obj.remove("cache_control");
                    }

                    if !matches!(role, MessageRole::Assistant) {
                        *part = serde_json::json!({
                            "type": "text",
                            "text": "<thinking>[REDACTED]</thinking>"
                        });
                    }
                }
                _ => {}
            }
        }
    }

    fn apply_document_citations(
        message_custom: &std::collections::HashMap<String, serde_json::Value>,
        content_json: &mut serde_json::Value,
    ) {
        let Some(parts) = content_json.as_array_mut() else {
            return;
        };

        let Some(map) = message_custom
            .get("anthropic_document_citations")
            .and_then(|v| v.as_object())
        else {
            return;
        };

        for (idx_str, cfg) in map {
            let Ok(idx) = idx_str.parse::<usize>() else {
                continue;
            };
            let Some(part) = parts.get_mut(idx) else {
                continue;
            };
            let Some(obj) = part.as_object_mut() else {
                continue;
            };

            let enabled = if let Some(b) = cfg.as_bool() {
                b
            } else {
                cfg.get("enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            };
            if !enabled {
                continue;
            }

            if obj
                .get("type")
                .and_then(|v| v.as_str())
                .is_some_and(|t| t == "document")
            {
                obj.entry("citations".to_string())
                    .or_insert_with(|| serde_json::json!({ "enabled": true }));
            }
        }
    }

    fn apply_document_metadata(
        message_custom: &std::collections::HashMap<String, serde_json::Value>,
        content_json: &mut serde_json::Value,
    ) {
        let Some(parts) = content_json.as_array_mut() else {
            return;
        };

        let Some(map) = message_custom
            .get("anthropic_document_metadata")
            .and_then(|v| v.as_object())
        else {
            return;
        };

        for (idx_str, meta) in map {
            let Ok(idx) = idx_str.parse::<usize>() else {
                continue;
            };
            let Some(part) = parts.get_mut(idx) else {
                continue;
            };
            let Some(obj) = part.as_object_mut() else {
                continue;
            };
            if obj.get("type").and_then(|v| v.as_str()) != Some("document") {
                continue;
            }

            let Some(meta_obj) = meta.as_object() else {
                continue;
            };

            if let Some(title) = meta_obj.get("title").and_then(|v| v.as_str())
                && !title.is_empty()
            {
                obj.insert("title".to_string(), serde_json::json!(title));
            }
            if let Some(context) = meta_obj.get("context").and_then(|v| v.as_str())
                && !context.is_empty()
            {
                obj.insert("context".to_string(), serde_json::json!(context));
            }
        }
    }

    fn flatten_system_text(content: &MessageContent) -> String {
        match content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::MultiModal(parts) => parts
                .iter()
                .filter_map(|part| {
                    if let crate::types::ContentPart::Text { text } = part {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" "),
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => serde_json::to_string(v).unwrap_or_default(),
        }
    }

    for message in messages {
        match message.role {
            MessageRole::System => {
                // Vercel-aligned: system messages must be contiguous at the start.
                if !system_phase {
                    return Err(LlmError::InvalidParameter(
                        "System messages must appear at the beginning of the conversation for Anthropic"
                            .to_string(),
                    ));
                }

                let text = flatten_system_text(&message.content);
                if !text.trim().is_empty() {
                    let mut block = serde_json::json!({
                        "type": "text",
                        "text": text,
                    });
                    if let Some(cc) = map_cache_control(&message.metadata) {
                        maybe_attach_cache_control(&mut block, &cc, &mut cache_breakpoints);
                    }
                    system_blocks.push(block);
                }
            }
            MessageRole::User => {
                system_phase = false;
                let mut content_json = convert_message_content(&message.content)?;

                // Ensure content is an array so we can attach `cache_control`.
                if let serde_json::Value::String(s) = &content_json {
                    content_json = serde_json::Value::Array(vec![serde_json::json!({
                        "type": "text",
                        "text": s
                    })]);
                }

                normalize_reasoning_blocks(
                    &message.role,
                    &message.metadata.custom,
                    &mut content_json,
                );
                apply_document_citations(&message.metadata.custom, &mut content_json);
                apply_document_metadata(&message.metadata.custom, &mut content_json);

                // Part-level cache control map (preferred).
                let mut part_cache_controls: std::collections::HashMap<usize, serde_json::Value> =
                    std::collections::HashMap::new();
                if let Some(obj) = message
                    .metadata
                    .custom
                    .get("anthropic_content_cache_controls")
                    .and_then(|v| v.as_object())
                {
                    for (k, v) in obj {
                        if let Ok(idx) = k.parse::<usize>() {
                            part_cache_controls.insert(idx, v.clone());
                        }
                    }
                }

                // Content-level cache control indices from metadata.custom
                if let Some(indices) = message
                    .metadata
                    .custom
                    .get("anthropic_content_cache_indices")
                    .and_then(|v| v.as_array())
                    && let Some(parts) = content_json.as_array_mut()
                {
                    for idx_val in indices {
                        if let Some(i) = idx_val.as_u64().map(|u| u as usize)
                            && let Some(part) = parts.get_mut(i)
                            && part.is_object()
                        {
                            part_cache_controls
                                .entry(i)
                                .or_insert_with(|| serde_json::json!({"type":"ephemeral"}));
                        }
                    }
                }

                // Apply per-part cache controls in index order (Vercel-aligned behavior).
                if !part_cache_controls.is_empty()
                    && let Some(parts) = content_json.as_array_mut()
                {
                    let mut idxs: Vec<usize> = part_cache_controls.keys().copied().collect();
                    idxs.sort_unstable();
                    for i in idxs {
                        let Some(part) = parts.get_mut(i) else {
                            continue;
                        };
                        let Some(cc) = part_cache_controls.get(&i) else {
                            continue;
                        };
                        maybe_attach_cache_control(part, cc, &mut cache_breakpoints);
                    }
                }

                // Message-level cache control applies to the last content part (if any), unless already set.
                if let Some(cc) = map_cache_control(&message.metadata)
                    && let Some(parts) = content_json.as_array_mut()
                    && let Some(last) = parts.last_mut()
                {
                    maybe_attach_cache_control(last, &cc, &mut cache_breakpoints);
                }

                anthropic_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: content_json,
                    cache_control: None,
                });
            }
            MessageRole::Assistant => {
                system_phase = false;
                let mut content_json = convert_message_content(&message.content)?;

                // Ensure content is an array so we can attach `cache_control`.
                if let serde_json::Value::String(s) = &content_json {
                    content_json = serde_json::Value::Array(vec![serde_json::json!({
                        "type": "text",
                        "text": s
                    })]);
                }

                normalize_reasoning_blocks(
                    &message.role,
                    &message.metadata.custom,
                    &mut content_json,
                );
                apply_document_citations(&message.metadata.custom, &mut content_json);
                apply_document_metadata(&message.metadata.custom, &mut content_json);

                // Part-level cache control map (preferred).
                let mut part_cache_controls: std::collections::HashMap<usize, serde_json::Value> =
                    std::collections::HashMap::new();
                if let Some(obj) = message
                    .metadata
                    .custom
                    .get("anthropic_content_cache_controls")
                    .and_then(|v| v.as_object())
                {
                    for (k, v) in obj {
                        if let Ok(idx) = k.parse::<usize>() {
                            part_cache_controls.insert(idx, v.clone());
                        }
                    }
                }
                if let Some(indices) = message
                    .metadata
                    .custom
                    .get("anthropic_content_cache_indices")
                    .and_then(|v| v.as_array())
                {
                    for idx_val in indices {
                        if let Some(i) = idx_val.as_u64().map(|u| u as usize) {
                            part_cache_controls
                                .entry(i)
                                .or_insert_with(|| serde_json::json!({"type":"ephemeral"}));
                        }
                    }
                }

                if !part_cache_controls.is_empty()
                    && let Some(parts) = content_json.as_array_mut()
                {
                    let mut idxs: Vec<usize> = part_cache_controls.keys().copied().collect();
                    idxs.sort_unstable();
                    for i in idxs {
                        let Some(part) = parts.get_mut(i) else {
                            continue;
                        };
                        let Some(cc) = part_cache_controls.get(&i) else {
                            continue;
                        };
                        maybe_attach_cache_control(part, cc, &mut cache_breakpoints);
                    }
                }

                // Message-level cache control applies to the last content part (if any), unless already set.
                if let Some(cc) = map_cache_control(&message.metadata)
                    && let Some(parts) = content_json.as_array_mut()
                    && let Some(last) = parts.last_mut()
                {
                    maybe_attach_cache_control(last, &cc, &mut cache_breakpoints);
                }

                anthropic_messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: content_json,
                    cache_control: None,
                });
            }
            MessageRole::Developer => {
                // Treat developer as system-level instructions for Anthropic.
                // Vercel-aligned: must be contiguous at the start.
                if !system_phase {
                    return Err(LlmError::InvalidParameter(
                        "Developer messages must appear at the beginning of the conversation for Anthropic"
                            .to_string(),
                    ));
                }

                let text = flatten_system_text(&message.content);
                if !text.trim().is_empty() {
                    let mut block = serde_json::json!({
                        "type": "text",
                        "text": format!("Developer instructions: {text}"),
                    });
                    if let Some(cc) = map_cache_control(&message.metadata) {
                        maybe_attach_cache_control(&mut block, &cc, &mut cache_breakpoints);
                    }
                    system_blocks.push(block);
                }
            }
            MessageRole::Tool => {
                system_phase = false;
                // Handling of tool results for Anthropic
                let mut content_json = convert_message_content(&message.content)?;

                // Ensure content is an array so we can attach `cache_control`.
                if let serde_json::Value::String(s) = &content_json {
                    content_json = serde_json::Value::Array(vec![serde_json::json!({
                        "type": "text",
                        "text": s
                    })]);
                }

                normalize_reasoning_blocks(
                    &message.role,
                    &message.metadata.custom,
                    &mut content_json,
                );

                // Apply per-part cache controls (same escape hatch as user/assistant).
                let mut part_cache_controls: std::collections::HashMap<usize, serde_json::Value> =
                    std::collections::HashMap::new();
                if let Some(obj) = message
                    .metadata
                    .custom
                    .get("anthropic_content_cache_controls")
                    .and_then(|v| v.as_object())
                {
                    for (k, v) in obj {
                        if let Ok(idx) = k.parse::<usize>() {
                            part_cache_controls.insert(idx, v.clone());
                        }
                    }
                }
                if let Some(indices) = message
                    .metadata
                    .custom
                    .get("anthropic_content_cache_indices")
                    .and_then(|v| v.as_array())
                {
                    for idx_val in indices {
                        if let Some(i) = idx_val.as_u64().map(|u| u as usize) {
                            part_cache_controls
                                .entry(i)
                                .or_insert_with(|| serde_json::json!({"type":"ephemeral"}));
                        }
                    }
                }
                if !part_cache_controls.is_empty()
                    && let Some(parts) = content_json.as_array_mut()
                {
                    let mut idxs: Vec<usize> = part_cache_controls.keys().copied().collect();
                    idxs.sort_unstable();
                    for i in idxs {
                        let Some(part) = parts.get_mut(i) else {
                            continue;
                        };
                        let Some(cc) = part_cache_controls.get(&i) else {
                            continue;
                        };
                        maybe_attach_cache_control(part, cc, &mut cache_breakpoints);
                    }
                }

                // Message-level cache control applies to the last content part (if any), unless already set.
                if let Some(cc) = map_cache_control(&message.metadata)
                    && let Some(parts) = content_json.as_array_mut()
                    && let Some(last) = parts.last_mut()
                {
                    maybe_attach_cache_control(last, &cc, &mut cache_breakpoints);
                }

                anthropic_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: content_json,
                    cache_control: None,
                });
            }
        }
    }

    let system = if system_blocks.is_empty() {
        None
    } else if system_blocks.len() == 1
        && system_blocks[0].get("cache_control").is_none()
        && system_blocks[0].get("type").and_then(|v| v.as_str()) == Some("text")
    {
        // Backward-compatible: allow a single system string.
        system_blocks[0]
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| serde_json::Value::String(s.to_string()))
    } else {
        Some(serde_json::Value::Array(system_blocks))
    };

    Ok((anthropic_messages, system))
}

/// Parse Anthropic finish reason according to official API documentation
/// <https://docs.anthropic.com/en/api/handling-stop-reasons>
pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("end_turn") => Some(FinishReason::Stop),
        Some("max_tokens") => Some(FinishReason::Length),
        Some("tool_use") => Some(FinishReason::ToolCalls),
        Some("stop_sequence") => Some(FinishReason::Other("stop_sequence".to_string())),
        Some("pause_turn") => Some(FinishReason::Other("pause_turn".to_string())),
        Some("refusal") => Some(FinishReason::ContentFilter),
        Some(other) => Some(FinishReason::Other(other.to_string())),
        None => None,
    }
}

/// Get default models for Anthropic according to latest available models
pub fn get_default_models() -> Vec<String> {
    vec![
        "claude-opus-4-1".to_string(),
        "claude-sonnet-4-0".to_string(),
        "claude-3-7-sonnet-latest".to_string(),
        "claude-3-5-sonnet-20241022".to_string(),
        "claude-3-5-sonnet-20240620".to_string(),
        "claude-3-5-haiku-20241022".to_string(),
        "claude-3-opus-20240229".to_string(),
        "claude-3-sonnet-20240229".to_string(),
        "claude-3-haiku-20240307".to_string(),
    ]
}

/// Parse Anthropic response content with support for thinking blocks
pub fn parse_response_content(content_blocks: &[AnthropicContentBlock]) -> MessageContent {
    // Find the first text block (skip thinking blocks for main content)
    for content_block in content_blocks {
        if content_block.r#type.as_str() == "text" {
            return MessageContent::Text(content_block.text.clone().unwrap_or_default());
        }
    }
    MessageContent::Text(String::new())
}

#[cfg(test)]
mod system_message_tests {
    use super::*;

    #[test]
    fn allows_multiple_system_messages_at_start_as_system_blocks() {
        let messages = vec![
            ChatMessage::system("sys1").build(),
            ChatMessage::system("sys2").build(),
            ChatMessage::user("hi").build(),
        ];

        let (_msgs, system) = convert_messages(&messages).unwrap();
        let system = system.expect("system");

        let arr = system.as_array().expect("expected system blocks array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "sys1");
        assert_eq!(arr[1]["text"], "sys2");
    }

    #[test]
    fn rejects_system_messages_after_non_system_message() {
        let messages = vec![
            ChatMessage::user("hi").build(),
            ChatMessage::system("sys").build(),
        ];

        let err = convert_messages(&messages).unwrap_err();
        assert!(matches!(err, LlmError::InvalidParameter(_)));
    }

    #[test]
    fn system_cache_control_forces_system_blocks_representation() {
        let mut sys = ChatMessage::system("sys").build();
        sys.metadata.cache_control = Some(crate::types::CacheControl::Ephemeral);

        let messages = vec![sys, ChatMessage::user("hi").build()];
        let (_msgs, system) = convert_messages(&messages).unwrap();
        let system = system.expect("system");

        let arr = system.as_array().expect("expected system blocks array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn message_cache_control_applies_to_last_user_content_part() {
        let msg = ChatMessage::user("part1")
            .with_content_parts(vec![crate::types::ContentPart::text("part2")])
            .cache_control(crate::types::CacheControl::Ephemeral)
            .build();

        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let content = msgs[0].content.as_array().expect("content array");
        assert!(content[0].get("cache_control").is_none());
        assert_eq!(content[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn part_cache_control_map_applies_to_selected_part() {
        let msg = ChatMessage::user("part1")
            .with_content_parts(vec![crate::types::ContentPart::text("part2")])
            .cache_control_for_part(0, crate::types::CacheControl::Ephemeral)
            .build();

        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let content = msgs[0].content.as_array().expect("content array");
        assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
        assert!(content[1].get("cache_control").is_none());
    }

    #[test]
    fn enforces_cache_breakpoint_limit_of_four_by_dropping_extras() {
        let mut sys1 = ChatMessage::system("sys1").build();
        sys1.metadata.cache_control = Some(crate::types::CacheControl::Ephemeral);
        let mut sys2 = ChatMessage::system("sys2").build();
        sys2.metadata.cache_control = Some(crate::types::CacheControl::Ephemeral);
        let user1 = ChatMessage::user("u1")
            .cache_control(crate::types::CacheControl::Ephemeral)
            .build();
        let user2 = ChatMessage::user("u2")
            .cache_control(crate::types::CacheControl::Ephemeral)
            .build();
        let user3 = ChatMessage::user("u3")
            .cache_control(crate::types::CacheControl::Ephemeral)
            .build();

        let (msgs, system) = convert_messages(&[sys1, sys2, user1, user2, user3]).unwrap();
        let system = system.expect("system");
        let sys_arr = system.as_array().expect("system blocks");

        // 1) Two system blocks consume 2 breakpoints.
        assert!(sys_arr[0].get("cache_control").is_some());
        assert!(sys_arr[1].get("cache_control").is_some());

        // 2) user1 and user2 consume 2 more breakpoints (message-level applied to last part).
        assert!(msgs[0].content[0].get("cache_control").is_some());
        assert!(msgs[1].content[0].get("cache_control").is_some());

        // 3) user3 exceeds the max-4 limit and should be dropped.
        assert!(msgs[2].content[0].get("cache_control").is_none());
    }

    #[test]
    fn assistant_reasoning_without_signature_degrades_to_text_wrapper() {
        let msg = ChatMessage::assistant_with_content(vec![ContentPart::reasoning("hi")]).build();
        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let parts = msgs[0].content.as_array().expect("content array");

        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "<thinking>hi</thinking>");
    }

    #[test]
    fn assistant_reasoning_with_signature_emits_thinking_block() {
        let mut msg =
            ChatMessage::assistant_with_content(vec![ContentPart::reasoning("hi")]).build();
        msg.metadata.custom.insert(
            "anthropic_thinking_signature".to_string(),
            serde_json::json!("sig"),
        );

        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let parts = msgs[0].content.as_array().expect("content array");

        assert_eq!(parts[0]["type"], "thinking");
        assert_eq!(parts[0]["thinking"], "hi");
        assert_eq!(parts[0]["signature"], "sig");
        assert!(parts[0].get("cache_control").is_none());
    }

    #[test]
    fn cache_control_is_not_attached_to_thinking_blocks() {
        let mut msg = ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("hi"),
            ContentPart::text("ok"),
        ])
        .cache_control(crate::types::CacheControl::Ephemeral)
        .cache_control_for_part(0, crate::types::CacheControl::Ephemeral)
        .build();
        msg.metadata.custom.insert(
            "anthropic_thinking_signature".to_string(),
            serde_json::json!("sig"),
        );

        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let parts = msgs[0].content.as_array().expect("content array");

        assert_eq!(parts[0]["type"], "thinking");
        assert!(parts[0].get("cache_control").is_none());

        assert_eq!(parts[1]["type"], "text");
        assert_eq!(parts[1]["cache_control"]["type"], "ephemeral");
    }
}

/// Parse Anthropic response content and extract tool calls
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
                    fn wrap_code_execution_input(
                        name: &str,
                        input: &serde_json::Value,
                    ) -> serde_json::Value {
                        let mut obj = serde_json::Map::new();
                        let kind = if name == "code_execution" {
                            "programmatic-tool-call"
                        } else {
                            name
                        };
                        obj.insert("type".to_string(), serde_json::json!(kind));

                        if let serde_json::Value::Object(m) = input {
                            for (k, v) in m {
                                obj.insert(k.clone(), v.clone());
                            }
                        }

                        serde_json::Value::Object(obj)
                    }

                    let tool_name = match name.as_str() {
                        "tool_search_tool_regex" | "tool_search_tool_bm25" => "tool_search",
                        "text_editor_code_execution" | "bash_code_execution" | "code_execution" => {
                            "code_execution"
                        }
                        other => other,
                    }
                    .to_string();

                    let input = match name.as_str() {
                        "text_editor_code_execution" | "bash_code_execution" | "code_execution" => {
                            wrap_code_execution_input(name, input)
                        }
                        _ => input.clone(),
                    };
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

/// Extract thinking content from Anthropic response
pub fn extract_thinking_content(content_blocks: &[AnthropicContentBlock]) -> Option<String> {
    for content_block in content_blocks {
        if content_block.r#type == "thinking" {
            return content_block.thinking.clone();
        }
    }
    None
}

/// Create Anthropic usage from response
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

/// Map Anthropic error types to `LlmError` according to official documentation
/// <https://docs.anthropic.com/en/api/errors>
pub fn map_anthropic_error(
    status_code: u16,
    error_type: &str,
    error_message: &str,
    error_details: serde_json::Value,
) -> LlmError {
    match error_type {
        "authentication_error" => LlmError::AuthenticationError(error_message.to_string()),
        "permission_error" => {
            LlmError::AuthenticationError(format!("Permission denied: {error_message}"))
        }
        "invalid_request_error" => LlmError::InvalidInput(error_message.to_string()),
        "not_found_error" => LlmError::NotFound(error_message.to_string()),
        "request_too_large" => {
            LlmError::InvalidInput(format!("Request too large: {error_message}"))
        }
        "rate_limit_error" => LlmError::RateLimitError(error_message.to_string()),
        "api_error" => LlmError::ProviderError {
            provider: "anthropic".to_string(),
            message: format!("Internal API error: {error_message}"),
            error_code: Some("api_error".to_string()),
        },
        "overloaded_error" => LlmError::ProviderError {
            provider: "anthropic".to_string(),
            message: format!("API temporarily overloaded: {error_message}"),
            error_code: Some("overloaded_error".to_string()),
        },
        _ => LlmError::ApiError {
            code: status_code,
            message: format!("Anthropic API error ({error_type}): {error_message}"),
            details: Some(error_details),
        },
    }
}

/// Convert tools to Anthropic format
pub fn convert_tools_to_anthropic_format(
    tools: &[crate::types::Tool],
) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut anthropic_tools = Vec::new();

    for tool in tools {
        match tool {
            crate::types::Tool::Function { function } => {
                let mut tool_map = serde_json::Map::new();
                tool_map.insert("name".to_string(), serde_json::json!(function.name));
                if !function.description.is_empty() {
                    tool_map.insert(
                        "description".to_string(),
                        serde_json::json!(function.description),
                    );
                }
                tool_map.insert(
                    "input_schema".to_string(),
                    serde_json::json!(function.parameters),
                );

                let mut anthropic_tool = serde_json::Value::Object(tool_map);

                // Vercel-aligned: tool-level provider options for Anthropic.
                // Example: `{ providerOptions: { anthropic: { deferLoading: true } } }`
                if let Some(opts) = function.provider_options_map.get("anthropic")
                    && let Some(obj) = opts.as_object()
                    && let Some(v) = obj
                        .get("deferLoading")
                        .or_else(|| obj.get("defer_loading"))
                        .and_then(|v| v.as_bool())
                    && let Some(map) = anthropic_tool.as_object_mut()
                {
                    map.insert("defer_loading".to_string(), serde_json::json!(v));
                }

                anthropic_tools.push(anthropic_tool);
            }
            crate::types::Tool::ProviderDefined(provider_tool) => {
                // Check if this is an Anthropic provider-defined tool
                if provider_tool.provider() == Some("anthropic") {
                    let tool_type = provider_tool.tool_type().unwrap_or("unknown");

                    // Vercel alignment:
                    // - provider tool args live in SDK-shaped camelCase (e.g., maxUses),
                    //   while Anthropic Messages API expects snake_case fields in the tool object.
                    // - Accept both shapes for backward compatibility.
                    let mut anthropic_tool = serde_json::json!({
                        "type": tool_type,
                        "name": provider_tool.name,
                    });

                    if let serde_json::Value::Object(args_map) = &provider_tool.args
                        && let serde_json::Value::Object(tool_map) = &mut anthropic_tool
                    {
                        match tool_type {
                            "code_execution_20250522" => {
                                tool_map.insert(
                                    "type".to_string(),
                                    serde_json::json!("code_execution_20250522"),
                                );
                                tool_map.insert(
                                    "name".to_string(),
                                    serde_json::json!("code_execution"),
                                );
                            }
                            "tool_search_regex_20251119" => {
                                tool_map.insert(
                                    "type".to_string(),
                                    serde_json::json!("tool_search_tool_regex_20251119"),
                                );
                                tool_map.insert(
                                    "name".to_string(),
                                    serde_json::json!("tool_search_tool_regex"),
                                );
                            }
                            "tool_search_bm25_20251119" => {
                                tool_map.insert(
                                    "type".to_string(),
                                    serde_json::json!("tool_search_tool_bm25_20251119"),
                                );
                                tool_map.insert(
                                    "name".to_string(),
                                    serde_json::json!("tool_search_tool_bm25"),
                                );
                            }
                            "web_fetch_20250910" => {
                                if let Some(v) =
                                    args_map.get("maxUses").or_else(|| args_map.get("max_uses"))
                                {
                                    tool_map.insert("max_uses".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("allowedDomains")
                                    .or_else(|| args_map.get("allowed_domains"))
                                {
                                    tool_map.insert("allowed_domains".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("blockedDomains")
                                    .or_else(|| args_map.get("blocked_domains"))
                                {
                                    tool_map.insert("blocked_domains".to_string(), v.clone());
                                }
                                if let Some(v) = args_map.get("citations") {
                                    tool_map.insert("citations".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("maxContentTokens")
                                    .or_else(|| args_map.get("max_content_tokens"))
                                {
                                    tool_map.insert("max_content_tokens".to_string(), v.clone());
                                }
                            }
                            "web_search_20250305" => {
                                if let Some(v) =
                                    args_map.get("maxUses").or_else(|| args_map.get("max_uses"))
                                {
                                    tool_map.insert("max_uses".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("allowedDomains")
                                    .or_else(|| args_map.get("allowed_domains"))
                                {
                                    tool_map.insert("allowed_domains".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("blockedDomains")
                                    .or_else(|| args_map.get("blocked_domains"))
                                {
                                    tool_map.insert("blocked_domains".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("userLocation")
                                    .or_else(|| args_map.get("user_location"))
                                {
                                    tool_map.insert("user_location".to_string(), v.clone());
                                }
                            }
                            _ => {
                                // Best-effort passthrough for unknown tools
                                for (k, v) in args_map {
                                    tool_map.insert(k.clone(), v.clone());
                                }
                            }
                        }
                    }

                    anthropic_tools.push(anthropic_tool);
                } else {
                    // Ignore provider-defined tools from other providers
                    // This allows users to mix tools for different providers
                    continue;
                }
            }
        }
    }

    Ok(anthropic_tools)
}

#[cfg(test)]
mod provider_tool_tests {
    use super::*;
    use crate::types::Tool;

    #[test]
    fn maps_anthropic_provider_defined_web_search() {
        let t = Tool::provider_defined("anthropic.web_search_20250305", "web_search").with_args(
            serde_json::json!({
                "maxUses": 2,
                "allowedDomains": ["example.com"],
                "blockedDomains": ["bad.com"]
            }),
        );
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("web_search_20250305")
        );
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("web_search"));
        assert_eq!(obj.get("max_uses").and_then(|v| v.as_u64()), Some(2));
        assert!(obj.get("allowed_domains").is_some());
        assert!(obj.get("blocked_domains").is_some());
    }

    #[test]
    fn maps_anthropic_provider_defined_web_fetch() {
        let t = Tool::provider_defined("anthropic.web_fetch_20250910", "web_fetch").with_args(
            serde_json::json!({
                "maxUses": 1,
                "allowedDomains": ["example.com"],
                "citations": { "enabled": true },
                "maxContentTokens": 2048
            }),
        );
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("web_fetch_20250910")
        );
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("web_fetch"));
        assert_eq!(obj.get("max_uses").and_then(|v| v.as_u64()), Some(1));
        assert!(obj.get("allowed_domains").is_some());
        assert!(obj.get("citations").is_some());
        assert_eq!(
            obj.get("max_content_tokens").and_then(|v| v.as_u64()),
            Some(2048)
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_tool_search_regex() {
        let t = Tool::provider_defined("anthropic.tool_search_regex_20251119", "tool_search");
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("tool_search_tool_regex_20251119")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("tool_search_tool_regex")
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_tool_search_bm25() {
        let t = Tool::provider_defined("anthropic.tool_search_bm25_20251119", "tool_search");
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("tool_search_tool_bm25_20251119")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("tool_search_tool_bm25")
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_code_execution() {
        let t = Tool::provider_defined("anthropic.code_execution_20250522", "code_execution");
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("code_execution_20250522")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("code_execution")
        );
    }
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
                if let ContentPart::Text { text } = &parts[0] {
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
                if let ContentPart::Text { text } = &parts[2] {
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

/// Convert provider-agnostic ToolChoice to Anthropic format
///
/// # Anthropic Format
///
/// - `Auto` → `{"type": "auto"}`
/// - `Required` → `{"type": "any"}`
/// - `None` → `None` (tools should be removed from request)
/// - `Tool { name }` → `{"type": "tool", "name": "..."}`
///
/// Note: Anthropic doesn't support "none" tool choice. When `ToolChoice::None` is used,
/// the caller should remove tools from the request entirely.
///
/// # Example
///
/// ```rust
/// use siumai::types::ToolChoice;
/// use siumai::providers::anthropic::utils::convert_tool_choice;
///
/// let choice = ToolChoice::tool("weather");
/// if let Some(anthropic_format) = convert_tool_choice(&choice) {
///     // Use anthropic_format in request
/// }
/// ```
pub fn convert_tool_choice(choice: &crate::types::ToolChoice) -> Option<serde_json::Value> {
    use crate::types::ToolChoice;

    match choice {
        ToolChoice::Auto => Some(serde_json::json!({
            "type": "auto"
        })),
        ToolChoice::Required => Some(serde_json::json!({
            "type": "any"
        })),
        ToolChoice::None => None, // Anthropic doesn't support 'none', remove tools instead
        ToolChoice::Tool { name } => Some(serde_json::json!({
            "type": "tool",
            "name": name
        })),
    }
}

#[cfg(test)]
mod tool_choice_tests {
    use super::*;

    #[test]
    fn test_convert_tool_choice() {
        use crate::types::ToolChoice;

        // Test Auto
        let result = convert_tool_choice(&ToolChoice::Auto);
        assert_eq!(result, Some(serde_json::json!({"type": "auto"})));

        // Test Required (maps to "any" in Anthropic)
        let result = convert_tool_choice(&ToolChoice::Required);
        assert_eq!(result, Some(serde_json::json!({"type": "any"})));

        // Test None (returns None, tools should be removed)
        let result = convert_tool_choice(&ToolChoice::None);
        assert_eq!(result, None);

        // Test Tool
        let result = convert_tool_choice(&ToolChoice::tool("weather"));
        assert_eq!(
            result,
            Some(serde_json::json!({
                "type": "tool",
                "name": "weather"
            }))
        );
    }
}

#[cfg(test)]
mod document_citations_tests {
    use super::*;
    use crate::types::{ChatMessage, ContentPart};

    #[test]
    fn attaches_citations_to_document_parts_when_enabled() {
        let msg = ChatMessage::user("hi")
            .with_content_parts(vec![ContentPart::file_url(
                "https://example.com/a.txt",
                "text/plain",
            )])
            .anthropic_document_citations_for_part(1, true)
            .anthropic_document_metadata_for_part(
                1,
                Some("My Document".to_string()),
                Some("This is background context.".to_string()),
            )
            .build();

        let (msgs, sys) = convert_messages(&[msg]).expect("convert messages");
        assert!(sys.is_none());
        assert_eq!(msgs.len(), 1);

        let arr = msgs[0].content.as_array().expect("content array");
        let doc = arr.get(1).and_then(|v| v.as_object()).expect("document");
        assert_eq!(doc.get("type").and_then(|v| v.as_str()), Some("document"));
        assert_eq!(
            doc.get("citations")
                .and_then(|v| v.get("enabled"))
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            doc.get("title").and_then(|v| v.as_str()),
            Some("My Document")
        );
        assert_eq!(
            doc.get("context").and_then(|v| v.as_str()),
            Some("This is background context.")
        );
    }
}
