use super::*;

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
                    if let crate::types::ContentPart::Text { text, .. } = part {
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
