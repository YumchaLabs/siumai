use super::*;

pub fn convert_messages(
    messages: &[ChatMessage],
) -> Result<(Vec<AnthropicMessage>, Option<serde_json::Value>), LlmError> {
    let mut anthropic_messages = Vec::new();
    let mut system_blocks: Vec<serde_json::Value> = Vec::new();
    let mut system_phase = true;
    let mut cache_breakpoints: usize = 0;
    const MAX_CACHE_BREAKPOINTS: usize = 4;

    fn cache_control_from_provider_options(
        provider_options: &crate::types::ProviderOptionsMap,
    ) -> Option<serde_json::Value> {
        provider_options
            .get_object("anthropic")
            .and_then(|anthropic| {
                anthropic
                    .get("cacheControl")
                    .or_else(|| anthropic.get("cache_control"))
            })
            .cloned()
    }

    // Helper: map message-level cache control to Anthropic JSON.
    fn map_cache_control(message: &ChatMessage) -> Option<serde_json::Value> {
        if let Some(cache_control) = cache_control_from_provider_options(&message.provider_options)
        {
            return Some(cache_control);
        }

        let meta = &message.metadata;
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

    fn collect_part_cache_controls(
        message: &ChatMessage,
    ) -> std::collections::HashMap<usize, serde_json::Value> {
        let mut part_cache_controls: std::collections::HashMap<usize, serde_json::Value> =
            std::collections::HashMap::new();

        if let MessageContent::MultiModal(parts) = &message.content {
            for (idx, part) in parts.iter().enumerate() {
                let Some(provider_options) = part.provider_options() else {
                    continue;
                };
                let Some(cache_control) = cache_control_from_provider_options(provider_options)
                else {
                    continue;
                };
                part_cache_controls.insert(idx, cache_control);
            }
        }

        part_cache_controls
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

    fn apply_message_cache_controls(
        message: &ChatMessage,
        content_json: &mut serde_json::Value,
        cache_breakpoints: &mut usize,
    ) {
        let part_cache_controls = collect_part_cache_controls(message);

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
                maybe_attach_cache_control(part, cc, cache_breakpoints);
            }
        }

        // Message-level cache control applies to the last content part (if any), unless already set.
        if let Some(cc) = map_cache_control(message)
            && let Some(parts) = content_json.as_array_mut()
            && let Some(last) = parts.last_mut()
        {
            maybe_attach_cache_control(last, &cc, cache_breakpoints);
        }
    }

    #[derive(Default)]
    struct AnthropicReasoningReplayMetadata {
        signature: Option<String>,
        redacted_data: Option<String>,
    }

    fn anthropic_reasoning_replay_value(
        map: &serde_json::Map<String, serde_json::Value>,
        keys: &[&str],
    ) -> Option<String> {
        keys.iter().find_map(|key| {
            map.get(*key)
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        })
    }

    fn anthropic_reasoning_replay_metadata_from_map(
        map: &serde_json::Map<String, serde_json::Value>,
    ) -> AnthropicReasoningReplayMetadata {
        AnthropicReasoningReplayMetadata {
            signature: anthropic_reasoning_replay_value(map, &["signature"]),
            redacted_data: anthropic_reasoning_replay_value(
                map,
                &["redactedData", "redacted_data", "redacted_thinking_data"],
            ),
        }
    }

    fn anthropic_part_metadata<'a>(
        message: &'a ChatMessage,
        index: usize,
    ) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
        let MessageContent::MultiModal(parts) = &message.content else {
            return None;
        };

        match parts.get(index)? {
            ContentPart::Text {
                provider_metadata: Some(provider_metadata),
                ..
            }
            | ContentPart::Image {
                provider_metadata: Some(provider_metadata),
                ..
            }
            | ContentPart::Audio {
                provider_metadata: Some(provider_metadata),
                ..
            }
            | ContentPart::File {
                provider_metadata: Some(provider_metadata),
                ..
            }
            | ContentPart::ReasoningFile {
                provider_metadata: Some(provider_metadata),
                ..
            }
            | ContentPart::Custom {
                provider_metadata: Some(provider_metadata),
                ..
            }
            | ContentPart::ToolCall {
                provider_metadata: Some(provider_metadata),
                ..
            }
            | ContentPart::ToolApprovalRequest {
                provider_metadata: Some(provider_metadata),
                ..
            }
            | ContentPart::ToolResult {
                provider_metadata: Some(provider_metadata),
                ..
            }
            | ContentPart::Reasoning {
                provider_metadata: Some(provider_metadata),
                ..
            } => provider_metadata.get("anthropic")?.as_object(),
            ContentPart::ToolApprovalResponse { .. } | ContentPart::Source { .. } => None,
            _ => None,
        }
    }

    fn anthropic_part_options<'a>(
        message: &'a ChatMessage,
        index: usize,
    ) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
        let MessageContent::MultiModal(parts) = &message.content else {
            return None;
        };

        parts
            .get(index)?
            .provider_options()?
            .get_object("anthropic")
    }

    fn anthropic_reasoning_replay_metadata(
        message: &ChatMessage,
        index: usize,
    ) -> AnthropicReasoningReplayMetadata {
        let options = anthropic_part_options(message, index)
            .map(anthropic_reasoning_replay_metadata_from_map)
            .unwrap_or_default();
        let metadata = anthropic_part_metadata(message, index)
            .map(anthropic_reasoning_replay_metadata_from_map)
            .unwrap_or_default();

        AnthropicReasoningReplayMetadata {
            signature: options.signature.or(metadata.signature),
            redacted_data: options.redacted_data.or(metadata.redacted_data),
        }
    }

    fn normalize_reasoning_blocks(message: &ChatMessage, content_json: &mut serde_json::Value) {
        let Some(parts) = content_json.as_array_mut() else {
            return;
        };

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

                    // Only assistant messages can safely replay Anthropic reasoning blocks.
                    if !matches!(message.role, MessageRole::Assistant) {
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

                    let replay = anthropic_reasoning_replay_metadata(message, idx);

                    if let Some(data) = replay.redacted_data {
                        *part = serde_json::json!({
                            "type": "redacted_thinking",
                            "data": data
                        });
                        continue;
                    }

                    if let Some(sig) = replay.signature {
                        part["signature"] = serde_json::json!(sig);
                    } else {
                        *part = serde_json::Value::Null;
                    }
                }
                "redacted_thinking" => {
                    // Remove any accidental cache_control on redacted thinking blocks.
                    if let Some(obj) = part.as_object_mut() {
                        obj.remove("cache_control");
                    }

                    if !matches!(message.role, MessageRole::Assistant) {
                        *part = serde_json::json!({
                            "type": "text",
                            "text": "<thinking>[REDACTED]</thinking>"
                        });
                    }
                }
                _ => {}
            }
        }

        parts.retain(|part| !part.is_null());
    }

    fn flatten_system_text(content: &MessageContent) -> String {
        content.all_text()
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
                    if let Some(cc) = map_cache_control(message) {
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

                normalize_reasoning_blocks(message, &mut content_json);
                apply_message_cache_controls(message, &mut content_json, &mut cache_breakpoints);

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

                normalize_reasoning_blocks(message, &mut content_json);
                apply_message_cache_controls(message, &mut content_json, &mut cache_breakpoints);

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
                    if let Some(cc) = map_cache_control(message) {
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

                normalize_reasoning_blocks(message, &mut content_json);
                apply_message_cache_controls(message, &mut content_json, &mut cache_breakpoints);

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
    fn message_provider_options_cache_control_overrides_legacy_metadata_cache_control() {
        let mut msg = ChatMessage::user("part1")
            .with_content_parts(vec![crate::types::ContentPart::text("part2")])
            .build();
        msg.metadata.cache_control = Some(crate::types::CacheControl::Persistent {
            ttl: Some(std::time::Duration::from_secs(60)),
        });
        msg.provider_options.insert(
            "anthropic",
            serde_json::json!({
                "cacheControl": {
                    "type": "ephemeral",
                    "ttl": 1
                }
            }),
        );

        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let content = msgs[0].content.as_array().expect("content array");
        assert_eq!(content[1]["cache_control"]["ttl"], serde_json::json!(1));
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
    fn legacy_message_custom_part_cache_controls_are_ignored() {
        let mut msg = ChatMessage::user("part1")
            .with_content_parts(vec![crate::types::ContentPart::text("part2")])
            .build();

        msg.metadata.custom.insert(
            "anthropic_content_cache_controls".to_string(),
            serde_json::json!({
                "0": {
                    "type": "ephemeral",
                    "ttl": 60
                }
            }),
        );

        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let content = msgs[0].content.as_array().expect("content array");
        assert!(content[0].get("cache_control").is_none());
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
    fn assistant_reasoning_without_replay_metadata_is_omitted() {
        let msg = ChatMessage::assistant_with_content(vec![ContentPart::reasoning("hi")]).build();
        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let parts = msgs[0].content.as_array().expect("content array");

        assert!(parts.is_empty());
    }

    #[test]
    fn assistant_reasoning_with_signature_emits_thinking_block() {
        let msg = ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("hi")
                .with_provider_option("anthropic", serde_json::json!({ "signature": "sig" })),
        ])
        .build();

        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let parts = msgs[0].content.as_array().expect("content array");

        assert_eq!(parts[0]["type"], "thinking");
        assert_eq!(parts[0]["thinking"], "hi");
        assert_eq!(parts[0]["signature"], "sig");
        assert!(parts[0].get("cache_control").is_none());
    }

    #[test]
    fn assistant_reasoning_with_redacted_data_emits_redacted_thinking_block() {
        let msg = ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("")
                .with_provider_option("anthropic", serde_json::json!({ "redactedData": "abc123" })),
        ])
        .build();

        let (msgs, _system) = convert_messages(&[msg]).unwrap();
        let parts = msgs[0].content.as_array().expect("content array");

        assert_eq!(parts[0]["type"], "redacted_thinking");
        assert_eq!(parts[0]["data"], "abc123");
        assert!(parts[0].get("cache_control").is_none());
    }

    #[test]
    fn cache_control_is_not_attached_to_thinking_blocks() {
        let msg = ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("hi")
                .with_provider_option("anthropic", serde_json::json!({ "signature": "sig" })),
            ContentPart::text("ok"),
        ])
        .cache_control(crate::types::CacheControl::Ephemeral)
        .cache_control_for_part(0, crate::types::CacheControl::Ephemeral)
        .build();

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

    #[test]
    fn legacy_message_custom_document_settings_are_ignored() {
        let part = ContentPart::File {
            source: crate::types::chat::MediaSource::url("https://example.com/a.txt"),
            media_type: "text/plain".to_string(),
            filename: Some("fallback.txt".to_string()),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: None,
        };

        let mut msg = ChatMessage::user("hi")
            .with_content_parts(vec![part])
            .build();
        msg.metadata.custom.insert(
            "anthropic_document_citations".to_string(),
            serde_json::json!({
                "1": { "enabled": true }
            }),
        );
        msg.metadata.custom.insert(
            "anthropic_document_metadata".to_string(),
            serde_json::json!({
                "1": {
                    "title": "Legacy Title",
                    "context": "Legacy Context"
                }
            }),
        );

        let (msgs, sys) = convert_messages(&[msg]).expect("convert messages");
        assert!(sys.is_none());
        assert_eq!(msgs.len(), 1);

        let arr = msgs[0].content.as_array().expect("content array");
        let doc = arr.get(1).and_then(|v| v.as_object()).expect("document");
        assert_eq!(doc.get("type").and_then(|v| v.as_str()), Some("document"));
        assert!(doc.get("citations").is_none());
        assert_eq!(
            doc.get("title").and_then(|v| v.as_str()),
            Some("fallback.txt")
        );
        assert!(doc.get("context").is_none());
    }
}
