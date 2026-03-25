use super::*;

pub(super) fn serialize_event(
    this: &AnthropicEventConverter,
    event: &ChatStreamEvent,
) -> Result<Vec<u8>, LlmError> {
    fn sse_event_data_frame(event: &str, value: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
        let data = serde_json::to_vec(value)
            .map_err(|e| LlmError::JsonError(format!("Failed to serialize SSE JSON: {e}")))?;
        let mut out = Vec::with_capacity(data.len() + event.len() + 16);
        out.extend_from_slice(b"event: ");
        out.extend_from_slice(event.as_bytes());
        out.extend_from_slice(b"\n");
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(&data);
        out.extend_from_slice(b"\n\n");
        Ok(out)
    }

    fn sse_typed_frame(value: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
        let event = value.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
            LlmError::InternalError("Anthropic SSE payload is missing `type`".to_string())
        })?;
        sse_event_data_frame(event, value)
    }

    fn reset_stream_state(state: &mut AnthropicSerializeState) {
        *state = AnthropicSerializeState::default();
    }

    fn ensure_message_metadata(state: &mut AnthropicSerializeState) {
        if state.message_id.is_none() {
            state.message_id = Some("msg_siumai_0".to_string());
        }
        if state.model.is_none() {
            state.model = Some("unknown".to_string());
        }
    }

    fn build_message_start_payload(state: &mut AnthropicSerializeState) -> serde_json::Value {
        ensure_message_metadata(state);
        serde_json::json!({
            "type": "message_start",
            "message": {
                "id": state.message_id.clone().unwrap_or_else(|| "msg_siumai_0".to_string()),
                "type": "message",
                "role": "assistant",
                "model": state.model.clone().unwrap_or_else(|| "unknown".to_string()),
                "content": [],
                "stop_reason": serde_json::Value::Null,
                "stop_sequence": serde_json::Value::Null,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        })
    }

    fn ensure_message_start_emitted(
        state: &mut AnthropicSerializeState,
    ) -> Result<Vec<u8>, LlmError> {
        if state.message_start_emitted {
            return Ok(Vec::new());
        }

        if state.terminal_emitted {
            *state = AnthropicSerializeState::default();
        }

        state.message_start_emitted = true;
        let payload = build_message_start_payload(state);
        sse_typed_frame(&payload)
    }

    fn emit_content_block_start(
        index: usize,
        content_block: serde_json::Value,
    ) -> Result<Vec<u8>, LlmError> {
        let start = serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        });
        sse_typed_frame(&start)
    }

    fn emit_content_block_stop(index: usize) -> Result<Vec<u8>, LlmError> {
        let stop = serde_json::json!({
            "type": "content_block_stop",
            "index": index,
        });
        sse_typed_frame(&stop)
    }

    fn close_active_block(state: &mut AnthropicSerializeState) -> Result<Vec<u8>, LlmError> {
        let Some(active) = state.active_block.take() else {
            return Ok(Vec::new());
        };
        emit_content_block_stop(active.index)
    }

    fn ensure_active_block<F>(
        state: &mut AnthropicSerializeState,
        kind: AnthropicSerializeBlockKind,
        build_start: F,
    ) -> Result<(Vec<u8>, usize), LlmError>
    where
        F: FnOnce(usize) -> serde_json::Value,
    {
        if let Some(active) = &state.active_block
            && active.kind == kind
        {
            return Ok((Vec::new(), active.index));
        }

        let mut out = close_active_block(state)?;
        let index = state.next_block_index;
        state.next_block_index += 1;
        out.extend_from_slice(&sse_typed_frame(&build_start(index))?);
        state.active_block = Some(AnthropicSerializeBlock { index, kind });
        Ok((out, index))
    }

    fn map_stop_reason(reason: &FinishReason) -> Option<&'static str> {
        match reason {
            FinishReason::Stop => Some("end_turn"),
            FinishReason::Length => Some("max_tokens"),
            FinishReason::ToolCalls => Some("tool_use"),
            FinishReason::ContentFilter => Some("refusal"),
            FinishReason::StopSequence => Some("stop_sequence"),
            FinishReason::Error => Some("error"),
            FinishReason::Other(_) => None,
            FinishReason::Unknown => None,
        }
    }

    fn map_v3_finish_reason_unified(unified: &str) -> Option<&'static str> {
        let u = unified.trim().to_ascii_lowercase();
        if u.is_empty() {
            return None;
        }
        if u.contains("length") || u.contains("max") {
            return Some("max_tokens");
        }
        if u.contains("tool") {
            return Some("tool_use");
        }
        if u.contains("stop") {
            return Some("end_turn");
        }
        if u.contains("safety") || u.contains("content") || u.contains("refusal") {
            return Some("refusal");
        }
        None
    }

    fn serialize_inner(
        event: &ChatStreamEvent,
        state: &mut AnthropicSerializeState,
    ) -> Result<Vec<u8>, LlmError> {
        match event {
            ChatStreamEvent::StreamStart { metadata } => {
                *state = AnthropicSerializeState::default();
                state.message_id = metadata
                    .id
                    .clone()
                    .or_else(|| Some("msg_siumai_0".to_string()));
                state.model = metadata.model.clone();
                state.message_start_emitted = true;
                let payload = build_message_start_payload(state);
                sse_typed_frame(&payload)
            }
            ChatStreamEvent::ContentDelta { delta, .. } => {
                state.last_v3_thinking_delta = None;
                state.last_v3_tool_call = None;
                if state.last_v3_text_delta.as_deref() == Some(delta.as_str()) {
                    state.last_v3_text_delta = None;
                    return Ok(Vec::new());
                }
                state.last_v3_text_delta = None;

                let (mut out, idx) =
                    ensure_active_block(state, AnthropicSerializeBlockKind::Text, |idx| {
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": { "type": "text", "text": "" }
                        })
                    })?;

                let delta_payload = serde_json::json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": { "type": "text_delta", "text": delta }
                });
                out.extend_from_slice(&sse_typed_frame(&delta_payload)?);
                Ok(out)
            }
            ChatStreamEvent::ThinkingDelta { delta } => {
                state.last_v3_text_delta = None;
                state.last_v3_tool_call = None;
                if state.last_v3_thinking_delta.as_deref() == Some(delta.as_str()) {
                    state.last_v3_thinking_delta = None;
                    return Ok(Vec::new());
                }
                state.last_v3_thinking_delta = None;

                let (mut out, idx) =
                    ensure_active_block(state, AnthropicSerializeBlockKind::Thinking, |idx| {
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": { "type": "thinking", "thinking": "" }
                        })
                    })?;

                let delta_payload = serde_json::json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": { "type": "thinking_delta", "thinking": delta }
                });
                out.extend_from_slice(&sse_typed_frame(&delta_payload)?);
                Ok(out)
            }
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            } => {
                state.last_v3_text_delta = None;
                state.last_v3_thinking_delta = None;
                let signature = AnthropicToolDeltaSignature {
                    id: id.clone(),
                    function_name: function_name.clone(),
                    arguments_delta: arguments_delta.clone(),
                };
                if state.last_v3_tool_call.as_ref() == Some(&signature) {
                    state.last_v3_tool_call = None;
                    return Ok(Vec::new());
                }
                state.last_v3_tool_call = None;
                state.seen_tool_call_ids.insert(id.clone());

                let (mut out, idx) = ensure_active_block(
                    state,
                    AnthropicSerializeBlockKind::Tool { id: id.clone() },
                    |idx| {
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": id,
                                "name": function_name.clone().unwrap_or_else(|| "tool".to_string()),
                                "input": {}
                            }
                        })
                    },
                )?;

                if let Some(delta) = arguments_delta.clone() {
                    let delta_payload = serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "input_json_delta", "partial_json": delta }
                    });
                    out.extend_from_slice(&sse_typed_frame(&delta_payload)?);
                }

                Ok(out)
            }
            ChatStreamEvent::UsageUpdate { usage } => {
                state.last_v3_text_delta = None;
                state.last_v3_thinking_delta = None;
                state.last_v3_tool_call = None;
                state.latest_usage = Some(usage.clone());
                Ok(Vec::new())
            }
            ChatStreamEvent::StreamEnd { response } => {
                state.last_v3_text_delta = None;
                state.last_v3_thinking_delta = None;
                state.last_v3_tool_call = None;
                if state.ignore_next_stream_end {
                    state.ignore_next_stream_end = false;
                    return Ok(Vec::new());
                }
                if state.terminal_emitted {
                    return Ok(Vec::new());
                }

                // Ensure we have a model/id even if StreamStart was not present.
                if state.model.is_none() {
                    state.model = response.model.clone();
                }
                if state.message_id.is_none() {
                    state.message_id = response
                        .id
                        .clone()
                        .or_else(|| Some("msg_siumai_0".to_string()));
                }

                let mut out = Vec::new();

                out.extend_from_slice(&close_active_block(state)?);

                let stop_reason = response
                    .finish_reason
                    .as_ref()
                    .and_then(map_stop_reason)
                    .map(|s| serde_json::Value::String(s.to_string()))
                    .unwrap_or(serde_json::Value::Null);

                let usage = response
                    .usage
                    .clone()
                    .or_else(|| state.latest_usage.clone());
                let usage_obj = usage.as_ref().map(|u| {
            serde_json::json!({ "input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens })
        }).unwrap_or_else(|| serde_json::json!({}));

                let msg_delta = serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": stop_reason, "stop_sequence": serde_json::Value::Null },
                    "usage": usage_obj
                });
                out.extend_from_slice(&sse_typed_frame(&msg_delta)?);

                let msg_stop = serde_json::json!({ "type": "message_stop" });
                out.extend_from_slice(&sse_typed_frame(&msg_stop)?);
                state.terminal_emitted = true;
                reset_stream_state(state);

                Ok(out)
            }
            ChatStreamEvent::Error { error } => {
                state.last_v3_text_delta = None;
                state.last_v3_thinking_delta = None;
                state.last_v3_tool_call = None;
                let payload = serde_json::json!({
                    "type": "error",
                    "error": { "type": "api_error", "message": error, "details": serde_json::Value::Null }
                });
                // Anthropic streams sometimes prefix error frames with `event: error`.
                // We do the same for better official-API compatibility (and Vercel parity).
                sse_event_data_frame("error", &payload)
            }
            ChatStreamEvent::Custom { .. } => Ok(Vec::new()),
        }
    }

    fn build_provider_content_block(data: &serde_json::Value) -> Option<serde_json::Value> {
        let raw = data.get("rawContentBlock").cloned();
        if raw.as_ref().is_some_and(serde_json::Value::is_object) {
            return raw;
        }

        match data.get("type").and_then(|v| v.as_str()) {
            Some("tool-call") => {
                let tool_call_id = data.get("toolCallId").and_then(|v| v.as_str())?;
                let tool_name = data.get("toolName").and_then(|v| v.as_str())?;
                let input = data
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                Some(serde_json::json!({
                    "type": "server_tool_use",
                    "id": tool_call_id,
                    "name": tool_name,
                    "input": input,
                }))
            }
            Some("tool-result") => {
                let tool_name = data.get("toolName").and_then(|v| v.as_str())?;
                let tool_call_id = data.get("toolCallId").and_then(|v| v.as_str())?;
                let block_type = match tool_name {
                    "web_search" => "web_search_tool_result",
                    "web_fetch" => "web_fetch_tool_result",
                    "tool_search" => "tool_search_tool_result",
                    "code_execution" => "code_execution_tool_result",
                    "mcp" => "mcp_tool_result",
                    other => {
                        return Some(serde_json::json!({
                            "type": format!("{other}_tool_result"),
                            "tool_use_id": tool_call_id,
                            "content": data.get("result").cloned().unwrap_or(serde_json::Value::Null),
                        }));
                    }
                };

                Some(serde_json::json!({
                    "type": block_type,
                    "tool_use_id": tool_call_id,
                    "content": data.get("result").cloned().unwrap_or(serde_json::Value::Null),
                }))
            }
            _ => None,
        }
    }

    fn serialize_provider_custom_event(
        data: &serde_json::Value,
        state: &mut AnthropicSerializeState,
    ) -> Result<Option<Vec<u8>>, LlmError> {
        let event_type = data.get("type").and_then(|v| v.as_str());
        if !matches!(event_type, Some("tool-call" | "tool-result")) {
            return Ok(None);
        }
        if !data
            .get("providerExecuted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return Ok(None);
        }

        let Some(content_block) = build_provider_content_block(data) else {
            return Ok(None);
        };

        let mut out = ensure_message_start_emitted(state)?;
        out.extend_from_slice(&close_active_block(state)?);

        let index = data
            .get("contentBlockIndex")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(state.next_block_index);
        state.next_block_index = state.next_block_index.max(index.saturating_add(1));

        out.extend_from_slice(&emit_content_block_start(index, content_block)?);
        out.extend_from_slice(&emit_content_block_stop(index)?);

        Ok(Some(out))
    }

    let mut state = this
        .serialize_state
        .lock()
        .map_err(|_| LlmError::InternalError("serialize_state lock poisoned".to_string()))?;

    match event {
        ChatStreamEvent::Custom { event_type, data } => {
            if matches!(
                event_type.as_str(),
                "openai:tool-input-start" | "openai:tool-input-delta" | "openai:tool-input-end"
            ) && let Some(id) = data.get("id").and_then(|v| v.as_str())
            {
                if event_type == "openai:tool-input-start"
                    && data
                        .get("providerExecuted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                {
                    state
                        .provider_executed_tool_input_ids
                        .insert(id.to_string());
                    return Ok(Vec::new());
                }

                if state.provider_executed_tool_input_ids.contains(id) {
                    return Ok(Vec::new());
                }
            }

            if let Some(out) = serialize_provider_custom_event(data, &mut state)? {
                return Ok(out);
            }

            let Some(part) = LanguageModelV3StreamPart::parse_loose_json(data) else {
                return Ok(Vec::new());
            };

            let mut out = Vec::new();
            match &part {
                LanguageModelV3StreamPart::TextDelta { delta, .. } => {
                    state.last_v3_thinking_delta = None;
                    state.last_v3_tool_call = None;
                    for ev in part.to_best_effort_chat_events() {
                        out.extend_from_slice(&serialize_inner(&ev, &mut state)?);
                    }
                    state.last_v3_text_delta = Some(delta.clone());
                    return Ok(out);
                }
                LanguageModelV3StreamPart::ReasoningDelta { delta, .. } => {
                    state.last_v3_text_delta = None;
                    state.last_v3_tool_call = None;
                    for ev in part.to_best_effort_chat_events() {
                        out.extend_from_slice(&serialize_inner(&ev, &mut state)?);
                    }
                    state.last_v3_thinking_delta = Some(delta.clone());
                    return Ok(out);
                }
                LanguageModelV3StreamPart::ToolInputStart { id, tool_name, .. } => {
                    state.last_v3_text_delta = None;
                    state.last_v3_thinking_delta = None;
                    for ev in part.to_best_effort_chat_events() {
                        out.extend_from_slice(&serialize_inner(&ev, &mut state)?);
                    }
                    state.last_v3_tool_call = Some(AnthropicToolDeltaSignature {
                        id: id.clone(),
                        function_name: Some(tool_name.clone()),
                        arguments_delta: None,
                    });
                    return Ok(out);
                }
                LanguageModelV3StreamPart::ToolInputDelta { id, delta, .. } => {
                    state.last_v3_text_delta = None;
                    state.last_v3_thinking_delta = None;
                    for ev in part.to_best_effort_chat_events() {
                        out.extend_from_slice(&serialize_inner(&ev, &mut state)?);
                    }
                    state.last_v3_tool_call = Some(AnthropicToolDeltaSignature {
                        id: id.clone(),
                        function_name: None,
                        arguments_delta: Some(delta.clone()),
                    });
                    return Ok(out);
                }
                LanguageModelV3StreamPart::ToolCall(call) => {
                    state.last_v3_text_delta = None;
                    state.last_v3_thinking_delta = None;
                    state.last_v3_tool_call = None;
                    if state.seen_tool_call_ids.contains(&call.tool_call_id) {
                        return Ok(Vec::new());
                    }

                    for ev in part.to_best_effort_chat_events() {
                        out.extend_from_slice(&serialize_inner(&ev, &mut state)?);
                    }
                    state.seen_tool_call_ids.insert(call.tool_call_id.clone());
                    return Ok(out);
                }
                LanguageModelV3StreamPart::Finish {
                    usage,
                    finish_reason,
                    ..
                } => {
                    state.last_v3_text_delta = None;
                    state.last_v3_thinking_delta = None;
                    state.last_v3_tool_call = None;
                    if state.terminal_emitted {
                        return Ok(Vec::new());
                    }

                    // Best-effort: synthesize an Anthropic message_stop sequence.
                    if state.message_id.is_none() {
                        state.message_id = Some("msg_siumai_0".to_string());
                    }
                    if state.model.is_none() {
                        state.model = Some("unknown".to_string());
                    }

                    out.extend_from_slice(&ensure_message_start_emitted(&mut state)?);

                    out.extend_from_slice(&close_active_block(&mut state)?);

                    let stop_reason = map_v3_finish_reason_unified(&finish_reason.unified)
                        .map(|s| serde_json::Value::String(s.to_string()))
                        .unwrap_or(serde_json::Value::Null);

                    let prompt = usage.input_tokens.total.unwrap_or(0).min(u32::MAX as u64) as u32;
                    let completion =
                        usage.output_tokens.total.unwrap_or(0).min(u32::MAX as u64) as u32;
                    let usage_obj = serde_json::json!({
                        "input_tokens": prompt,
                        "output_tokens": completion,
                    });

                    let msg_delta = serde_json::json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": stop_reason, "stop_sequence": serde_json::Value::Null },
                        "usage": usage_obj
                    });
                    out.extend_from_slice(&sse_typed_frame(&msg_delta)?);

                    let msg_stop = serde_json::json!({ "type": "message_stop" });
                    out.extend_from_slice(&sse_typed_frame(&msg_stop)?);
                    state.terminal_emitted = true;
                    reset_stream_state(&mut state);
                    state.ignore_next_stream_end = true;
                    return Ok(out);
                }
                _ => {}
            }

            if this.v3_unsupported_part_behavior == V3UnsupportedPartBehavior::AsText
                && let Some(text) = part.to_lossy_text()
            {
                return serialize_inner(
                    &ChatStreamEvent::ContentDelta {
                        delta: text,
                        index: None,
                    },
                    &mut state,
                );
            }

            Ok(Vec::new())
        }
        other => serialize_inner(other, &mut state),
    }
}
