use super::*;

pub(super) fn serialize_event(
    this: &AnthropicEventConverter,
    event: &ChatStreamEvent,
) -> Result<Vec<u8>, LlmError> {
    fn sse_data_frame(value: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
        let data = serde_json::to_vec(value)
            .map_err(|e| LlmError::JsonError(format!("Failed to serialize SSE JSON: {e}")))?;
        let mut out = Vec::with_capacity(data.len() + 8);
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(&data);
        out.extend_from_slice(b"\n\n");
        Ok(out)
    }

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

                let payload = serde_json::json!({
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
                });
                sse_data_frame(&payload)
            }
            ChatStreamEvent::ContentDelta { delta, .. } => {
                let mut out = Vec::new();

                let idx = match state.text_block_index {
                    Some(i) => i,
                    None => {
                        let i = state.next_block_index;
                        state.next_block_index += 1;
                        state.text_block_index = Some(i);
                        i
                    }
                };

                if !state.started_block_indices.contains(&idx) {
                    state.started_block_indices.push(idx);
                    let start = serde_json::json!({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": { "type": "text", "text": "" }
                    });
                    out.extend_from_slice(&sse_data_frame(&start)?);
                }

                let delta_payload = serde_json::json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": { "type": "text_delta", "text": delta }
                });
                out.extend_from_slice(&sse_data_frame(&delta_payload)?);
                Ok(out)
            }
            ChatStreamEvent::ThinkingDelta { delta } => {
                let mut out = Vec::new();
                let idx = match state.thinking_block_index {
                    Some(i) => i,
                    None => {
                        let i = state.next_block_index;
                        state.next_block_index += 1;
                        state.thinking_block_index = Some(i);
                        i
                    }
                };

                if !state.started_block_indices.contains(&idx) {
                    state.started_block_indices.push(idx);
                    let start = serde_json::json!({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": { "type": "thinking", "thinking": "" }
                    });
                    out.extend_from_slice(&sse_data_frame(&start)?);
                }

                let delta_payload = serde_json::json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": { "type": "thinking_delta", "thinking": delta }
                });
                out.extend_from_slice(&sse_data_frame(&delta_payload)?);
                Ok(out)
            }
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            } => {
                let mut out = Vec::new();

                let idx = match state.tool_block_index_by_id.get(id).copied() {
                    Some(i) => i,
                    None => {
                        let i = state.next_block_index;
                        state.next_block_index += 1;
                        state.tool_block_index_by_id.insert(id.clone(), i);
                        i
                    }
                };

                if !state.started_block_indices.contains(&idx) {
                    state.started_block_indices.push(idx);
                    let start = serde_json::json!({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": id,
                            "name": function_name.clone().unwrap_or_else(|| "tool".to_string()),
                            "input": {}
                        }
                    });
                    out.extend_from_slice(&sse_data_frame(&start)?);
                }

                if let Some(delta) = arguments_delta.clone() {
                    let delta_payload = serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "input_json_delta", "partial_json": delta }
                    });
                    out.extend_from_slice(&sse_data_frame(&delta_payload)?);
                }

                Ok(out)
            }
            ChatStreamEvent::UsageUpdate { usage } => {
                state.latest_usage = Some(usage.clone());
                Ok(Vec::new())
            }
            ChatStreamEvent::StreamEnd { response } => {
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

                // Close all opened content blocks (best-effort).
                let mut indices = state.started_block_indices.clone();
                indices.sort_unstable();
                for idx in indices {
                    let stop = serde_json::json!({ "type": "content_block_stop", "index": idx });
                    out.extend_from_slice(&sse_data_frame(&stop)?);
                }

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
                out.extend_from_slice(&sse_data_frame(&msg_delta)?);

                let msg_stop = serde_json::json!({ "type": "message_stop" });
                out.extend_from_slice(&sse_data_frame(&msg_stop)?);

                Ok(out)
            }
            ChatStreamEvent::Error { error } => {
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

    let mut state = this
        .serialize_state
        .lock()
        .map_err(|_| LlmError::InternalError("serialize_state lock poisoned".to_string()))?;

    match event {
        ChatStreamEvent::Custom { data, .. } => {
            let Some(part) = LanguageModelV3StreamPart::parse_loose_json(data) else {
                return Ok(Vec::new());
            };

            let mut out = Vec::new();
            let mapped = part.to_best_effort_chat_events();
            if !mapped.is_empty() {
                for ev in mapped {
                    out.extend_from_slice(&serialize_inner(&ev, &mut state)?);
                }
                return Ok(out);
            }

            if let LanguageModelV3StreamPart::Finish {
                usage,
                finish_reason,
                ..
            } = &part
            {
                // Best-effort: synthesize an Anthropic message_stop sequence.
                if state.message_id.is_none() {
                    state.message_id = Some("msg_siumai_0".to_string());
                }
                if state.model.is_none() {
                    state.model = Some("unknown".to_string());
                }

                if !state.message_start_emitted {
                    let payload = serde_json::json!({
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
                    });
                    out.extend_from_slice(&sse_data_frame(&payload)?);
                    state.message_start_emitted = true;
                }

                let mut indices = state.started_block_indices.clone();
                indices.sort_unstable();
                for idx in indices {
                    let stop = serde_json::json!({ "type": "content_block_stop", "index": idx });
                    out.extend_from_slice(&sse_data_frame(&stop)?);
                }
                state.started_block_indices.clear();
                state.text_block_index = None;
                state.thinking_block_index = None;
                state.tool_block_index_by_id.clear();

                let stop_reason = map_v3_finish_reason_unified(&finish_reason.unified)
                    .map(|s| serde_json::Value::String(s.to_string()))
                    .unwrap_or(serde_json::Value::Null);

                let prompt = usage.input_tokens.total.unwrap_or(0).min(u32::MAX as u64) as u32;
                let completion = usage.output_tokens.total.unwrap_or(0).min(u32::MAX as u64) as u32;
                let usage_obj = serde_json::json!({
                    "input_tokens": prompt,
                    "output_tokens": completion,
                });

                let msg_delta = serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": stop_reason, "stop_sequence": serde_json::Value::Null },
                    "usage": usage_obj
                });
                out.extend_from_slice(&sse_data_frame(&msg_delta)?);

                let msg_stop = serde_json::json!({ "type": "message_stop" });
                out.extend_from_slice(&sse_data_frame(&msg_stop)?);
                return Ok(out);
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
