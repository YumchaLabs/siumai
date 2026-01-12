use super::OpenAiResponsesEventConverter;
use super::state::{OpenAiResponsesFunctionCallSerializeState, OpenAiResponsesSerializeState};

pub(super) fn serialize_event(
    this: &super::OpenAiResponsesEventConverter,
    event: &crate::streaming::ChatStreamEvent,
) -> Result<Vec<u8>, crate::error::LlmError> {
    use crate::error::LlmError;

    fn sse_event_frame(event: &str, value: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
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

    fn sse_done_frame() -> Vec<u8> {
        // OpenAI-style end marker; some clients rely on it, others just close the connection.
        b"data: [DONE]\n\n".to_vec()
    }

    fn now_epoch_seconds() -> i64 {
        chrono::Utc::now().timestamp()
    }

    fn next_sequence_number(state: &mut OpenAiResponsesSerializeState) -> u64 {
        let n = state.next_sequence_number;
        state.next_sequence_number = state.next_sequence_number.saturating_add(1);
        n
    }

    fn ensure_response_metadata(
        this: &OpenAiResponsesEventConverter,
        state: &mut OpenAiResponsesSerializeState,
    ) -> (String, String, i64) {
        let response_id = state
            .response_id
            .clone()
            .or_else(|| this.created_response_id())
            .unwrap_or_else(|| "resp_siumai_0".to_string());
        let model_id = state
            .model_id
            .clone()
            .or_else(|| this.created_model_id())
            .unwrap_or_else(|| "unknown".to_string());
        let created_at = state
            .created_at
            .or_else(|| this.created_created_at.lock().ok().and_then(|v| *v))
            .unwrap_or_else(now_epoch_seconds);

        state.response_id = Some(response_id.clone());
        state.model_id = Some(model_id.clone());
        state.created_at = Some(created_at);

        (response_id, model_id, created_at)
    }

    fn maybe_emit_response_created(
        this: &OpenAiResponsesEventConverter,
        state: &mut OpenAiResponsesSerializeState,
    ) -> Result<Vec<u8>, LlmError> {
        if state.response_created_emitted {
            return Ok(Vec::new());
        }

        let (response_id, model_id, created_at) = ensure_response_metadata(this, state);
        state.response_created_emitted = true;

        let payload = serde_json::json!({
            "type": "response.created",
            "sequence_number": next_sequence_number(state),
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": model_id,
                "output": [],
                "usage": serde_json::Value::Null,
                "metadata": {},
            }
        });

        sse_event_frame("response.created", &payload)
    }

    fn alloc_output_index(state: &mut OpenAiResponsesSerializeState) -> u64 {
        let mut candidate = state.next_output_index;
        loop {
            if !state.used_output_indices.contains(&candidate) {
                state.used_output_indices.insert(candidate);
                state.next_output_index = candidate.saturating_add(1);
                return candidate;
            }
            candidate = candidate.saturating_add(1);
        }
    }

    fn alloc_or_reuse_output_index(
        state: &mut OpenAiResponsesSerializeState,
        requested: Option<u64>,
    ) -> u64 {
        if let Some(idx) = requested
            && !state.used_output_indices.contains(&idx)
        {
            state.used_output_indices.insert(idx);
            state.next_output_index = std::cmp::max(state.next_output_index, idx.saturating_add(1));
            return idx;
        }
        alloc_output_index(state)
    }

    fn provider_tool_output_index(
        state: &mut OpenAiResponsesSerializeState,
        tool_call_id: Option<&str>,
        requested: Option<u64>,
    ) -> u64 {
        if let Some(req) = requested {
            let idx = alloc_or_reuse_output_index(state, Some(req));
            if let Some(id) = tool_call_id
                && !id.is_empty()
            {
                state
                    .provider_tool_output_index_by_tool_call_id
                    .insert(id.to_string(), idx);
            }
            return idx;
        }

        if let Some(id) = tool_call_id
            && !id.is_empty()
        {
            if let Some(idx) = state.provider_tool_output_index_by_tool_call_id.get(id) {
                return *idx;
            }

            let idx = alloc_output_index(state);
            state
                .provider_tool_output_index_by_tool_call_id
                .insert(id.to_string(), idx);
            return idx;
        }

        alloc_output_index(state)
    }

    let mut state = this
        .serialize_state
        .lock()
        .map_err(|_| LlmError::InternalError("serialize_state lock poisoned".to_string()))?;

    match event {
        crate::streaming::ChatStreamEvent::StreamStart { metadata } => {
            *state = OpenAiResponsesSerializeState::default();

            let response_id = metadata
                .id
                .clone()
                .unwrap_or_else(|| "resp_siumai_0".to_string());
            let model_id = metadata
                .model
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            let created_at = metadata
                .created
                .map(|dt| dt.timestamp())
                .unwrap_or_else(now_epoch_seconds);

            // Persist metadata for subsequent `StreamEnd` frames (best-effort).
            this.record_created_response_metadata(&response_id, &model_id, created_at);
            state.response_id = Some(response_id.clone());
            state.model_id = Some(model_id.clone());
            state.created_at = Some(created_at);
            state.response_created_emitted = true;

            let payload = serde_json::json!({
                "type": "response.created",
                "sequence_number": next_sequence_number(&mut state),
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "in_progress",
                    "model": model_id,
                    "output": [],
                    "usage": serde_json::Value::Null,
                    "metadata": {},
                }
            });
            sse_event_frame("response.created", &payload)
        }
        crate::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
            maybe_emit_response_created(this, &mut state)?;
            let (response_id, _, _) = ensure_response_metadata(this, &mut state);

            if state.message_output_index.is_none() {
                state.message_output_index = Some(alloc_output_index(&mut state));
            }
            let output_index = state.message_output_index.unwrap_or(0);

            if state.message_item_id.is_none() {
                state.message_item_id = Some(format!("msg_{response_id}_0"));
            }
            let item_id = state
                .message_item_id
                .clone()
                .unwrap_or_else(|| "msg_siumai_0".to_string());

            // Emit message scaffolding (output_item.added + content_part.added) once.
            let mut out = Vec::new();
            if !state.message_scaffold_emitted {
                let added = serde_json::json!({
                    "type": "response.output_item.added",
                    "sequence_number": next_sequence_number(&mut state),
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                });
                out.extend_from_slice(&sse_event_frame("response.output_item.added", &added)?);

                let part_added = serde_json::json!({
                    "type": "response.content_part.added",
                    "sequence_number": next_sequence_number(&mut state),
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": state.message_content_index,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                        "logprobs": [],
                    }
                });
                out.extend_from_slice(&sse_event_frame(
                    "response.content_part.added",
                    &part_added,
                )?);
                state.message_scaffold_emitted = true;
            }

            state.message_text.push_str(delta);
            let payload = serde_json::json!({
                "type": "response.output_text.delta",
                "sequence_number": next_sequence_number(&mut state),
                "item_id": item_id,
                "output_index": output_index,
                "content_index": state.message_content_index,
                "delta": delta,
                "logprobs": [],
            });
            out.extend_from_slice(&sse_event_frame("response.output_text.delta", &payload)?);
            Ok(out)
        }
        crate::streaming::ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            index,
        } => {
            maybe_emit_response_created(this, &mut state)?;
            ensure_response_metadata(this, &mut state);

            // Map ToolCallDelta into Responses-style function_call item + arguments deltas.
            // This aligns better with real `response.output_item.added` + `response.function_call_arguments.delta`.
            let mut emit_added: Option<(String, u64, String, String)> = None;
            let mut emit_args_delta: Option<(String, u64, String)> = None;

            {
                let output_index_fallback = index
                    .map(|i| i as u64)
                    .filter(|idx| !state.used_output_indices.contains(idx));

                if !state.function_calls_by_call_id.contains_key(id) {
                    let output_index =
                        output_index_fallback.unwrap_or_else(|| alloc_output_index(&mut state));
                    state.function_calls_by_call_id.insert(
                        id.clone(),
                        OpenAiResponsesFunctionCallSerializeState {
                            item_id: format!("fc_siumai_{output_index}"),
                            output_index,
                            name: None,
                            arguments: String::new(),
                            arguments_done: false,
                        },
                    );
                }

                let call = state
                    .function_calls_by_call_id
                    .get_mut(id)
                    .unwrap_or_else(|| unreachable!("function call state must exist"));

                if let Some(name) = function_name.clone()
                    && call.name.is_none()
                {
                    call.name = Some(name);
                    emit_added = Some((
                        call.item_id.clone(),
                        call.output_index,
                        id.clone(),
                        call.name.clone().unwrap_or_else(|| "tool".to_string()),
                    ));
                }

                if let Some(delta) = arguments_delta.clone() {
                    call.arguments.push_str(&delta);
                    emit_args_delta = Some((call.item_id.clone(), call.output_index, delta));
                }
            }

            let mut out = Vec::new();
            if let Some((item_id, output_index, call_id, name)) = emit_added {
                state.emitted_output_item_added_ids.insert(item_id.clone());
                let added = serde_json::json!({
                    "type": "response.output_item.added",
                    "sequence_number": next_sequence_number(&mut state),
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "arguments": "",
                        "call_id": call_id,
                        "name": name,
                    }
                });
                out.extend_from_slice(&sse_event_frame("response.output_item.added", &added)?);
            }

            if let Some((item_id, output_index, delta)) = emit_args_delta {
                let payload = serde_json::json!({
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": next_sequence_number(&mut state),
                    "item_id": item_id,
                    "output_index": output_index,
                    "delta": delta,
                });
                out.extend_from_slice(&sse_event_frame(
                    "response.function_call_arguments.delta",
                    &payload,
                )?);
            }

            Ok(out)
        }
        crate::streaming::ChatStreamEvent::ThinkingDelta { delta } => {
            // Best-effort mapping into Responses reasoning summary text deltas.
            maybe_emit_response_created(this, &mut state)?;
            ensure_response_metadata(this, &mut state);

            if state.reasoning_output_index.is_none() {
                state.reasoning_output_index = Some(alloc_output_index(&mut state));
            }
            let output_index = state.reasoning_output_index.unwrap_or(0);

            let emit_added = state.reasoning_item_id.is_none();
            if emit_added {
                state.reasoning_item_id = Some(format!("rs_siumai_{output_index}"));
            }
            let item_id = state
                .reasoning_item_id
                .clone()
                .unwrap_or_else(|| "rs_siumai_0".to_string());

            let mut out = Vec::new();
            if emit_added {
                let added = serde_json::json!({
                    "type": "response.output_item.added",
                    "sequence_number": next_sequence_number(&mut state),
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "reasoning",
                        "summary": [],
                    }
                });
                out.extend_from_slice(&sse_event_frame("response.output_item.added", &added)?);
            }

            let payload = serde_json::json!({
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": next_sequence_number(&mut state),
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": state.reasoning_summary_index,
                "delta": delta,
            });
            out.extend_from_slice(&sse_event_frame(
                "response.reasoning_summary_text.delta",
                &payload,
            )?);
            Ok(out)
        }
        crate::streaming::ChatStreamEvent::UsageUpdate { usage } => {
            maybe_emit_response_created(this, &mut state)?;
            state.latest_usage = Some(usage.clone());
            let payload = serde_json::json!({
                "type": "response.usage",
                "sequence_number": next_sequence_number(&mut state),
                "usage": {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            });
            sse_event_frame("response.usage", &payload)
        }
        crate::streaming::ChatStreamEvent::StreamEnd { response } => {
            if state.response_completed_emitted {
                *state = OpenAiResponsesSerializeState::default();
                return Ok(Vec::new());
            }

            let (response_id, model_id, created_at) = ensure_response_metadata(this, &mut state);

            let mut out = Vec::new();

            // Close open function calls (best-effort): emit done + output_item.done.
            let mut function_outputs: Vec<serde_json::Value> = Vec::new();
            let calls: Vec<(String, OpenAiResponsesFunctionCallSerializeState)> =
                state.function_calls_by_call_id.drain().collect();
            for (call_id, call) in calls {
                if !call.arguments.is_empty() && !call.arguments_done {
                    let done = serde_json::json!({
                        "type": "response.function_call_arguments.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": call.item_id,
                        "output_index": call.output_index,
                        "arguments": call.arguments,
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.function_call_arguments.done",
                        &done,
                    )?);
                }

                let item_done = serde_json::json!({
                    "type": "response.output_item.done",
                    "sequence_number": next_sequence_number(&mut state),
                    "output_index": call.output_index,
                    "item": {
                        "id": call.item_id,
                        "type": "function_call",
                        "status": "completed",
                        "arguments": call.arguments,
                        "call_id": call_id,
                        "name": call.name.clone().unwrap_or_else(|| "tool".to_string()),
                    }
                });
                out.extend_from_slice(&sse_event_frame("response.output_item.done", &item_done)?);

                function_outputs.push(item_done["item"].clone());
            }

            // Close message output (best-effort): emit output_text.done + content_part.done + output_item.done.
            let final_text = {
                let txt = response.content.all_text();
                if txt.is_empty() {
                    state.message_text.clone()
                } else {
                    txt
                }
            };

            let mut output: Vec<serde_json::Value> = Vec::new();
            output.extend(function_outputs);

            if let (Some(item_id), Some(output_index)) =
                (state.message_item_id.clone(), state.message_output_index)
            {
                let output_text_done = serde_json::json!({
                    "type": "response.output_text.done",
                    "sequence_number": next_sequence_number(&mut state),
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": state.message_content_index,
                    "text": final_text,
                });
                out.extend_from_slice(&sse_event_frame(
                    "response.output_text.done",
                    &output_text_done,
                )?);

                let part_done = serde_json::json!({
                    "type": "response.content_part.done",
                    "sequence_number": next_sequence_number(&mut state),
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": state.message_content_index,
                    "part": {
                        "type": "output_text",
                        "text": final_text,
                        "annotations": [],
                    }
                });
                out.extend_from_slice(&sse_event_frame("response.content_part.done", &part_done)?);

                let item_done = serde_json::json!({
                    "type": "response.output_item.done",
                    "sequence_number": next_sequence_number(&mut state),
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            { "type": "output_text", "text": final_text, "annotations": [] }
                        ]
                    }
                });
                out.extend_from_slice(&sse_event_frame("response.output_item.done", &item_done)?);
                output.push(item_done["item"].clone());
            } else if !final_text.is_empty() {
                output.push(serde_json::json!({ "type": "output_text", "text": final_text }));
            }

            let usage = response
                .usage
                .clone()
                .or_else(|| state.latest_usage.clone());
            let usage_json = usage.as_ref().map(|u| {
                serde_json::json!({
                    "input_tokens": u.prompt_tokens,
                    "output_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                })
            });

            let payload = serde_json::json!({
                "type": "response.completed",
                "sequence_number": next_sequence_number(&mut state),
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "completed",
                    "model": model_id,
                    "output": output,
                    "usage": usage_json.unwrap_or(serde_json::Value::Null),
                    "metadata": {},
                }
            });

            out.extend_from_slice(&sse_event_frame("response.completed", &payload)?);
            out.extend_from_slice(&sse_done_frame());
            state.response_completed_emitted = true;
            Ok(out)
        }
        crate::streaming::ChatStreamEvent::Error { error } => {
            let payload = serde_json::json!({
                "type": "response.error",
                "sequence_number": next_sequence_number(&mut state),
                "error": { "message": error },
            });
            sse_event_frame("response.error", &payload)
        }
        crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
            fn provider_metadata_value(metadata: &serde_json::Value) -> Option<&serde_json::Value> {
                let obj = metadata.as_object()?;
                obj.get("openai").or_else(|| obj.values().next())
            }

            fn stream_part_type_to_openai_event_type(tpe: &str) -> Option<&'static str> {
                match tpe {
                    "stream-start" => Some("openai:stream-start"),
                    "response-metadata" => Some("openai:response-metadata"),
                    "text-start" => Some("openai:text-start"),
                    "text-delta" => Some("openai:text-delta"),
                    "text-end" => Some("openai:text-end"),
                    "reasoning-start" => Some("openai:reasoning-start"),
                    "reasoning-delta" => Some("openai:reasoning-delta"),
                    "reasoning-end" => Some("openai:reasoning-end"),
                    "tool-input-start" => Some("openai:tool-input-start"),
                    "tool-input-delta" => Some("openai:tool-input-delta"),
                    "tool-input-end" => Some("openai:tool-input-end"),
                    "tool-approval-request" => Some("openai:tool-approval-request"),
                    "tool-call" => Some("openai:tool-call"),
                    "tool-result" => Some("openai:tool-result"),
                    "source" => Some("openai:source"),
                    "finish" => Some("openai:finish"),
                    "error" => Some("openai:error"),
                    _ => None,
                }
            }

            fn normalize_json_string(value: Option<&serde_json::Value>) -> Option<String> {
                let v = value?;
                match v {
                    serde_json::Value::String(s) => Some(s.clone()),
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        serde_json::to_string(v).ok()
                    }
                    _ => None,
                }
            }

            // Best-effort: ignore the custom event prefix and rely on the v3 part tag
            // (`data.type`) to select the OpenAI stream part handler.
            let effective_event_type = if event_type.starts_with("openai:") {
                event_type.as_str()
            } else {
                data.get("type")
                    .and_then(|v| v.as_str())
                    .and_then(stream_part_type_to_openai_event_type)
                    .unwrap_or(event_type.as_str())
            };

            match effective_event_type {
                "openai:stream-start" => {
                    // Vercel stream part; OpenAI SSE has no direct equivalent.
                    // Reset state so subsequent parts start a fresh response.
                    *state = OpenAiResponsesSerializeState::default();
                    Ok(Vec::new())
                }
                "openai:response-metadata" => {
                    let response_id = data.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let model_id = data.get("modelId").and_then(|v| v.as_str()).unwrap_or("");
                    let timestamp = data.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");

                    if !response_id.is_empty() {
                        state.response_id = Some(response_id.to_string());
                    }
                    if !model_id.is_empty() {
                        state.model_id = Some(model_id.to_string());
                    }

                    if !timestamp.is_empty()
                        && let Ok(dt) = chrono::DateTime::parse_from_rfc3339(timestamp)
                    {
                        state.created_at = Some(dt.timestamp());
                    }

                    // Prefer emitting a real `response.created` only once we have stable ids.
                    maybe_emit_response_created(this, &mut state)
                }
                "openai:text-delta" => {
                    let Some(delta) = data.get("delta").and_then(|v| v.as_str()) else {
                        return Ok(Vec::new());
                    };
                    if delta.is_empty() {
                        return Ok(Vec::new());
                    }

                    maybe_emit_response_created(this, &mut state)?;
                    let (response_id, _, _) = ensure_response_metadata(this, &mut state);
                    if state.message_output_index.is_none() {
                        state.message_output_index = Some(alloc_output_index(&mut state));
                    }
                    let output_index = state.message_output_index.unwrap_or(0);

                    if state.message_item_id.is_none() {
                        let fallback_id = data
                            .get("id")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("msg_{response_id}_0"));
                        state.message_item_id = Some(fallback_id);
                    }
                    let item_id = state
                        .message_item_id
                        .clone()
                        .unwrap_or_else(|| "msg_siumai_0".to_string());

                    let mut out = Vec::new();
                    if !state.message_scaffold_emitted {
                        let added = serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": {
                                "id": item_id,
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_item.added",
                            &added,
                        )?);

                        let part_added = serde_json::json!({
                            "type": "response.content_part.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": state.message_content_index,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                                "logprobs": [],
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.content_part.added",
                            &part_added,
                        )?);
                        state.message_scaffold_emitted = true;
                    }

                    state.message_text.push_str(delta);
                    let payload = serde_json::json!({
                        "type": "response.output_text.delta",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message_content_index,
                        "delta": delta,
                        "logprobs": [],
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.output_text.delta",
                        &payload,
                    )?);
                    Ok(out)
                }
                "openai:text-start" => {
                    maybe_emit_response_created(this, &mut state)?;

                    let id = data.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let item_id = data
                        .get("providerMetadata")
                        .and_then(provider_metadata_value)
                        .and_then(|m| m.get("itemId"))
                        .and_then(|v| v.as_str())
                        .or(if id.is_empty() { None } else { Some(id) })
                        .unwrap_or("");

                    if item_id.is_empty() {
                        return Ok(Vec::new());
                    }

                    if state.message_output_index.is_none() {
                        state.message_output_index = Some(alloc_output_index(&mut state));
                    }
                    let output_index = state.message_output_index.unwrap_or(0);

                    state.message_item_id = Some(item_id.to_string());

                    if state.message_scaffold_emitted {
                        return Ok(Vec::new());
                    }

                    let added = serde_json::json!({
                        "type": "response.output_item.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        }
                    });

                    let part_added = serde_json::json!({
                        "type": "response.content_part.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message_content_index,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                            "logprobs": [],
                        }
                    });

                    state.message_scaffold_emitted = true;

                    let mut out = Vec::new();
                    out.extend_from_slice(&sse_event_frame("response.output_item.added", &added)?);
                    out.extend_from_slice(&sse_event_frame(
                        "response.content_part.added",
                        &part_added,
                    )?);
                    Ok(out)
                }
                "openai:text-end" => {
                    maybe_emit_response_created(this, &mut state)?;

                    let id = data.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let item_id = data
                        .get("providerMetadata")
                        .and_then(provider_metadata_value)
                        .and_then(|m| m.get("itemId"))
                        .and_then(|v| v.as_str())
                        .or(if id.is_empty() { None } else { Some(id) })
                        .unwrap_or("");
                    if item_id.is_empty() {
                        return Ok(Vec::new());
                    }

                    if state.message_item_id.is_none() {
                        state.message_item_id = Some(item_id.to_string());
                    }
                    if state.message_output_index.is_none() {
                        state.message_output_index = Some(alloc_output_index(&mut state));
                    }
                    let output_index = state.message_output_index.unwrap_or(0);

                    let final_text = state.message_text.clone();

                    let mut out = Vec::new();
                    if !state.message_scaffold_emitted {
                        let added = serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": {
                                "id": item_id,
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_item.added",
                            &added,
                        )?);

                        let part_added = serde_json::json!({
                            "type": "response.content_part.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": state.message_content_index,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                                "logprobs": [],
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.content_part.added",
                            &part_added,
                        )?);
                        state.message_scaffold_emitted = true;
                    }

                    let output_text_done = serde_json::json!({
                        "type": "response.output_text.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message_content_index,
                        "text": final_text,
                    });
                    let part_done = serde_json::json!({
                        "type": "response.content_part.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message_content_index,
                        "part": {
                            "type": "output_text",
                            "text": final_text,
                            "annotations": [],
                        }
                    });
                    let item_done = serde_json::json!({
                        "type": "response.output_item.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                { "type": "output_text", "text": final_text, "annotations": [] }
                            ]
                        }
                    });

                    out.extend_from_slice(&sse_event_frame(
                        "response.output_text.done",
                        &output_text_done,
                    )?);
                    out.extend_from_slice(&sse_event_frame(
                        "response.content_part.done",
                        &part_done,
                    )?);
                    out.extend_from_slice(&sse_event_frame(
                        "response.output_item.done",
                        &item_done,
                    )?);
                    Ok(out)
                }
                "openai:reasoning-delta" => {
                    let Some(delta) = data.get("delta").and_then(|v| v.as_str()) else {
                        return Ok(Vec::new());
                    };
                    if delta.is_empty() {
                        return Ok(Vec::new());
                    }

                    maybe_emit_response_created(this, &mut state)?;
                    ensure_response_metadata(this, &mut state);

                    if state.reasoning_output_index.is_none() {
                        state.reasoning_output_index = Some(alloc_output_index(&mut state));
                    }
                    let output_index = state.reasoning_output_index.unwrap_or(0);

                    let emit_added = state.reasoning_item_id.is_none();
                    if emit_added {
                        let item_id = data
                            .get("providerMetadata")
                            .and_then(provider_metadata_value)
                            .and_then(|m| m.get("itemId"))
                            .and_then(|v| v.as_str())
                            .or_else(|| {
                                data.get("id")
                                    .and_then(|v| v.as_str())
                                    .and_then(|id| id.split(':').next())
                            })
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("rs_siumai_{output_index}"));
                        state.reasoning_item_id = Some(item_id);
                    }
                    let item_id = state
                        .reasoning_item_id
                        .clone()
                        .unwrap_or_else(|| "rs_siumai_0".to_string());

                    let summary_index = data
                        .get("id")
                        .and_then(|v| v.as_str())
                        .and_then(|id| id.rsplit_once(':').map(|(_, n)| n))
                        .and_then(|n| n.parse::<u64>().ok())
                        .unwrap_or(state.reasoning_summary_index);

                    let mut out = Vec::new();
                    if emit_added {
                        let added = serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": {
                                "id": item_id,
                                "type": "reasoning",
                                "summary": [],
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_item.added",
                            &added,
                        )?);
                    }

                    let payload = serde_json::json!({
                        "type": "response.reasoning_summary_text.delta",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "summary_index": summary_index,
                        "delta": delta,
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.reasoning_summary_text.delta",
                        &payload,
                    )?);
                    Ok(out)
                }
                "openai:reasoning-start" => {
                    maybe_emit_response_created(this, &mut state)?;

                    let id = data.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let item_id = data
                        .get("providerMetadata")
                        .and_then(provider_metadata_value)
                        .and_then(|m| m.get("itemId"))
                        .and_then(|v| v.as_str())
                        .or_else(|| {
                            if id.is_empty() {
                                None
                            } else {
                                Some(id.split(':').next().unwrap_or(""))
                            }
                        })
                        .unwrap_or("");
                    if item_id.is_empty() {
                        return Ok(Vec::new());
                    }

                    if state.reasoning_output_index.is_none() {
                        state.reasoning_output_index = Some(alloc_output_index(&mut state));
                    }
                    let output_index = state.reasoning_output_index.unwrap_or(0);

                    if state.reasoning_item_id.is_none() {
                        state.reasoning_item_id = Some(item_id.to_string());
                    }

                    let item_id = state
                        .reasoning_item_id
                        .clone()
                        .unwrap_or_else(|| item_id.to_string());

                    if !state.emitted_output_item_added_ids.insert(item_id.clone()) {
                        return Ok(Vec::new());
                    }

                    let added = serde_json::json!({
                        "type": "response.output_item.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "reasoning",
                            "summary": [],
                        }
                    });
                    sse_event_frame("response.output_item.added", &added)
                }
                "openai:reasoning-end" => {
                    maybe_emit_response_created(this, &mut state)?;

                    let id = data.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let item_id = data
                        .get("providerMetadata")
                        .and_then(provider_metadata_value)
                        .and_then(|m| m.get("itemId"))
                        .and_then(|v| v.as_str())
                        .or_else(|| {
                            if id.is_empty() {
                                None
                            } else {
                                Some(id.split(':').next().unwrap_or(""))
                            }
                        })
                        .unwrap_or("");
                    if item_id.is_empty() {
                        return Ok(Vec::new());
                    }

                    if state.reasoning_output_index.is_none() {
                        state.reasoning_output_index = Some(alloc_output_index(&mut state));
                    }
                    let output_index = state.reasoning_output_index.unwrap_or(0);

                    if state.reasoning_item_id.is_none() {
                        state.reasoning_item_id = Some(item_id.to_string());
                    }
                    let item_id = state
                        .reasoning_item_id
                        .clone()
                        .unwrap_or_else(|| item_id.to_string());

                    if !state.emitted_output_item_done_ids.insert(item_id.clone()) {
                        return Ok(Vec::new());
                    }

                    let done = serde_json::json!({
                        "type": "response.output_item.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "reasoning",
                            "summary": [],
                        }
                    });
                    sse_event_frame("response.output_item.done", &done)
                }
                "openai:source" => {
                    // Tool result sources (e.g. web_search_call) should stay as Vercel parts and
                    // must not be mapped into output_text annotations.
                    if data.get("toolCallId").is_some() {
                        return Ok(Vec::new());
                    }

                    let Some(source_type) = data.get("sourceType").and_then(|v| v.as_str()) else {
                        return Ok(Vec::new());
                    };

                    maybe_emit_response_created(this, &mut state)?;
                    let (response_id, _, _) = ensure_response_metadata(this, &mut state);

                    if state.message_output_index.is_none() {
                        state.message_output_index = Some(alloc_output_index(&mut state));
                    }
                    let output_index = state.message_output_index.unwrap_or(0);

                    if state.message_item_id.is_none() {
                        state.message_item_id = Some(format!("msg_{response_id}_0"));
                    }
                    let item_id = state
                        .message_item_id
                        .clone()
                        .unwrap_or_else(|| "msg_siumai_0".to_string());

                    let mut out = Vec::new();
                    if !state.message_scaffold_emitted {
                        let added = serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": {
                                "id": item_id,
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_item.added",
                            &added,
                        )?);

                        let part_added = serde_json::json!({
                            "type": "response.content_part.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": state.message_content_index,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                                "logprobs": [],
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.content_part.added",
                            &part_added,
                        )?);
                        state.message_scaffold_emitted = true;
                    }

                    let annotation = match source_type {
                        "url" => {
                            let Some(url) = data.get("url").and_then(|v| v.as_str()) else {
                                return Ok(Vec::new());
                            };
                            let title = data.get("title").and_then(|v| v.as_str());
                            let mut ann = serde_json::Map::new();
                            ann.insert("type".to_string(), serde_json::json!("url_citation"));
                            ann.insert("url".to_string(), serde_json::json!(url));
                            if let Some(t) = title {
                                ann.insert("title".to_string(), serde_json::json!(t));
                            }
                            serde_json::Value::Object(ann)
                        }
                        "document" => {
                            let Some(file_id) = data.get("url").and_then(|v| v.as_str()) else {
                                return Ok(Vec::new());
                            };
                            let filename = data
                                .get("filename")
                                .and_then(|v| v.as_str())
                                .unwrap_or(file_id);
                            let quote = data.get("title").and_then(|v| v.as_str());
                            let media_type = data.get("mediaType").and_then(|v| v.as_str());

                            let provider_meta = data
                                .get("providerMetadata")
                                .and_then(provider_metadata_value);
                            let container_id = provider_meta
                                .and_then(|v| v.get("containerId"))
                                .filter(|v| !v.is_null())
                                .cloned();
                            let index = provider_meta.and_then(|v| v.get("index")).cloned();

                            let ann_type = if media_type == Some("application/octet-stream") {
                                "file_path"
                            } else if container_id.is_some() {
                                "container_file_citation"
                            } else {
                                "file_citation"
                            };

                            let mut ann = serde_json::Map::new();
                            ann.insert("type".to_string(), serde_json::json!(ann_type));
                            ann.insert("file_id".to_string(), serde_json::json!(file_id));
                            ann.insert("filename".to_string(), serde_json::json!(filename));
                            if let Some(q) = quote {
                                ann.insert("quote".to_string(), serde_json::json!(q));
                            }
                            if ann_type == "container_file_citation" {
                                if let Some(cid) = container_id {
                                    ann.insert("container_id".to_string(), cid);
                                }
                                if let Some(idx) = index {
                                    ann.insert("index".to_string(), idx);
                                }
                            } else if ann_type == "file_path"
                                && let Some(idx) = index
                            {
                                ann.insert("index".to_string(), idx);
                            }
                            serde_json::Value::Object(ann)
                        }
                        _ => return Ok(Vec::new()),
                    };

                    let annotation_index = state.message_annotation_index;
                    state.message_annotation_index =
                        state.message_annotation_index.saturating_add(1);

                    let payload = serde_json::json!({
                        "type": "response.output_text.annotation.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message_content_index,
                        "annotation_index": annotation_index,
                        "annotation": annotation,
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.output_text.annotation.added",
                        &payload,
                    )?);
                    Ok(out)
                }
                "openai:error" => {
                    let msg = data
                        .get("error")
                        .and_then(|v| v.get("error"))
                        .and_then(|v| v.get("message"))
                        .and_then(|v| v.as_str())
                        .or_else(|| {
                            data.get("error")
                                .and_then(|v| v.get("message"))
                                .and_then(|v| v.as_str())
                        })
                        .unwrap_or("Unknown error");

                    let payload = serde_json::json!({
                        "type": "response.error",
                        "sequence_number": next_sequence_number(&mut state),
                        "error": { "message": msg },
                    });
                    sse_event_frame("response.error", &payload)
                }
                "openai:finish" => {
                    if state.response_completed_emitted {
                        return Ok(Vec::new());
                    }

                    let provider_meta = data
                        .get("providerMetadata")
                        .and_then(provider_metadata_value);

                    if let Some(id) = provider_meta
                        .and_then(|v| v.get("responseId"))
                        .and_then(|v| v.as_str())
                        && !id.is_empty()
                    {
                        state.response_id = Some(id.to_string());
                    }

                    let input_tokens = data
                        .get("usage")
                        .and_then(|u| u.get("inputTokens"))
                        .and_then(|u| u.get("total"))
                        .and_then(|v| v.as_u64());
                    let output_tokens = data
                        .get("usage")
                        .and_then(|u| u.get("outputTokens"))
                        .and_then(|u| u.get("total"))
                        .and_then(|v| v.as_u64());

                    if let (Some(prompt), Some(completion)) = (input_tokens, output_tokens) {
                        state.latest_usage = Some(
                            crate::types::Usage::builder()
                                .prompt_tokens(prompt as u32)
                                .completion_tokens(completion as u32)
                                .total_tokens(prompt.saturating_add(completion) as u32)
                                .build(),
                        );
                    }

                    maybe_emit_response_created(this, &mut state)?;

                    let (response_id, model_id, created_at) =
                        ensure_response_metadata(this, &mut state);

                    let mut out = Vec::new();

                    // Close open function calls (best-effort): emit done + output_item.done.
                    let mut function_outputs: Vec<serde_json::Value> = Vec::new();
                    let calls: Vec<(String, OpenAiResponsesFunctionCallSerializeState)> =
                        state.function_calls_by_call_id.drain().collect();
                    for (call_id, call) in calls {
                        if !call.arguments.is_empty() && !call.arguments_done {
                            let done = serde_json::json!({
                                "type": "response.function_call_arguments.done",
                                "sequence_number": next_sequence_number(&mut state),
                                "item_id": call.item_id,
                                "output_index": call.output_index,
                                "arguments": call.arguments,
                            });
                            out.extend_from_slice(&sse_event_frame(
                                "response.function_call_arguments.done",
                                &done,
                            )?);
                        }

                        let item_done = serde_json::json!({
                            "type": "response.output_item.done",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": call.output_index,
                            "item": {
                                "id": call.item_id,
                                "type": "function_call",
                                "status": "completed",
                                "arguments": call.arguments,
                                "call_id": call_id,
                                "name": call.name.clone().unwrap_or_else(|| "tool".to_string()),
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_item.done",
                            &item_done,
                        )?);

                        function_outputs.push(item_done["item"].clone());
                    }

                    // Close message output (best-effort): emit output_text.done + content_part.done + output_item.done.
                    let final_text = state.message_text.clone();

                    let mut output: Vec<serde_json::Value> = Vec::new();
                    output.extend(function_outputs);

                    if let (Some(item_id), Some(output_index)) =
                        (state.message_item_id.clone(), state.message_output_index)
                    {
                        let output_text_done = serde_json::json!({
                            "type": "response.output_text.done",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": state.message_content_index,
                            "text": final_text,
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_text.done",
                            &output_text_done,
                        )?);

                        let part_done = serde_json::json!({
                            "type": "response.content_part.done",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": state.message_content_index,
                            "part": {
                                "type": "output_text",
                                "text": final_text,
                                "annotations": [],
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.content_part.done",
                            &part_done,
                        )?);

                        let item_done = serde_json::json!({
                            "type": "response.output_item.done",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": {
                                "id": item_id,
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    { "type": "output_text", "text": final_text, "annotations": [] }
                                ]
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_item.done",
                            &item_done,
                        )?);
                        output.push(item_done["item"].clone());
                    } else if !final_text.is_empty() {
                        output
                            .push(serde_json::json!({ "type": "output_text", "text": final_text }));
                    }

                    let usage = state.latest_usage.clone();
                    let usage_json = usage.as_ref().map(|u| {
                        serde_json::json!({
                            "input_tokens": u.prompt_tokens,
                            "output_tokens": u.completion_tokens,
                            "total_tokens": u.total_tokens,
                        })
                    });

                    let payload = serde_json::json!({
                        "type": "response.completed",
                        "sequence_number": next_sequence_number(&mut state),
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": created_at,
                            "status": "completed",
                            "model": model_id,
                            "output": output,
                            "usage": usage_json.unwrap_or(serde_json::Value::Null),
                            "metadata": {},
                        }
                    });

                    out.extend_from_slice(&sse_event_frame("response.completed", &payload)?);
                    out.extend_from_slice(&sse_done_frame());
                    state.response_completed_emitted = true;
                    Ok(out)
                }
                "openai:tool-input-start" => {
                    let Some(call_id) = data.get("id").and_then(|v| v.as_str()) else {
                        return Ok(Vec::new());
                    };
                    let Some(tool_name) = data.get("toolName").and_then(|v| v.as_str()) else {
                        return Ok(Vec::new());
                    };
                    if call_id.is_empty() || tool_name.is_empty() {
                        return Ok(Vec::new());
                    }

                    maybe_emit_response_created(this, &mut state)?;
                    ensure_response_metadata(this, &mut state);

                    if !state.function_calls_by_call_id.contains_key(call_id) {
                        let output_index = alloc_output_index(&mut state);
                        state.function_calls_by_call_id.insert(
                            call_id.to_string(),
                            OpenAiResponsesFunctionCallSerializeState {
                                item_id: format!("fc_siumai_{output_index}"),
                                output_index,
                                name: None,
                                arguments: String::new(),
                                arguments_done: false,
                            },
                        );
                    }

                    let (item_id, output_index) = {
                        let call = state
                            .function_calls_by_call_id
                            .get_mut(call_id)
                            .unwrap_or_else(|| unreachable!("function call state must exist"));

                        if call.name.is_some() {
                            return Ok(Vec::new());
                        }

                        call.name = Some(tool_name.to_string());
                        (call.item_id.clone(), call.output_index)
                    };

                    state.emitted_output_item_added_ids.insert(item_id.clone());

                    let added = serde_json::json!({
                        "type": "response.output_item.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "arguments": "",
                            "call_id": call_id,
                            "name": tool_name,
                        }
                    });

                    sse_event_frame("response.output_item.added", &added)
                }
                "openai:tool-input-delta" => {
                    let Some(call_id) = data.get("id").and_then(|v| v.as_str()) else {
                        return Ok(Vec::new());
                    };
                    let Some(delta) = data.get("delta").and_then(|v| v.as_str()) else {
                        return Ok(Vec::new());
                    };
                    if call_id.is_empty() || delta.is_empty() {
                        return Ok(Vec::new());
                    }

                    maybe_emit_response_created(this, &mut state)?;
                    ensure_response_metadata(this, &mut state);

                    if !state.function_calls_by_call_id.contains_key(call_id) {
                        let output_index = alloc_output_index(&mut state);
                        state.function_calls_by_call_id.insert(
                            call_id.to_string(),
                            OpenAiResponsesFunctionCallSerializeState {
                                item_id: format!("fc_siumai_{output_index}"),
                                output_index,
                                name: Some("tool".to_string()),
                                arguments: String::new(),
                                arguments_done: false,
                            },
                        );
                    }

                    let (item_id, output_index, name, has_name) = {
                        let call = state
                            .function_calls_by_call_id
                            .get_mut(call_id)
                            .unwrap_or_else(|| unreachable!("function call state must exist"));

                        call.arguments.push_str(delta);

                        (
                            call.item_id.clone(),
                            call.output_index,
                            call.name.clone().unwrap_or_else(|| "tool".to_string()),
                            call.name.is_some(),
                        )
                    };

                    // Ensure a function_call item exists so that downstream parsers can associate
                    // argument deltas with a call id.
                    let mut out = Vec::new();
                    if has_name && state.emitted_output_item_added_ids.insert(item_id.clone()) {
                        let added = serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": {
                                "id": item_id,
                                "type": "function_call",
                                "status": "in_progress",
                                "arguments": "",
                                "call_id": call_id,
                                "name": name,
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_item.added",
                            &added,
                        )?);
                    }

                    let payload = serde_json::json!({
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": delta,
                    });
                    out.extend_from_slice(&sse_event_frame(
                        "response.function_call_arguments.delta",
                        &payload,
                    )?);
                    Ok(out)
                }
                "openai:tool-input-end" => {
                    let Some(call_id) = data.get("id").and_then(|v| v.as_str()) else {
                        return Ok(Vec::new());
                    };
                    if call_id.is_empty() {
                        return Ok(Vec::new());
                    }

                    maybe_emit_response_created(this, &mut state)?;
                    ensure_response_metadata(this, &mut state);

                    let (item_id, output_index, arguments) = {
                        let Some(call) = state.function_calls_by_call_id.get_mut(call_id) else {
                            return Ok(Vec::new());
                        };
                        if call.arguments_done {
                            return Ok(Vec::new());
                        }
                        call.arguments_done = true;
                        (
                            call.item_id.clone(),
                            call.output_index,
                            call.arguments.clone(),
                        )
                    };

                    let payload = serde_json::json!({
                        "type": "response.function_call_arguments.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "arguments": arguments,
                    });
                    sse_event_frame("response.function_call_arguments.done", &payload)
                }
                "openai:tool-call" => {
                    maybe_emit_response_created(this, &mut state)?;
                    ensure_response_metadata(this, &mut state);

                    let raw_item = data.get("rawItem").cloned();
                    let provider_executed = data
                        .get("providerExecuted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    let tool_call_id_for_index =
                        data.get("toolCallId").and_then(|v| v.as_str()).or_else(|| {
                            raw_item
                                .as_ref()
                                .and_then(|it| it.get("id"))
                                .and_then(|v| v.as_str())
                        });

                    let output_index = provider_tool_output_index(
                        &mut state,
                        tool_call_id_for_index,
                        data.get("outputIndex").and_then(|v| v.as_u64()),
                    );

                    if let Some(item) = raw_item {
                        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("");

                        // Provider tool calls may be surfaced via output_item.added (in_progress)
                        // or output_item.done (completed). Some tools (e.g. code interpreter, MCP)
                        // only emit stream parts on done; dedupe via item id.
                        let is_done = !provider_executed || status == "completed";
                        if is_done {
                            if !item_id.is_empty()
                                && !state
                                    .emitted_output_item_done_ids
                                    .insert(item_id.to_string())
                            {
                                return Ok(Vec::new());
                            }

                            let payload = serde_json::json!({
                                "type": "response.output_item.done",
                                "sequence_number": next_sequence_number(&mut state),
                                "output_index": output_index,
                                "item": item,
                            });
                            return sse_event_frame("response.output_item.done", &payload);
                        }

                        if !item_id.is_empty()
                            && !state
                                .emitted_output_item_added_ids
                                .insert(item_id.to_string())
                        {
                            return Ok(Vec::new());
                        }

                        let payload = serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": item,
                        });
                        return sse_event_frame("response.output_item.added", &payload);
                    }

                    // Best-effort fallback: represent the stream part as a function_call item.
                    let call_id = data
                        .get("toolCallId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let tool_name = data.get("toolName").and_then(|v| v.as_str()).unwrap_or("");
                    let input = normalize_json_string(data.get("input")).unwrap_or_default();
                    if call_id.is_empty() || tool_name.is_empty() {
                        return Ok(Vec::new());
                    }

                    if !state.function_calls_by_call_id.contains_key(call_id) {
                        state.function_calls_by_call_id.insert(
                            call_id.to_string(),
                            OpenAiResponsesFunctionCallSerializeState {
                                item_id: format!("fc_siumai_{output_index}"),
                                output_index,
                                name: None,
                                arguments: String::new(),
                                arguments_done: false,
                            },
                        );
                    }

                    let (item_id, output_index, name, arguments, emit_added, emit_done) = {
                        let call = state
                            .function_calls_by_call_id
                            .get_mut(call_id)
                            .unwrap_or_else(|| unreachable!("function call state must exist"));

                        let mut emit_added = false;
                        if call.name.is_none() {
                            call.name = Some(tool_name.to_string());
                            emit_added = true;
                        }

                        if !input.is_empty() {
                            call.arguments.push_str(&input);
                        }

                        let mut emit_done = false;
                        if !call.arguments_done {
                            call.arguments_done = true;
                            emit_done = true;
                        }

                        (
                            call.item_id.clone(),
                            call.output_index,
                            call.name.clone().unwrap_or_else(|| "tool".to_string()),
                            call.arguments.clone(),
                            emit_added,
                            emit_done,
                        )
                    };

                    let mut out = Vec::new();

                    if emit_added && state.emitted_output_item_added_ids.insert(item_id.clone()) {
                        let added = serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": {
                                "id": item_id,
                                "type": "function_call",
                                "status": "in_progress",
                                "arguments": "",
                                "call_id": call_id,
                                "name": name,
                            }
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.output_item.added",
                            &added,
                        )?);
                    }

                    if emit_done {
                        let payload = serde_json::json!({
                            "type": "response.function_call_arguments.done",
                            "sequence_number": next_sequence_number(&mut state),
                            "item_id": item_id,
                            "output_index": output_index,
                            "arguments": arguments,
                        });
                        out.extend_from_slice(&sse_event_frame(
                            "response.function_call_arguments.done",
                            &payload,
                        )?);
                    }

                    Ok(out)
                }
                "openai:tool-result" => {
                    maybe_emit_response_created(this, &mut state)?;
                    ensure_response_metadata(this, &mut state);

                    if let Some(item) = data.get("rawItem").cloned() {
                        let tool_call_id_for_index = data
                            .get("toolCallId")
                            .and_then(|v| v.as_str())
                            .or_else(|| item.get("id").and_then(|v| v.as_str()));

                        let output_index = provider_tool_output_index(
                            &mut state,
                            tool_call_id_for_index,
                            data.get("outputIndex").and_then(|v| v.as_u64()),
                        );

                        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        if !item_id.is_empty()
                            && !state
                                .emitted_output_item_done_ids
                                .insert(item_id.to_string())
                        {
                            return Ok(Vec::new());
                        }

                        let payload = serde_json::json!({
                            "type": "response.output_item.done",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": item,
                        });
                        return sse_event_frame("response.output_item.done", &payload);
                    }

                    // Best-effort fallback: synthesize a completed tool item from v3 parts.
                    let call_id = data
                        .get("toolCallId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let tool_name = data.get("toolName").and_then(|v| v.as_str()).unwrap_or("");
                    if call_id.is_empty() || tool_name.is_empty() {
                        return Ok(Vec::new());
                    }

                    let output_index = provider_tool_output_index(
                        &mut state,
                        Some(call_id),
                        data.get("outputIndex").and_then(|v| v.as_u64()),
                    );

                    let result = data
                        .get("result")
                        .cloned()
                        .or_else(|| data.get("output").cloned())
                        .unwrap_or(serde_json::Value::Null);

                    let input = normalize_json_string(data.get("input"))
                        .map(serde_json::Value::String)
                        .unwrap_or_else(|| serde_json::Value::String("{}".to_string()));

                    let item = serde_json::json!({
                        "id": call_id,
                        "type": "custom_tool_call",
                        "status": "completed",
                        "name": tool_name,
                        "input": input,
                        "output": result,
                    });

                    if !state
                        .emitted_output_item_done_ids
                        .insert(call_id.to_string())
                    {
                        return Ok(Vec::new());
                    }

                    let payload = serde_json::json!({
                        "type": "response.output_item.done",
                        "sequence_number": next_sequence_number(&mut state),
                        "output_index": output_index,
                        "item": item,
                    });
                    sse_event_frame("response.output_item.done", &payload)
                }
                "openai:tool-approval-request" => {
                    maybe_emit_response_created(this, &mut state)?;
                    ensure_response_metadata(this, &mut state);

                    let output_index = alloc_or_reuse_output_index(
                        &mut state,
                        data.get("outputIndex").and_then(|v| v.as_u64()),
                    );

                    if let Some(item) = data.get("rawItem").cloned() {
                        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        if !item_id.is_empty()
                            && !state
                                .emitted_output_item_done_ids
                                .insert(item_id.to_string())
                        {
                            return Ok(Vec::new());
                        }

                        let payload = serde_json::json!({
                            "type": "response.output_item.done",
                            "sequence_number": next_sequence_number(&mut state),
                            "output_index": output_index,
                            "item": item,
                        });
                        return sse_event_frame("response.output_item.done", &payload);
                    }

                    Ok(Vec::new())
                }
                _ => Ok(Vec::new()),
            }
        }
    }
}
