use super::OpenAiResponsesEventConverter;
use super::state::{
    OpenAiResponsesFunctionCallSerializeState, OpenAiResponsesReasoningItemSerializeState,
    OpenAiResponsesSerializeState,
};

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

    fn openai_finish_reason_str(
        reason: Option<&crate::types::FinishReason>,
    ) -> Option<&'static str> {
        match reason? {
            crate::types::FinishReason::Stop | crate::types::FinishReason::StopSequence => {
                Some("stop")
            }
            crate::types::FinishReason::Length => Some("length"),
            crate::types::FinishReason::ToolCalls => Some("tool_calls"),
            crate::types::FinishReason::ContentFilter => Some("content_filter"),
            crate::types::FinishReason::Error => Some("error"),
            crate::types::FinishReason::Unknown | crate::types::FinishReason::Other(_) => None,
        }
    }

    fn openai_finish_reason_str_from_candidate(candidate: Option<&str>) -> Option<&'static str> {
        let normalized = candidate?.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return None;
        }

        match normalized.as_str() {
            "stop" | "end_turn" | "end-turn" | "stop_sequence" | "stop-sequence"
            | "stop_symbol" | "stop-symbol" => Some("stop"),
            "length" | "max_tokens" | "max_output_tokens" | "max_token" | "max_output_token" => {
                Some("length")
            }
            "tool_calls" | "tool-calls" | "function_call" => Some("tool_calls"),
            "content_filter" | "content-filter" | "safety" | "refusal" => Some("content_filter"),
            "error" | "failed" | "failure" => Some("error"),
            _ if normalized.contains("tool") => Some("tool_calls"),
            _ if normalized.contains("content")
                || normalized.contains("filter")
                || normalized.contains("safety")
                || normalized.contains("refusal") =>
            {
                Some("content_filter")
            }
            _ if normalized.contains("length") || normalized.contains("max") => Some("length"),
            _ if normalized.contains("error") || normalized.contains("fail") => Some("error"),
            _ => None,
        }
    }

    fn openai_finish_reason_str_from_finish_payload(
        data: &serde_json::Value,
    ) -> Option<&'static str> {
        let finish = data.get("finishReason")?;
        openai_finish_reason_str_from_candidate(finish.get("unified").and_then(|v| v.as_str()))
            .or_else(|| {
                openai_finish_reason_str_from_candidate(finish.get("raw").and_then(|v| v.as_str()))
            })
    }

    fn openai_response_status(finish_reason: Option<&str>) -> &'static str {
        match finish_reason {
            Some("error") => "failed",
            _ => "completed",
        }
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

    fn ensure_reasoning_item(
        state: &mut OpenAiResponsesSerializeState,
        requested_item_id: Option<&str>,
    ) -> (String, u64) {
        if let Some(item_id) = requested_item_id.filter(|s| !s.is_empty()) {
            if let Some(existing) = state.reasoning_items_by_item_id.get(item_id) {
                state.latest_reasoning_item_id = Some(item_id.to_string());
                return (item_id.to_string(), existing.output_index);
            }

            let output_index = alloc_output_index(state);
            state.reasoning_items_by_item_id.insert(
                item_id.to_string(),
                OpenAiResponsesReasoningItemSerializeState { output_index },
            );
            state.latest_reasoning_item_id = Some(item_id.to_string());
            return (item_id.to_string(), output_index);
        }

        if let Some(item_id) = state.latest_reasoning_item_id.clone() {
            if let Some(existing) = state.reasoning_items_by_item_id.get(&item_id) {
                return (item_id, existing.output_index);
            }
        }

        if let Some(item_id) = state.fallback_reasoning_item_id.clone() {
            if let Some(existing) = state.reasoning_items_by_item_id.get(&item_id) {
                state.latest_reasoning_item_id = Some(item_id.clone());
                return (item_id, existing.output_index);
            }

            let output_index = alloc_output_index(state);
            state.reasoning_items_by_item_id.insert(
                item_id.clone(),
                OpenAiResponsesReasoningItemSerializeState { output_index },
            );
            state.latest_reasoning_item_id = Some(item_id.clone());
            return (item_id, output_index);
        }

        let output_index = alloc_output_index(state);
        let item_id = format!("rs_siumai_{output_index}");
        state.reasoning_items_by_item_id.insert(
            item_id.clone(),
            OpenAiResponsesReasoningItemSerializeState { output_index },
        );
        state.fallback_reasoning_item_id = Some(item_id.clone());
        state.latest_reasoning_item_id = Some(item_id.clone());
        (item_id, output_index)
    }

    fn ensure_function_call_state<'a>(
        state: &'a mut OpenAiResponsesSerializeState,
        call_id: &str,
        output_index_seed: Option<u64>,
        default_name: Option<&str>,
    ) -> &'a mut OpenAiResponsesFunctionCallSerializeState {
        if state.function_calls_by_call_id.contains_key(call_id) {
            return state
                .function_calls_by_call_id
                .get_mut(call_id)
                .unwrap_or_else(|| unreachable!("function call state must exist"));
        }

        let output_index = output_index_seed.unwrap_or_else(|| alloc_output_index(state));
        state.function_calls_by_call_id.insert(
            call_id.to_string(),
            OpenAiResponsesFunctionCallSerializeState {
                item_id: format!("fc_siumai_{output_index}"),
                output_index,
                name: default_name.map(ToString::to_string),
                arguments: String::new(),
                arguments_done: false,
            },
        );
        state
            .function_calls_by_call_id
            .get_mut(call_id)
            .unwrap_or_else(|| unreachable!("function call state must exist"))
    }

    fn ensure_message_item(
        state: &mut OpenAiResponsesSerializeState,
        requested_item_id: Option<&str>,
        response_id_fallback: Option<&str>,
        prefer_requested_item_id: bool,
    ) -> (String, u64) {
        if state.message.output_index.is_none() {
            state.message.output_index = Some(alloc_output_index(state));
        }
        let output_index = state.message.output_index.unwrap_or(0);

        let requested_item_id = requested_item_id.filter(|item_id| !item_id.is_empty());
        if prefer_requested_item_id {
            if let Some(item_id) = requested_item_id {
                state.message.item_id = Some(item_id.to_string());
            }
        } else if state.message.item_id.is_none()
            && let Some(item_id) = requested_item_id
        {
            state.message.item_id = Some(item_id.to_string());
        }

        if state.message.item_id.is_none() {
            let fallback_id = response_id_fallback
                .filter(|response_id| !response_id.is_empty())
                .map(|response_id| format!("msg_{response_id}_0"))
                .unwrap_or_else(|| format!("msg_siumai_{output_index}"));
            state.message.item_id = Some(fallback_id);
        }

        (
            state
                .message
                .item_id
                .clone()
                .unwrap_or_else(|| format!("msg_siumai_{output_index}")),
            output_index,
        )
    }

    fn ensure_message_scaffold_emitted(
        state: &mut OpenAiResponsesSerializeState,
        item_id: &str,
        output_index: u64,
    ) -> Result<Vec<u8>, LlmError> {
        if state.message.scaffold_emitted {
            return Ok(Vec::new());
        }

        let added = serde_json::json!({
            "type": "response.output_item.added",
            "sequence_number": next_sequence_number(state),
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
            "sequence_number": next_sequence_number(state),
            "item_id": item_id,
            "output_index": output_index,
            "content_index": state.message.content_index,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            }
        });

        let mut out = Vec::new();
        out.extend_from_slice(&sse_event_frame("response.output_item.added", &added)?);
        out.extend_from_slice(&sse_event_frame(
            "response.content_part.added",
            &part_added,
        )?);
        state.message.scaffold_emitted = true;
        Ok(out)
    }

    fn emit_message_done_frames(
        state: &mut OpenAiResponsesSerializeState,
        item_id: &str,
        output_index: u64,
        final_text: &str,
        annotations: &[serde_json::Value],
        logprobs: Option<serde_json::Value>,
    ) -> Result<(Vec<u8>, serde_json::Value), LlmError> {
        let output_text_done = serde_json::json!({
            "type": "response.output_text.done",
            "sequence_number": next_sequence_number(state),
            "item_id": item_id,
            "output_index": output_index,
            "content_index": state.message.content_index,
            "text": final_text,
        });
        let mut part = serde_json::json!({
            "type": "output_text",
            "text": final_text,
            "annotations": annotations,
        });
        if let Some(logprobs) = output_text_logprobs_value(logprobs)
            && let Some(obj) = part.as_object_mut()
        {
            obj.insert("logprobs".to_string(), logprobs);
        }

        let part_done = serde_json::json!({
            "type": "response.content_part.done",
            "sequence_number": next_sequence_number(state),
            "item_id": item_id,
            "output_index": output_index,
            "content_index": state.message.content_index,
            "part": part.clone(),
        });
        let item = serde_json::json!({
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [part],
        });
        let item_done = serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": next_sequence_number(state),
            "output_index": output_index,
            "item": item.clone(),
        });

        let mut out = Vec::new();
        out.extend_from_slice(&sse_event_frame(
            "response.output_text.done",
            &output_text_done,
        )?);
        out.extend_from_slice(&sse_event_frame("response.content_part.done", &part_done)?);
        out.extend_from_slice(&sse_event_frame("response.output_item.done", &item_done)?);
        Ok((out, item))
    }

    fn output_item_id(item: &serde_json::Value) -> Option<&str> {
        item.get("id")
            .and_then(|value| value.as_str())
            .filter(|id| !id.is_empty())
    }

    fn emit_output_item_frame(
        state: &mut OpenAiResponsesSerializeState,
        event_name: &str,
        output_index: u64,
        item: serde_json::Value,
    ) -> Result<Vec<u8>, LlmError> {
        let payload = serde_json::json!({
            "type": event_name,
            "sequence_number": next_sequence_number(state),
            "output_index": output_index,
            "item": item,
        });
        sse_event_frame(event_name, &payload)
    }

    fn emit_deduped_output_item_frame(
        state: &mut OpenAiResponsesSerializeState,
        event_name: &str,
        output_index: u64,
        item: serde_json::Value,
    ) -> Result<Option<Vec<u8>>, LlmError> {
        let inserted = match event_name {
            "response.output_item.added" => output_item_id(&item)
                .map(|item_id| {
                    state
                        .emitted_output_item_added_ids
                        .insert(item_id.to_string())
                })
                .unwrap_or(true),
            "response.output_item.done" => output_item_id(&item)
                .map(|item_id| {
                    state
                        .emitted_output_item_done_ids
                        .insert(item_id.to_string())
                })
                .unwrap_or(true),
            _ => true,
        };

        if !inserted {
            return Ok(None);
        }

        emit_output_item_frame(state, event_name, output_index, item).map(Some)
    }

    fn build_reasoning_item(item_id: &str) -> serde_json::Value {
        serde_json::json!({
            "id": item_id,
            "type": "reasoning",
            "summary": [],
        })
    }

    fn build_function_call_item(
        item_id: &str,
        call_id: &str,
        name: &str,
        status: &str,
        arguments: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "id": item_id,
            "type": "function_call",
            "status": status,
            "arguments": arguments,
            "call_id": call_id,
            "name": name,
        })
    }

    fn build_mcp_call_item(
        item_id: &str,
        name: &str,
        status: &str,
        arguments: &str,
        output: Option<serde_json::Value>,
        server_label: Option<String>,
    ) -> serde_json::Value {
        let mut item = serde_json::json!({
            "id": item_id,
            "type": "mcp_call",
            "status": status,
            "name": name,
            "arguments": arguments,
        });

        if let Some(output) = output
            && let Some(obj) = item.as_object_mut()
        {
            obj.insert("output".to_string(), output);
        }
        if let Some(server_label) = server_label
            && let Some(obj) = item.as_object_mut()
        {
            obj.insert(
                "server_label".to_string(),
                serde_json::Value::String(server_label),
            );
        }

        item
    }

    fn output_text_logprobs_value(
        logprobs: Option<serde_json::Value>,
    ) -> Option<serde_json::Value> {
        let logprobs = logprobs?;
        match logprobs {
            serde_json::Value::Array(groups) => {
                let first = groups.first().cloned()?;
                if first.is_array() {
                    Some(first)
                } else {
                    Some(serde_json::Value::Array(groups))
                }
            }
            other => Some(other),
        }
    }

    fn emit_function_call_arguments_delta_frame(
        state: &mut OpenAiResponsesSerializeState,
        item_id: &str,
        output_index: u64,
        delta: &str,
    ) -> Result<Vec<u8>, LlmError> {
        let payload = serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "sequence_number": next_sequence_number(state),
            "item_id": item_id,
            "output_index": output_index,
            "delta": delta,
        });
        sse_event_frame("response.function_call_arguments.delta", &payload)
    }

    fn emit_function_call_arguments_done_frame(
        state: &mut OpenAiResponsesSerializeState,
        item_id: &str,
        output_index: u64,
        arguments: &str,
    ) -> Result<Vec<u8>, LlmError> {
        let payload = serde_json::json!({
            "type": "response.function_call_arguments.done",
            "sequence_number": next_sequence_number(state),
            "item_id": item_id,
            "output_index": output_index,
            "arguments": arguments,
        });
        sse_event_frame("response.function_call_arguments.done", &payload)
    }

    fn emit_mcp_call_arguments_done_frame(
        state: &mut OpenAiResponsesSerializeState,
        item_id: &str,
        output_index: u64,
        arguments: &str,
    ) -> Result<Vec<u8>, LlmError> {
        let payload = serde_json::json!({
            "type": "response.mcp_call_arguments.done",
            "sequence_number": next_sequence_number(state),
            "item_id": item_id,
            "output_index": output_index,
            "arguments": arguments,
        });
        sse_event_frame("response.mcp_call_arguments.done", &payload)
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
            let (item_id, output_index) =
                ensure_message_item(&mut state, None, Some(response_id.as_str()), false);
            let mut out = ensure_message_scaffold_emitted(&mut state, &item_id, output_index)?;

            state.message.text.push_str(delta);
            let payload = serde_json::json!({
                "type": "response.output_text.delta",
                "sequence_number": next_sequence_number(&mut state),
                "item_id": item_id,
                "output_index": output_index,
                "content_index": state.message.content_index,
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
                let call = ensure_function_call_state(&mut state, id, output_index_fallback, None);

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
                let item = build_function_call_item(&item_id, &call_id, &name, "in_progress", "");
                if let Some(added) = emit_deduped_output_item_frame(
                    &mut state,
                    "response.output_item.added",
                    output_index,
                    item,
                )? {
                    out.extend_from_slice(&added);
                }
            }

            if let Some((item_id, output_index, delta)) = emit_args_delta {
                out.extend_from_slice(&emit_function_call_arguments_delta_frame(
                    &mut state,
                    &item_id,
                    output_index,
                    &delta,
                )?);
            }

            Ok(out)
        }
        crate::streaming::ChatStreamEvent::ThinkingDelta { delta } => {
            // Best-effort mapping into Responses reasoning summary text deltas.
            maybe_emit_response_created(this, &mut state)?;
            ensure_response_metadata(this, &mut state);

            let (item_id, output_index) = ensure_reasoning_item(&mut state, None);
            let emit_added = state.emitted_output_item_added_ids.insert(item_id.clone());

            let mut out = Vec::new();
            if emit_added {
                out.extend_from_slice(&emit_output_item_frame(
                    &mut state,
                    "response.output_item.added",
                    output_index,
                    build_reasoning_item(&item_id),
                )?);
            }

            let payload = serde_json::json!({
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": next_sequence_number(&mut state),
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
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
                    out.extend_from_slice(&emit_function_call_arguments_done_frame(
                        &mut state,
                        &call.item_id,
                        call.output_index,
                        &call.arguments,
                    )?);
                }

                let item = build_function_call_item(
                    &call.item_id,
                    &call_id,
                    &call.name.clone().unwrap_or_else(|| "tool".to_string()),
                    "completed",
                    &call.arguments,
                );
                out.extend_from_slice(&emit_output_item_frame(
                    &mut state,
                    "response.output_item.done",
                    call.output_index,
                    item.clone(),
                )?);

                function_outputs.push(item);
            }

            // Close message output (best-effort): emit output_text.done + content_part.done + output_item.done.
            let final_text = {
                let txt = response.content.all_text();
                if txt.is_empty() {
                    state.message.text.clone()
                } else {
                    txt
                }
            };

            let annotations = state.message.annotations.clone();
            let response_logprobs =
                crate::provider_metadata::openai::OpenAiChatResponseExt::openai_metadata(response)
                    .and_then(|meta| meta.logprobs);
            let mut output: Vec<serde_json::Value> = Vec::new();
            output.extend(function_outputs);

            if let (Some(item_id), Some(output_index)) =
                (state.message.item_id.clone(), state.message.output_index)
            {
                let (message_out, item) = emit_message_done_frames(
                    &mut state,
                    &item_id,
                    output_index,
                    &final_text,
                    &annotations,
                    response_logprobs.clone(),
                )?;
                out.extend_from_slice(&message_out);
                output.push(item);
            } else if !final_text.is_empty() {
                let mut output_text = serde_json::json!({
                    "type": "output_text",
                    "text": final_text,
                });
                if let Some(logprobs) = output_text_logprobs_value(response_logprobs.clone())
                    && let Some(obj) = output_text.as_object_mut()
                {
                    obj.insert("logprobs".to_string(), logprobs);
                }
                output.push(output_text);
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
            let finish_reason = openai_finish_reason_str(response.finish_reason.as_ref());

            let payload = serde_json::json!({
                "type": "response.completed",
                "sequence_number": next_sequence_number(&mut state),
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": openai_response_status(finish_reason),
                    "model": model_id,
                    "output": output,
                    "usage": usage_json.unwrap_or(serde_json::Value::Null),
                    "finish_reason": finish_reason,
                    "metadata": {},
                }
            });

            out.extend_from_slice(&sse_event_frame("response.completed", &payload)?);
            out.extend_from_slice(&sse_done_frame());
            state.response_completed_emitted = true;
            Ok(out)
        }
        crate::streaming::ChatStreamEvent::Error { error } => {
            if state.latest_error_message.as_deref() == Some(error.as_str()) {
                return Ok(Vec::new());
            }

            state.latest_error_message = Some(error.clone());
            let payload = serde_json::json!({
                "type": "error",
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

            fn reasoning_item_id_from_part(data: &serde_json::Value) -> Option<String> {
                data.get("providerMetadata")
                    .and_then(provider_metadata_value)
                    .and_then(|metadata| metadata.get("itemId"))
                    .and_then(|value| value.as_str())
                    .or_else(|| {
                        data.get("id")
                            .and_then(|value| value.as_str())
                            .and_then(|id| id.split(':').next())
                    })
                    .filter(|item_id| !item_id.is_empty())
                    .map(ToString::to_string)
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
                    let requested_item_id = data.get("id").and_then(|v| v.as_str());
                    let (item_id, output_index) = ensure_message_item(
                        &mut state,
                        requested_item_id,
                        Some(response_id.as_str()),
                        false,
                    );
                    let mut out =
                        ensure_message_scaffold_emitted(&mut state, &item_id, output_index)?;

                    state.message.text.push_str(delta);
                    let payload = serde_json::json!({
                        "type": "response.output_text.delta",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message.content_index,
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

                    let (item_id, output_index) =
                        ensure_message_item(&mut state, Some(item_id), None, true);
                    ensure_message_scaffold_emitted(&mut state, &item_id, output_index)
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

                    let (item_id, output_index) =
                        ensure_message_item(&mut state, Some(item_id), None, false);
                    let mut out =
                        ensure_message_scaffold_emitted(&mut state, &item_id, output_index)?;
                    let final_text = state.message.text.clone();
                    let (done_frames, _) = emit_message_done_frames(
                        &mut state,
                        &item_id,
                        output_index,
                        &final_text,
                        &[],
                        None,
                    )?;
                    out.extend_from_slice(&done_frames);
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

                    let requested_item_id = reasoning_item_id_from_part(data);
                    let (item_id, output_index) =
                        ensure_reasoning_item(&mut state, requested_item_id.as_deref());
                    let emit_added = state.emitted_output_item_added_ids.insert(item_id.clone());

                    let summary_index = data
                        .get("id")
                        .and_then(|v| v.as_str())
                        .and_then(|id| id.rsplit_once(':').map(|(_, n)| n))
                        .and_then(|n| n.parse::<u64>().ok())
                        .unwrap_or(0);

                    let mut out = Vec::new();
                    if emit_added {
                        out.extend_from_slice(&emit_output_item_frame(
                            &mut state,
                            "response.output_item.added",
                            output_index,
                            build_reasoning_item(&item_id),
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

                    let Some(item_id) = reasoning_item_id_from_part(data) else {
                        return Ok(Vec::new());
                    };
                    let (item_id, output_index) =
                        ensure_reasoning_item(&mut state, Some(item_id.as_str()));

                    if !state.emitted_output_item_added_ids.insert(item_id.clone()) {
                        return Ok(Vec::new());
                    }

                    emit_output_item_frame(
                        &mut state,
                        "response.output_item.added",
                        output_index,
                        build_reasoning_item(&item_id),
                    )
                }
                "openai:reasoning-end" => {
                    maybe_emit_response_created(this, &mut state)?;

                    let Some(item_id) = reasoning_item_id_from_part(data) else {
                        return Ok(Vec::new());
                    };
                    let (item_id, output_index) =
                        ensure_reasoning_item(&mut state, Some(item_id.as_str()));

                    if !state.emitted_output_item_done_ids.insert(item_id.clone()) {
                        return Ok(Vec::new());
                    }

                    emit_output_item_frame(
                        &mut state,
                        "response.output_item.done",
                        output_index,
                        build_reasoning_item(&item_id),
                    )
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
                    let (item_id, output_index) =
                        ensure_message_item(&mut state, None, Some(response_id.as_str()), false);
                    let mut out =
                        ensure_message_scaffold_emitted(&mut state, &item_id, output_index)?;

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

                    let annotation_index = state.message.annotation_index;
                    state.message.annotation_index =
                        state.message.annotation_index.saturating_add(1);
                    state.message.annotations.push(annotation.clone());

                    let payload = serde_json::json!({
                        "type": "response.output_text.annotation.added",
                        "sequence_number": next_sequence_number(&mut state),
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": state.message.content_index,
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
                    let source_error = data
                        .get("error")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let source_error_obj = source_error.as_object();
                    let nested_error_obj = source_error
                        .get("error")
                        .and_then(|value| value.as_object())
                        .cloned();
                    let error_obj = nested_error_obj.unwrap_or_else(|| {
                        source_error_obj.cloned().unwrap_or_else(|| {
                            serde_json::Map::from_iter([(
                                "message".to_string(),
                                serde_json::Value::String("Unknown error".to_string()),
                            )])
                        })
                    });
                    let msg = error_obj
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown error")
                        .to_string();

                    state.latest_error_message = Some(msg);

                    let payload = serde_json::json!({
                        "type": "error",
                        "sequence_number": next_sequence_number(&mut state),
                        "error": error_obj,
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
                    let finish_logprobs = provider_meta.and_then(|v| v.get("logprobs")).cloned();

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
                    let final_text = state.message.text.clone();

                    let annotations = state.message.annotations.clone();
                    let mut output: Vec<serde_json::Value> = Vec::new();
                    output.extend(function_outputs);

                    if let (Some(item_id), Some(output_index)) =
                        (state.message.item_id.clone(), state.message.output_index)
                    {
                        let (message_out, item) = emit_message_done_frames(
                            &mut state,
                            &item_id,
                            output_index,
                            &final_text,
                            &annotations,
                            finish_logprobs.clone(),
                        )?;
                        out.extend_from_slice(&message_out);
                        output.push(item);
                    } else if !final_text.is_empty() {
                        let mut output_text = serde_json::json!({
                            "type": "output_text",
                            "text": final_text,
                        });
                        if let Some(logprobs) = output_text_logprobs_value(finish_logprobs.clone())
                            && let Some(obj) = output_text.as_object_mut()
                        {
                            obj.insert("logprobs".to_string(), logprobs);
                        }
                        output.push(output_text);
                    }

                    let usage = state.latest_usage.clone();
                    let usage_json = usage.as_ref().map(|u| {
                        serde_json::json!({
                            "input_tokens": u.prompt_tokens,
                            "output_tokens": u.completion_tokens,
                            "total_tokens": u.total_tokens,
                        })
                    });
                    let finish_reason = openai_finish_reason_str_from_finish_payload(data);
                    let response_status = if state.latest_error_message.is_some()
                        && data
                            .pointer("/finishReason/unified")
                            .and_then(|v| v.as_str())
                            == Some("other")
                    {
                        "failed"
                    } else {
                        openai_response_status(finish_reason)
                    };
                    let response_event_name = if response_status == "failed" {
                        "response.failed"
                    } else {
                        "response.completed"
                    };

                    let payload = serde_json::json!({
                        "type": response_event_name,
                        "sequence_number": next_sequence_number(&mut state),
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": created_at,
                            "status": response_status,
                            "error": state.latest_error_message.as_ref().map(|message| serde_json::json!({
                                "message": message,
                            })).unwrap_or(serde_json::Value::Null),
                            "model": model_id,
                            "output": output,
                            "usage": usage_json.unwrap_or(serde_json::Value::Null),
                            "finish_reason": finish_reason,
                            "metadata": {},
                        }
                    });

                    out.extend_from_slice(&sse_event_frame(response_event_name, &payload)?);
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

                    let (item_id, output_index) = {
                        let call = ensure_function_call_state(&mut state, call_id, None, None);

                        if call.name.is_some() {
                            return Ok(Vec::new());
                        }

                        call.name = Some(tool_name.to_string());
                        (call.item_id.clone(), call.output_index)
                    };

                    state.emitted_output_item_added_ids.insert(item_id.clone());
                    emit_output_item_frame(
                        &mut state,
                        "response.output_item.added",
                        output_index,
                        build_function_call_item(&item_id, call_id, tool_name, "in_progress", ""),
                    )
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

                    let (item_id, output_index, name, has_name) = {
                        let call =
                            ensure_function_call_state(&mut state, call_id, None, Some("tool"));

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
                        out.extend_from_slice(&emit_output_item_frame(
                            &mut state,
                            "response.output_item.added",
                            output_index,
                            build_function_call_item(&item_id, call_id, &name, "in_progress", ""),
                        )?);
                    }

                    out.extend_from_slice(&emit_function_call_arguments_delta_frame(
                        &mut state,
                        &item_id,
                        output_index,
                        delta,
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

                    emit_function_call_arguments_done_frame(
                        &mut state,
                        &item_id,
                        output_index,
                        &arguments,
                    )
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
                        let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("");

                        // Provider tool calls may be surfaced via output_item.added (in_progress)
                        // or output_item.done (completed). Some tools (e.g. code interpreter, MCP)
                        // only emit stream parts on done; dedupe via item id.
                        let is_done = !provider_executed || status == "completed";
                        if is_done {
                            if let Some(done) = emit_deduped_output_item_frame(
                                &mut state,
                                "response.output_item.done",
                                output_index,
                                item,
                            )? {
                                return Ok(done);
                            }
                            return Ok(Vec::new());
                        }

                        if let Some(added) = emit_deduped_output_item_frame(
                            &mut state,
                            "response.output_item.added",
                            output_index,
                            item,
                        )? {
                            return Ok(added);
                        }
                        return Ok(Vec::new());
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

                    if let Some(mcp_name) = tool_name.strip_prefix("mcp.") {
                        let output_index = provider_tool_output_index(
                            &mut state,
                            Some(call_id),
                            data.get("outputIndex").and_then(|v| v.as_u64()),
                        );
                        let arguments =
                            normalize_json_string(data.get("input")).unwrap_or_default();
                        let mut out = Vec::new();

                        if let Some(added) = emit_deduped_output_item_frame(
                            &mut state,
                            "response.output_item.added",
                            output_index,
                            build_mcp_call_item(call_id, mcp_name, "in_progress", "", None, None),
                        )? {
                            out.extend_from_slice(&added);
                        }

                        out.extend_from_slice(&emit_mcp_call_arguments_done_frame(
                            &mut state,
                            call_id,
                            output_index,
                            &arguments,
                        )?);

                        return Ok(out);
                    }

                    let (item_id, output_index, name, arguments, emit_added, emit_done) = {
                        let call = ensure_function_call_state(
                            &mut state,
                            call_id,
                            Some(output_index),
                            None,
                        );

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
                        out.extend_from_slice(&emit_output_item_frame(
                            &mut state,
                            "response.output_item.added",
                            output_index,
                            build_function_call_item(&item_id, call_id, &name, "in_progress", ""),
                        )?);
                    }

                    if emit_done {
                        out.extend_from_slice(&emit_function_call_arguments_done_frame(
                            &mut state,
                            &item_id,
                            output_index,
                            &arguments,
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

                        if let Some(done) = emit_deduped_output_item_frame(
                            &mut state,
                            "response.output_item.done",
                            output_index,
                            item,
                        )? {
                            return Ok(done);
                        }
                        return Ok(Vec::new());
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

                    if let Some(mcp_name) = tool_name.strip_prefix("mcp.") {
                        let (name, arguments, output, server_label) = result
                            .as_object()
                            .map(|obj| {
                                let name = obj
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(mcp_name)
                                    .to_string();
                                let arguments = obj
                                    .get("arguments")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("{}")
                                    .to_string();
                                let output =
                                    obj.get("output").cloned().unwrap_or_else(|| result.clone());
                                let server_label = obj
                                    .get("serverLabel")
                                    .and_then(|v| v.as_str())
                                    .map(ToString::to_string);
                                (name, arguments, output, server_label)
                            })
                            .unwrap_or_else(|| {
                                (
                                    mcp_name.to_string(),
                                    input.as_str().unwrap_or("{}").to_string(),
                                    result.clone(),
                                    None,
                                )
                            });

                        if let Some(done) = emit_deduped_output_item_frame(
                            &mut state,
                            "response.output_item.done",
                            output_index,
                            build_mcp_call_item(
                                call_id,
                                &name,
                                "completed",
                                &arguments,
                                Some(output),
                                server_label,
                            ),
                        )? {
                            return Ok(done);
                        }
                        return Ok(Vec::new());
                    }

                    let item = serde_json::json!({
                        "id": call_id,
                        "type": "custom_tool_call",
                        "status": "completed",
                        "name": tool_name,
                        "input": input,
                        "output": result,
                    });

                    if let Some(done) = emit_deduped_output_item_frame(
                        &mut state,
                        "response.output_item.done",
                        output_index,
                        item,
                    )? {
                        return Ok(done);
                    }
                    Ok(Vec::new())
                }
                "openai:tool-approval-request" => {
                    maybe_emit_response_created(this, &mut state)?;
                    ensure_response_metadata(this, &mut state);

                    let output_index = alloc_or_reuse_output_index(
                        &mut state,
                        data.get("outputIndex").and_then(|v| v.as_u64()),
                    );

                    if let Some(item) = data.get("rawItem").cloned() {
                        if let Some(done) = emit_deduped_output_item_frame(
                            &mut state,
                            "response.output_item.done",
                            output_index,
                            item,
                        )? {
                            return Ok(done);
                        }
                        return Ok(Vec::new());
                    }

                    Ok(Vec::new())
                }
                _ => Ok(Vec::new()),
            }
        }
    }
}
