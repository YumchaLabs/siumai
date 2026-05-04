use super::*;
use crate::standards::anthropic::utils::{raw_anthropic_stop_reason, replay_anthropic_stop_reason};

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

    fn ensure_active_block_at<F>(
        state: &mut AnthropicSerializeState,
        kind: AnthropicSerializeBlockKind,
        index: Option<usize>,
        build_start: F,
    ) -> Result<(Vec<u8>, usize), LlmError>
    where
        F: FnOnce(usize) -> serde_json::Value,
    {
        if let Some(active) = &state.active_block
            && active.kind == kind
            && match index {
                Some(index) => index == active.index,
                None => true,
            }
        {
            return Ok((Vec::new(), active.index));
        }

        let mut out = close_active_block(state)?;
        let index = index.unwrap_or(state.next_block_index);
        state.next_block_index = state.next_block_index.max(index.saturating_add(1));
        out.extend_from_slice(&sse_typed_frame(&build_start(index))?);
        state.active_block = Some(AnthropicSerializeBlock { index, kind });
        Ok((out, index))
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

    fn insert_usage_field_if_missing(
        usage_obj: &mut serde_json::Map<String, serde_json::Value>,
        key: &'static str,
        value: serde_json::Value,
    ) {
        usage_obj.entry(key.to_string()).or_insert(value);
    }

    fn merge_usage_summary_into_payload(
        usage_obj: &mut serde_json::Map<String, serde_json::Value>,
        usage_summary: &serde_json::Value,
    ) {
        let input_tokens = usage_summary
            .get("inputTokens")
            .and_then(|value| value.as_object());
        let output_tokens = usage_summary
            .get("outputTokens")
            .and_then(|value| value.as_object());

        if let Some(input_tokens) = input_tokens {
            let cache_read = input_tokens
                .get("cacheRead")
                .and_then(|value| value.as_u64());
            let cache_write = input_tokens
                .get("cacheWrite")
                .and_then(|value| value.as_u64());
            let no_cache = input_tokens
                .get("noCache")
                .and_then(|value| value.as_u64())
                .or_else(|| {
                    input_tokens
                        .get("total")
                        .and_then(|value| value.as_u64())
                        .map(|total| {
                            total
                                .saturating_sub(cache_read.unwrap_or(0))
                                .saturating_sub(cache_write.unwrap_or(0))
                        })
                });

            if let Some(no_cache) = no_cache {
                insert_usage_field_if_missing(
                    usage_obj,
                    "input_tokens",
                    serde_json::json!(no_cache),
                );
            }
            if let Some(cache_read) = cache_read {
                insert_usage_field_if_missing(
                    usage_obj,
                    "cache_read_input_tokens",
                    serde_json::json!(cache_read),
                );
            }
            if let Some(cache_write) = cache_write {
                insert_usage_field_if_missing(
                    usage_obj,
                    "cache_creation_input_tokens",
                    serde_json::json!(cache_write),
                );
            }
        }

        if let Some(total) = output_tokens
            .and_then(|output_tokens| output_tokens.get("total"))
            .and_then(|value| value.as_u64())
        {
            insert_usage_field_if_missing(usage_obj, "output_tokens", serde_json::json!(total));
        }
    }

    fn merge_usage_payload(
        raw_usage: Option<serde_json::Value>,
        usage: Option<&Usage>,
        service_tier: Option<&str>,
        cache_creation_input_tokens: Option<&serde_json::Value>,
    ) -> serde_json::Value {
        let mut usage_obj = raw_usage
            .or_else(|| usage.and_then(Usage::raw_usage_value))
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();

        if let Some(usage) = usage {
            let normalized_input = usage.normalized_input_tokens();
            let normalized_output = usage.normalized_output_tokens();

            if let Some(input_tokens) = normalized_input.no_cache {
                insert_usage_field_if_missing(
                    &mut usage_obj,
                    "input_tokens",
                    serde_json::json!(input_tokens),
                );
            }
            if let Some(output_tokens) = normalized_output
                .total
                .or_else(|| usage.completion_tokens_value())
            {
                insert_usage_field_if_missing(
                    &mut usage_obj,
                    "output_tokens",
                    serde_json::json!(output_tokens),
                );
            }

            if let Some(cached_tokens) = normalized_input.cache_read {
                insert_usage_field_if_missing(
                    &mut usage_obj,
                    "cache_read_input_tokens",
                    serde_json::json!(cached_tokens),
                );
            }
            if let Some(cache_write_tokens) = normalized_input.cache_write {
                insert_usage_field_if_missing(
                    &mut usage_obj,
                    "cache_creation_input_tokens",
                    serde_json::json!(cache_write_tokens),
                );
            }
        }

        if let Some(cache_creation_input_tokens) =
            cache_creation_input_tokens.filter(|value| !value.is_null())
        {
            insert_usage_field_if_missing(
                &mut usage_obj,
                "cache_creation_input_tokens",
                cache_creation_input_tokens.clone(),
            );
        }

        if let Some(service_tier) = service_tier {
            insert_usage_field_if_missing(
                &mut usage_obj,
                "service_tier",
                serde_json::json!(service_tier),
            );
        }

        serde_json::Value::Object(usage_obj)
    }

    fn usage_payload_from_finish_data(data: &serde_json::Value) -> serde_json::Value {
        let raw_usage = data
            .pointer("/providerMetadata/anthropic/usage")
            .cloned()
            .filter(serde_json::Value::is_object)
            .or_else(|| {
                data.pointer("/usage/raw")
                    .cloned()
                    .filter(serde_json::Value::is_object)
            });

        let mut usage_obj = raw_usage
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();

        if let Some(usage_summary) = data.get("usage") {
            merge_usage_summary_into_payload(&mut usage_obj, usage_summary);
        }

        serde_json::Value::Object(usage_obj)
    }

    fn build_message_delta_payload(
        stop_reason: serde_json::Value,
        stop_sequence: Option<String>,
        usage: serde_json::Value,
        container: Option<serde_json::Value>,
        context_management: Option<serde_json::Value>,
    ) -> serde_json::Value {
        let mut delta = serde_json::Map::new();
        delta.insert("stop_reason".to_string(), stop_reason);
        delta.insert(
            "stop_sequence".to_string(),
            stop_sequence
                .map(serde_json::Value::String)
                .unwrap_or(serde_json::Value::Null),
        );
        if let Some(container) = container {
            delta.insert("container".to_string(), container);
        }

        let mut payload = serde_json::Map::new();
        payload.insert(
            "type".to_string(),
            serde_json::Value::String("message_delta".to_string()),
        );
        payload.insert("delta".to_string(), serde_json::Value::Object(delta));
        payload.insert("usage".to_string(), usage);
        if let Some(context_management) = context_management {
            payload.insert("context_management".to_string(), context_management);
        }

        serde_json::Value::Object(payload)
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
            ChatStreamEvent::StreamEnd { response } => {
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

                let stop_reason = replay_anthropic_stop_reason(
                    response.raw_finish_reason.as_deref(),
                    response.finish_reason.as_ref(),
                )
                .map(|s| serde_json::Value::String(s.to_string()))
                .unwrap_or(serde_json::Value::Null);

                let anthropic_meta = response
                    .provider_metadata
                    .as_ref()
                    .and_then(|provider_metadata| provider_metadata.get("anthropic"));
                let usage = response.usage.as_ref().or(state.latest_usage.as_ref());
                let usage_obj = merge_usage_payload(
                    anthropic_meta.and_then(|meta| meta.get("usage")).cloned(),
                    usage,
                    response.service_tier.as_deref(),
                    anthropic_meta.and_then(|meta| meta.get("cacheCreationInputTokens")),
                );
                let stop_sequence = anthropic_meta
                    .and_then(|meta| meta.get("stopSequence"))
                    .and_then(|value| value.as_str())
                    .map(ToOwned::to_owned);
                let container = anthropic_meta
                    .and_then(|meta| meta.get("container"))
                    .and_then(
                        crate::standards::anthropic::utils::raw_container_from_provider_metadata,
                    );
                let context_management = anthropic_meta
                    .and_then(|meta| meta.get("contextManagement"))
                    .and_then(
                        crate::standards::anthropic::utils::raw_context_management_from_provider_metadata,
                    );

                let msg_delta = build_message_delta_payload(
                    stop_reason,
                    stop_sequence,
                    usage_obj,
                    container,
                    context_management,
                );
                out.extend_from_slice(&sse_typed_frame(&msg_delta)?);

                let msg_stop = serde_json::json!({ "type": "message_stop" });
                out.extend_from_slice(&sse_typed_frame(&msg_stop)?);
                state.terminal_emitted = true;
                reset_stream_state(state);

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
            ChatStreamEvent::Part { .. } | ChatStreamEvent::PartWithReplay { .. } => Ok(Vec::new()),
            ChatStreamEvent::Custom { .. } => Ok(Vec::new()),
            _ => Ok(Vec::new()),
        }
    }

    fn serialize_tool_input_part(
        state: &mut AnthropicSerializeState,
        id: &str,
        tool_name: Option<&str>,
        arguments_delta: Option<&str>,
        index: Option<usize>,
    ) -> Result<Vec<u8>, LlmError> {
        if state.provider_executed_tool_call_ids.contains(id) {
            return Ok(Vec::new());
        }

        state.seen_tool_call_ids.insert(id.to_string());

        let tool_name = tool_name
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or("tool")
            .to_string();
        let id_for_block = id.to_string();
        let name_for_block = tool_name.clone();
        let mut out = ensure_message_start_emitted(state)?;
        let (block_out, idx) = ensure_active_block_at(
            state,
            AnthropicSerializeBlockKind::Tool {
                id: id_for_block.clone(),
            },
            index,
            move |idx| {
                serde_json::json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": id_for_block,
                        "name": name_for_block,
                        "input": {}
                    }
                })
            },
        )?;
        out.extend_from_slice(&block_out);

        if let Some(delta) = arguments_delta {
            let delta_payload = serde_json::json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": { "type": "input_json_delta", "partial_json": delta }
            });
            out.extend_from_slice(&sse_typed_frame(&delta_payload)?);
        }

        Ok(out)
    }

    fn serialize_tool_input_end_part(
        state: &mut AnthropicSerializeState,
        id: &str,
        index: Option<usize>,
    ) -> Result<Vec<u8>, LlmError> {
        if state.provider_executed_tool_call_ids.contains(id) {
            return Ok(Vec::new());
        }

        let active_tool_index = state.active_block.as_ref().and_then(|active| {
            if matches!(&active.kind, AnthropicSerializeBlockKind::Tool { id: active_id } if active_id == id)
            {
                Some(active.index)
            } else {
                None
            }
        });

        if let Some(active_index) = active_tool_index
            && index.is_none_or(|index| index == active_index)
        {
            return close_active_block(state);
        }

        if let Some(index) = index {
            return emit_content_block_stop(index);
        }

        Ok(Vec::new())
    }

    fn serialize_lossy_text(
        text: &str,
        state: &mut AnthropicSerializeState,
    ) -> Result<Vec<u8>, LlmError> {
        let part = LanguageModelV3StreamPart::TextDelta {
            id: "text_lossy".to_string(),
            delta: text.to_string(),
            provider_metadata: None,
        };
        serialize_text_part(&part, &serde_json::Value::Null, state).map(Option::unwrap_or_default)
    }

    fn anthropic_provider_metadata(
        data: &serde_json::Value,
    ) -> Option<&serde_json::Map<String, serde_json::Value>> {
        data.pointer("/providerMetadata/anthropic")?.as_object()
    }

    fn anthropic_provider_metadata_string(data: &serde_json::Value, key: &str) -> Option<String> {
        anthropic_provider_metadata(data)?
            .get(key)?
            .as_str()
            .map(ToOwned::to_owned)
    }

    fn anthropic_tool_call_caller(data: &serde_json::Value) -> Option<serde_json::Value> {
        let caller = anthropic_provider_metadata(data)?.get("caller")?.clone();
        let caller = serde_json::from_value::<
            crate::provider_metadata::anthropic::AnthropicToolCaller,
        >(caller)
        .ok()?;
        serde_json::to_value(caller).ok()
    }

    fn parse_json_value(value: Option<&serde_json::Value>) -> serde_json::Value {
        let Some(value) = value else {
            return serde_json::json!({});
        };
        match value {
            serde_json::Value::String(text) => serde_json::from_str(text)
                .unwrap_or_else(|_| serde_json::Value::String(text.clone())),
            other => other.clone(),
        }
    }

    fn provider_content_block_index(
        data: &serde_json::Value,
        state: &AnthropicSerializeState,
    ) -> Option<usize> {
        data.get("contentBlockIndex")
            .and_then(|value| value.as_u64())
            .or_else(|| {
                data.get("id")
                    .and_then(|value| value.as_str())
                    .and_then(|value| value.parse::<u64>().ok())
            })
            .or_else(|| {
                anthropic_provider_metadata(data)
                    .and_then(|meta| meta.get("contentBlockIndex"))
                    .and_then(|value| value.as_u64())
            })
            .or_else(|| {
                data.get("toolCallId")
                    .and_then(|value| value.as_str())
                    .and_then(|tool_call_id| {
                        if state.provider_executed_tool_call_ids.contains(tool_call_id) {
                            Some(state.next_block_index as u64)
                        } else {
                            None
                        }
                    })
            })
            .and_then(|value| usize::try_from(value).ok())
    }

    fn is_json_tool_text_part(data: &serde_json::Value) -> bool {
        anthropic_provider_metadata_string(data, "type").as_deref() == Some("jsonTool")
    }

    fn json_tool_call_id(data: &serde_json::Value, index: Option<usize>) -> String {
        anthropic_provider_metadata_string(data, "toolCallId")
            .or_else(|| {
                data.get("toolCallId")
                    .and_then(|value| value.as_str())
                    .map(ToOwned::to_owned)
            })
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("toolu_siumai_json_{}", index.unwrap_or_default()))
    }

    fn serialize_json_tool_text_part(
        part: &LanguageModelV3StreamPart,
        data: &serde_json::Value,
        state: &mut AnthropicSerializeState,
    ) -> Result<Option<Vec<u8>>, LlmError> {
        match part {
            LanguageModelV3StreamPart::TextStart { .. } => {
                let index = provider_content_block_index(data, state);
                let tool_call_id = json_tool_call_id(data, index);
                let mut out = ensure_message_start_emitted(state)?;
                let (block_out, _) = ensure_active_block_at(
                    state,
                    AnthropicSerializeBlockKind::Tool {
                        id: tool_call_id.clone(),
                    },
                    index,
                    move |idx| {
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_call_id,
                                "name": "json",
                                "input": {}
                            }
                        })
                    },
                )?;
                out.extend_from_slice(&block_out);
                Ok(Some(out))
            }
            LanguageModelV3StreamPart::TextDelta { delta, .. } => {
                let index = provider_content_block_index(data, state);
                let tool_call_id = json_tool_call_id(data, index);
                let mut out = ensure_message_start_emitted(state)?;
                let (block_out, idx) = ensure_active_block_at(
                    state,
                    AnthropicSerializeBlockKind::Tool {
                        id: tool_call_id.clone(),
                    },
                    index,
                    move |idx| {
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_call_id,
                                "name": "json",
                                "input": {}
                            }
                        })
                    },
                )?;
                out.extend_from_slice(&block_out);

                let delta_payload = serde_json::json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": { "type": "input_json_delta", "partial_json": delta }
                });
                out.extend_from_slice(&sse_typed_frame(&delta_payload)?);
                Ok(Some(out))
            }
            LanguageModelV3StreamPart::TextEnd { .. } => {
                let target_index = provider_content_block_index(data, state);
                let tool_call_id = json_tool_call_id(data, target_index);
                let active_json_index = state.active_block.as_ref().and_then(|active| {
                    if matches!(&active.kind, AnthropicSerializeBlockKind::Tool { id } if id == &tool_call_id)
                    {
                        Some(active.index)
                    } else {
                        None
                    }
                });

                if let Some(active_index) = active_json_index
                    && target_index.is_none_or(|index| index == active_index)
                {
                    return Ok(Some(close_active_block(state)?));
                }

                if let Some(target_index) = target_index {
                    return Ok(Some(emit_content_block_stop(target_index)?));
                }

                Ok(Some(Vec::new()))
            }
            _ => Ok(None),
        }
    }

    fn serialize_text_part(
        part: &LanguageModelV3StreamPart,
        data: &serde_json::Value,
        state: &mut AnthropicSerializeState,
    ) -> Result<Option<Vec<u8>>, LlmError> {
        if is_json_tool_text_part(data) {
            return serialize_json_tool_text_part(part, data, state);
        }

        let is_compaction =
            anthropic_provider_metadata_string(data, "type").as_deref() == Some("compaction");
        let block_kind = if is_compaction {
            AnthropicSerializeBlockKind::Compaction
        } else {
            AnthropicSerializeBlockKind::Text
        };

        match part {
            LanguageModelV3StreamPart::TextStart { .. } => {
                let index = provider_content_block_index(data, state);
                let mut out = ensure_message_start_emitted(state)?;
                let (block_out, _) =
                    ensure_active_block_at(state, block_kind, index, move |idx| {
                        if is_compaction {
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": { "type": "compaction" }
                            })
                        } else {
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": { "type": "text", "text": "" }
                            })
                        }
                    })?;
                out.extend_from_slice(&block_out);
                Ok(Some(out))
            }
            LanguageModelV3StreamPart::TextDelta { delta, .. } => {
                let index = provider_content_block_index(data, state);
                let mut out = ensure_message_start_emitted(state)?;
                let (block_out, idx) =
                    ensure_active_block_at(state, block_kind, index, move |idx| {
                        if is_compaction {
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": { "type": "compaction" }
                            })
                        } else {
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": { "type": "text", "text": "" }
                            })
                        }
                    })?;
                out.extend_from_slice(&block_out);

                let delta_payload = if is_compaction {
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "compaction_delta", "content": delta }
                    })
                } else {
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "text_delta", "text": delta }
                    })
                };
                out.extend_from_slice(&sse_typed_frame(&delta_payload)?);
                Ok(Some(out))
            }
            LanguageModelV3StreamPart::TextEnd { .. } => {
                let active_text_index = state.active_block.as_ref().and_then(|active| {
                    if matches!(
                        active.kind,
                        AnthropicSerializeBlockKind::Text | AnthropicSerializeBlockKind::Compaction
                    ) {
                        Some(active.index)
                    } else {
                        None
                    }
                });
                let target_index = provider_content_block_index(data, state);

                if let Some(active_index) = active_text_index
                    && target_index.is_none_or(|index| index == active_index)
                {
                    return Ok(Some(close_active_block(state)?));
                }

                if let Some(target_index) = target_index {
                    return Ok(Some(emit_content_block_stop(target_index)?));
                }

                Ok(Some(Vec::new()))
            }
            _ => Ok(None),
        }
    }

    fn serialize_reasoning_part(
        part: &LanguageModelV3StreamPart,
        data: &serde_json::Value,
        state: &mut AnthropicSerializeState,
    ) -> Result<Option<Vec<u8>>, LlmError> {
        match part {
            LanguageModelV3StreamPart::ReasoningStart { .. } => {
                let redacted_data = anthropic_provider_metadata_string(data, "redactedData")
                    .filter(|value| !value.is_empty());
                let index = provider_content_block_index(data, state);
                let block_kind = if redacted_data.is_some() {
                    AnthropicSerializeBlockKind::RedactedThinking
                } else {
                    AnthropicSerializeBlockKind::Thinking
                };

                let mut out = ensure_message_start_emitted(state)?;
                let (block_out, _) =
                    ensure_active_block_at(state, block_kind, index, move |idx| {
                        if let Some(redacted_data) = redacted_data {
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {
                                    "type": "redacted_thinking",
                                    "data": redacted_data,
                                }
                            })
                        } else {
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {
                                    "type": "thinking",
                                    "thinking": "",
                                }
                            })
                        }
                    })?;
                out.extend_from_slice(&block_out);
                Ok(Some(out))
            }
            LanguageModelV3StreamPart::ReasoningDelta { delta, .. } => {
                let signature = anthropic_provider_metadata_string(data, "signature")
                    .filter(|value| !value.is_empty());
                if delta.is_empty() && signature.is_none() {
                    return Ok(Some(Vec::new()));
                }

                let index = provider_content_block_index(data, state);
                let mut out = ensure_message_start_emitted(state)?;
                let (block_out, idx) = ensure_active_block_at(
                    state,
                    AnthropicSerializeBlockKind::Thinking,
                    index,
                    |idx| {
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "thinking",
                                "thinking": "",
                            }
                        })
                    },
                )?;
                out.extend_from_slice(&block_out);

                if !delta.is_empty() {
                    let delta_payload = serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "thinking_delta", "thinking": delta }
                    });
                    out.extend_from_slice(&sse_typed_frame(&delta_payload)?);
                }

                if let Some(signature) = signature {
                    let delta_payload = serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "signature_delta", "signature": signature }
                    });
                    out.extend_from_slice(&sse_typed_frame(&delta_payload)?);
                }

                Ok(Some(out))
            }
            LanguageModelV3StreamPart::ReasoningEnd { .. } => {
                let active_reasoning_index = state.active_block.as_ref().and_then(|active| {
                    if matches!(
                        active.kind,
                        AnthropicSerializeBlockKind::Thinking
                            | AnthropicSerializeBlockKind::RedactedThinking
                    ) {
                        Some(active.index)
                    } else {
                        None
                    }
                });
                let target_index = provider_content_block_index(data, state);

                if let Some(active_index) = active_reasoning_index
                    && target_index.is_none_or(|index| index == active_index)
                {
                    return Ok(Some(close_active_block(state)?));
                }

                if let Some(target_index) = target_index {
                    return Ok(Some(emit_content_block_stop(target_index)?));
                }

                Ok(Some(Vec::new()))
            }
            _ => Ok(None),
        }
    }

    fn is_provider_tool_event(data: &serde_json::Value, state: &AnthropicSerializeState) -> bool {
        if data
            .get("providerExecuted")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            return true;
        }

        let Some(event_type) = data.get("type").and_then(|value| value.as_str()) else {
            return false;
        };

        match event_type {
            "tool-call" => {
                data.get("rawContentBlock")
                    .is_some_and(serde_json::Value::is_object)
                    || anthropic_provider_metadata(data).is_some_and(|meta| {
                        meta.get("rawContentBlock")
                            .is_some_and(serde_json::Value::is_object)
                            || meta.contains_key("caller")
                            || meta.contains_key("serverToolName")
                            || meta.contains_key("serverName")
                    })
            }
            "tool-result" => {
                let Some(tool_call_id) = data.get("toolCallId").and_then(|value| value.as_str())
                else {
                    return false;
                };

                state.provider_executed_tool_call_ids.contains(tool_call_id)
                    || data
                        .get("dynamic")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false)
                    || anthropic_provider_metadata(data).is_some_and(|meta| {
                        meta.contains_key("contentBlockIndex")
                            || meta.contains_key("serverToolName")
                            || meta.contains_key("serverName")
                            || meta
                                .get("type")
                                .and_then(|value| value.as_str())
                                .is_some_and(|value| value == "mcp-tool-use")
                    })
            }
            _ => false,
        }
    }

    fn build_provider_content_block(
        data: &serde_json::Value,
        state: &AnthropicSerializeState,
    ) -> Option<(String, serde_json::Value)> {
        let tool_call_id = data
            .get("toolCallId")
            .and_then(|value| value.as_str())?
            .to_string();
        let raw = data.get("rawContentBlock").cloned().or_else(|| {
            data.pointer("/providerMetadata/anthropic/rawContentBlock")
                .cloned()
        });
        if raw.as_ref().is_some_and(serde_json::Value::is_object) {
            return Some((tool_call_id, raw?));
        }

        match data.get("type").and_then(|v| v.as_str()) {
            Some("tool-call") => {
                let tool_name = data.get("toolName").and_then(|v| v.as_str())?;
                let input = parse_json_value(data.get("input"));
                let caller = anthropic_tool_call_caller(data);
                let raw_server_tool_name =
                    anthropic_provider_metadata_string(data, "serverToolName").or_else(|| {
                        state
                            .provider_raw_server_tool_names_by_id
                            .get(&tool_call_id)
                            .cloned()
                    });
                let mcp_server_name = anthropic_provider_metadata_string(data, "serverName")
                    .or_else(|| {
                        state
                            .provider_mcp_server_names_by_id
                            .get(&tool_call_id)
                            .cloned()
                    });
                let provider_executed = data
                    .get("providerExecuted")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                let content_block = if provider_executed {
                    crate::standards::anthropic::json_response::provider_tool_use_block(
                        &tool_call_id,
                        tool_name,
                        &input,
                        caller,
                        raw_server_tool_name.as_deref(),
                        mcp_server_name.as_deref(),
                    )
                } else {
                    crate::standards::anthropic::json_response::tool_use_block(
                        &tool_call_id,
                        tool_name,
                        &input,
                        caller,
                    )
                };
                Some((tool_call_id, content_block))
            }
            Some("tool-result") => {
                let tool_name = data.get("toolName").and_then(|v| v.as_str())?;
                let result = data
                    .get("result")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let is_error = data
                    .get("isError")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                let output = if is_error {
                    crate::types::ToolResultOutput::error_json(result)
                } else {
                    crate::types::ToolResultOutput::json(result)
                };
                let raw_server_tool_name =
                    anthropic_provider_metadata_string(data, "serverToolName").or_else(|| {
                        state
                            .provider_raw_server_tool_names_by_id
                            .get(&tool_call_id)
                            .cloned()
                    });
                let content_block =
                    crate::standards::anthropic::json_response::provider_tool_result_block(
                        &tool_call_id,
                        tool_name,
                        &output,
                        raw_server_tool_name.as_deref(),
                    );
                Some((tool_call_id, content_block))
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
        if !is_provider_tool_event(data, state) {
            return Ok(None);
        }

        let Some((tool_call_id, content_block)) = build_provider_content_block(data, state) else {
            return Ok(None);
        };

        let mut out = ensure_message_start_emitted(state)?;
        out.extend_from_slice(&close_active_block(state)?);

        let index = provider_content_block_index(data, state).unwrap_or(state.next_block_index);
        state.next_block_index = state.next_block_index.max(index.saturating_add(1));

        if matches!(event_type, Some("tool-call")) {
            state
                .provider_executed_tool_call_ids
                .insert(tool_call_id.clone());
            match content_block.get("type").and_then(|value| value.as_str()) {
                Some("server_tool_use") => {
                    if let Some(server_tool_name) =
                        content_block.get("name").and_then(|value| value.as_str())
                    {
                        state
                            .provider_raw_server_tool_names_by_id
                            .insert(tool_call_id.clone(), server_tool_name.to_string());
                    }
                }
                Some("mcp_tool_use") => {
                    if let Some(server_name) = content_block
                        .get("server_name")
                        .and_then(|value| value.as_str())
                    {
                        state
                            .provider_mcp_server_names_by_id
                            .insert(tool_call_id.clone(), server_name.to_string());
                    }
                }
                _ => {}
            }
        }

        out.extend_from_slice(&emit_content_block_start(index, content_block)?);
        out.extend_from_slice(&emit_content_block_stop(index)?);

        Ok(Some(out))
    }

    if matches!(
        event,
        ChatStreamEvent::Part { .. } | ChatStreamEvent::PartWithReplay { .. }
    ) {
        let Some(part) = LanguageModelV3StreamPart::try_from_chat_event(event) else {
            return Ok(Vec::new());
        };
        if let Some(custom_event) =
            part.to_protocol_custom_event(crate::streaming::StreamPartNamespace::Anthropic)
        {
            return serialize_event(this, &custom_event);
        }

        if this.v3_unsupported_part_behavior == V3UnsupportedPartBehavior::AsText
            && let Some(text) = part.to_lossy_text()
        {
            let mut state = this.serialize_state.lock().map_err(|_| {
                LlmError::InternalError("serialize_state lock poisoned".to_string())
            })?;
            return serialize_lossy_text(&text, &mut state);
        }

        return Ok(Vec::new());
    }

    let mut state = this
        .serialize_state
        .lock()
        .map_err(|_| LlmError::InternalError("serialize_state lock poisoned".to_string()))?;

    match event {
        ChatStreamEvent::Part { .. } | ChatStreamEvent::PartWithReplay { .. } => {
            unreachable!("part events are normalized before serialize_state is locked")
        }
        ChatStreamEvent::Custom { event_type, data } => {
            if matches!(
                event_type.as_str(),
                "openai:tool-input-start"
                    | "openai:tool-input-delta"
                    | "openai:tool-input-end"
                    | "anthropic:tool-input-start"
                    | "anthropic:tool-input-delta"
                    | "anthropic:tool-input-end"
            ) && let Some(id) = data.get("id").and_then(|v| v.as_str())
            {
                if matches!(
                    event_type.as_str(),
                    "openai:tool-input-start" | "anthropic:tool-input-start"
                ) && data
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
                LanguageModelV3StreamPart::TextStart { .. }
                | LanguageModelV3StreamPart::TextDelta { .. }
                | LanguageModelV3StreamPart::TextEnd { .. } => {
                    if let Some(out) = serialize_text_part(&part, data, &mut state)? {
                        return Ok(out);
                    }
                    return Ok(out);
                }
                LanguageModelV3StreamPart::ReasoningStart { .. }
                | LanguageModelV3StreamPart::ReasoningDelta { .. }
                | LanguageModelV3StreamPart::ReasoningEnd { .. } => {
                    if let Some(out) = serialize_reasoning_part(&part, data, &mut state)? {
                        return Ok(out);
                    }
                    return Ok(out);
                }
                LanguageModelV3StreamPart::ToolInputStart { id, tool_name, .. } => {
                    let index = provider_content_block_index(data, &state);
                    return serialize_tool_input_part(&mut state, id, Some(tool_name), None, index);
                }
                LanguageModelV3StreamPart::ToolInputDelta { id, delta, .. } => {
                    let index = provider_content_block_index(data, &state);
                    return serialize_tool_input_part(&mut state, id, None, Some(delta), index);
                }
                LanguageModelV3StreamPart::ToolInputEnd { id, .. } => {
                    let index = provider_content_block_index(data, &state);
                    return serialize_tool_input_end_part(&mut state, id, index);
                }
                LanguageModelV3StreamPart::ToolCall(call) => {
                    if state.seen_tool_call_ids.contains(&call.tool_call_id) {
                        return Ok(Vec::new());
                    }

                    let index = provider_content_block_index(data, &state);
                    return serialize_tool_input_part(
                        &mut state,
                        &call.tool_call_id,
                        Some(&call.tool_name),
                        Some(&call.input),
                        index,
                    );
                }
                LanguageModelV3StreamPart::Finish {
                    usage: _,
                    finish_reason,
                    ..
                } => {
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

                    let stop_reason = raw_anthropic_stop_reason(finish_reason.raw.as_deref())
                        .or_else(|| map_v3_finish_reason_unified(&finish_reason.unified))
                        .map(|s| serde_json::Value::String(s.to_string()))
                        .unwrap_or(serde_json::Value::Null);

                    let usage_obj = usage_payload_from_finish_data(data);
                    let stop_sequence = data
                        .pointer("/providerMetadata/anthropic/stopSequence")
                        .and_then(|value| value.as_str())
                        .map(ToOwned::to_owned);
                    let container = data
                        .pointer("/providerMetadata/anthropic/container")
                        .and_then(
                        crate::standards::anthropic::utils::raw_container_from_provider_metadata,
                    );
                    let context_management = data
                        .pointer("/providerMetadata/anthropic/contextManagement")
                        .and_then(
                            crate::standards::anthropic::utils::raw_context_management_from_provider_metadata,
                        );

                    let msg_delta = build_message_delta_payload(
                        stop_reason,
                        stop_sequence,
                        usage_obj,
                        container,
                        context_management,
                    );
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
                return serialize_lossy_text(&text, &mut state);
            }

            Ok(Vec::new())
        }
        other => serialize_inner(other, &mut state),
    }
}
