use super::*;
use base64::Engine;

pub(super) fn serialize_event(
    this: &GeminiEventConverter,
    event: &ChatStreamEvent,
) -> Result<Vec<u8>, LlmError> {
    fn sse_data_frame(value: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
        let data = serde_json::to_vec(value).map_err(|e| {
            LlmError::JsonError(format!("Failed to serialize Gemini SSE JSON: {e}"))
        })?;
        let mut out = Vec::with_capacity(data.len() + 8);
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(&data);
        out.extend_from_slice(b"\n\n");
        Ok(out)
    }

    fn map_finish_reason(reason: &FinishReason) -> &'static str {
        match reason {
            FinishReason::Stop | FinishReason::StopSequence => "STOP",
            FinishReason::Length => "MAX_TOKENS",
            FinishReason::ContentFilter => "SAFETY",
            FinishReason::ToolCalls => "STOP",
            FinishReason::Error => "STOP",
            FinishReason::Unknown => "STOP",
            FinishReason::Other(_) => "STOP",
        }
    }

    fn finish_reason_payload(
        finish_reason: &FinishReason,
        raw_finish_reason: Option<&str>,
    ) -> String {
        raw_finish_reason
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| map_finish_reason(finish_reason).to_string())
    }

    fn serialize_error_payload(error: &serde_json::Value) -> serde_json::Value {
        match error {
            serde_json::Value::String(message) => serde_json::json!({
                "error": { "message": message }
            }),
            other => serde_json::json!({
                "error": other
            }),
        }
    }

    fn thought_signature_from_provider_metadata(
        provider_metadata: Option<&SharedV3ProviderMetadata>,
    ) -> Option<String> {
        let provider_metadata = provider_metadata?;

        for preferred in ["google", "vertex"] {
            if let Some(sig) = provider_metadata
                .get(preferred)
                .and_then(|value| value.get("thoughtSignature"))
                .and_then(|value| value.as_str())
            {
                return Some(sig.to_string());
            }
        }

        provider_metadata.values().find_map(|value| {
            value
                .get("thoughtSignature")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
        })
    }

    fn serialize_reasoning_chunk(
        delta: &str,
        provider_metadata: Option<&SharedV3ProviderMetadata>,
    ) -> Result<Vec<u8>, LlmError> {
        let mut part = serde_json::Map::new();
        part.insert(
            "text".to_string(),
            serde_json::Value::String(delta.to_string()),
        );
        part.insert("thought".to_string(), serde_json::Value::Bool(true));
        if let Some(sig) = thought_signature_from_provider_metadata(provider_metadata) {
            part.insert(
                "thoughtSignature".to_string(),
                serde_json::Value::String(sig),
            );
        }

        let payload = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [serde_json::Value::Object(part)]
                    }
                }
            ]
        });
        sse_data_frame(&payload)
    }

    fn serialize_text_chunk(
        delta: &str,
        provider_metadata: Option<&SharedV3ProviderMetadata>,
    ) -> Result<Vec<u8>, LlmError> {
        let mut part = serde_json::Map::new();
        part.insert(
            "text".to_string(),
            serde_json::Value::String(delta.to_string()),
        );
        if let Some(sig) = thought_signature_from_provider_metadata(provider_metadata) {
            part.insert(
                "thoughtSignature".to_string(),
                serde_json::Value::String(sig),
            );
        }

        let payload = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [serde_json::Value::Object(part)]
                    }
                }
            ]
        });
        sse_data_frame(&payload)
    }

    fn serialize_inline_data_chunk(
        file: &crate::types::ChatStreamFilePart,
        reasoning: bool,
    ) -> Result<Vec<u8>, LlmError> {
        let data = match &file.data {
            crate::types::ChatStreamFileData::Base64(data) => data.clone(),
            crate::types::ChatStreamFileData::Bytes(data) => {
                base64::engine::general_purpose::STANDARD.encode(data)
            }
        };

        let mut part = serde_json::Map::new();
        part.insert(
            "inlineData".to_string(),
            serde_json::json!({
                "mimeType": file.media_type,
                "data": data,
            }),
        );
        if reasoning {
            part.insert("thought".to_string(), serde_json::Value::Bool(true));
        }
        if let Some(sig) = thought_signature_from_provider_metadata(file.provider_metadata.as_ref())
        {
            part.insert(
                "thoughtSignature".to_string(),
                serde_json::Value::String(sig),
            );
        }

        let payload = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [serde_json::Value::Object(part)]
                    }
                }
            ]
        });
        sse_data_frame(&payload)
    }

    fn usage_metadata_payload(usage: &Usage) -> serde_json::Value {
        let normalized_input = usage.normalized_input_tokens();
        let normalized_output = usage.normalized_output_tokens();
        let prompt_tokens = normalized_input
            .total
            .or_else(|| usage.prompt_tokens_value());
        let output_tokens = normalized_output
            .total
            .or_else(|| usage.completion_tokens_value());
        let total_tokens = usage.total_tokens_value().or_else(|| {
            prompt_tokens
                .zip(output_tokens)
                .map(|(prompt, completion)| prompt.saturating_add(completion))
        });

        let mut usage_metadata = usage
            .raw_usage_value()
            .and_then(|value| serde_json::from_value::<GeminiUsageMetadata>(value).ok())
            .unwrap_or(GeminiUsageMetadata {
                prompt_token_count: None,
                cached_content_token_count: None,
                candidates_token_count: None,
                total_token_count: None,
                thoughts_token_count: None,
                traffic_type: None,
                prompt_tokens_details: None,
                candidates_tokens_details: None,
            });

        if let Some(prompt_tokens) = prompt_tokens {
            usage_metadata.prompt_token_count = Some(prompt_tokens);
        }
        if let Some(cached_tokens) = normalized_input.cache_read {
            usage_metadata.cached_content_token_count = Some(cached_tokens);
        }
        if let Some(text_tokens) = normalized_output.text {
            usage_metadata.candidates_token_count = Some(text_tokens);
        }
        if let Some(total_tokens) = total_tokens {
            usage_metadata.total_token_count = Some(total_tokens);
        }
        if let Some(reasoning_tokens) = normalized_output.reasoning {
            usage_metadata.thoughts_token_count = Some(reasoning_tokens);
        }

        serde_json::to_value(usage_metadata).unwrap_or(serde_json::json!({}))
    }

    fn flush_pending_reasoning_chunk(this: &GeminiEventConverter) -> Result<Vec<u8>, LlmError> {
        let pending = {
            let mut state = this.serialize_state.lock().map_err(|_| {
                LlmError::InternalError("serialize_state lock poisoned".to_string())
            })?;
            let pending = state.pending_reasoning_chunk.take();
            state.expect_reasoning_delta_custom_duplicate = false;
            pending
        };

        let Some(pending) = pending else {
            return Ok(Vec::new());
        };

        serialize_reasoning_chunk(&pending.delta, pending.provider_metadata.as_ref())
    }

    fn serialize_terminal_chunk(
        this: &GeminiEventConverter,
        finish_reason: &FinishReason,
        raw_finish_reason: Option<&str>,
        usage: Option<&Usage>,
    ) -> Result<Vec<u8>, LlmError> {
        let mut out = flush_pending_reasoning_chunk(this)?;
        let mut state = this
            .serialize_state
            .lock()
            .map_err(|_| LlmError::InternalError("serialize_state lock poisoned".to_string()))?;

        if state.terminal_emitted {
            return Ok(out);
        }
        state.terminal_emitted = true;

        for (call_id, call) in state.function_calls_by_id.iter_mut() {
            let Some(name) = call.name.clone() else {
                continue;
            };
            if call.arguments.trim().is_empty() {
                continue;
            }

            let parsed: Result<serde_json::Value, _> =
                crate::streaming::parse_json_with_repair(&call.arguments);
            let Ok(parsed) = parsed else {
                continue;
            };
            let Some(obj) = parsed.as_object() else {
                continue;
            };

            let args_json = match serde_json::to_string(obj) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if call
                .last_emitted_args_json
                .as_ref()
                .is_some_and(|v| v == &args_json)
            {
                continue;
            }
            call.last_emitted_args_json = Some(args_json);

            let payload = serde_json::json!({
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": name,
                                        "args": serde_json::Value::Object(obj.clone())
                                    }
                                }
                            ]
                        }
                    }
                ],
                "siumai": { "toolCallId": call_id }
            });
            if let Ok(frame) = sse_data_frame(&payload) {
                out.extend_from_slice(&frame);
            }
        }
        state.active_reasoning_provider_metadata = None;
        state.pending_reasoning_chunk = None;

        let payload = serde_json::json!({
            "candidates": [
                { "finishReason": finish_reason_payload(finish_reason, raw_finish_reason) }
            ],
            "usageMetadata": usage
                .map(usage_metadata_payload)
                .unwrap_or(serde_json::Value::Null),
        });
        out.extend_from_slice(&sse_data_frame(&payload)?);
        Ok(out)
    }

    match event {
        // Gemini streaming does not have an explicit "start" frame; the first chunk carries data.
        ChatStreamEvent::StreamStart { .. } => flush_pending_reasoning_chunk(this),
        ChatStreamEvent::ContentDelta { delta, .. } => {
            let mut out = flush_pending_reasoning_chunk(this)?;
            out.extend_from_slice(&serialize_text_chunk(delta, None)?);
            Ok(out)
        }
        ChatStreamEvent::ThinkingDelta { delta } => {
            let (pending, active_provider_metadata) = {
                let mut state = this.serialize_state.lock().map_err(|_| {
                    LlmError::InternalError("serialize_state lock poisoned".to_string())
                })?;
                state.expect_reasoning_delta_custom_duplicate = false;
                (
                    state.pending_reasoning_chunk.take(),
                    state.active_reasoning_provider_metadata.clone(),
                )
            };

            if let Some(pending) = pending {
                if pending.delta == *delta {
                    return serialize_reasoning_chunk(
                        &pending.delta,
                        pending.provider_metadata.as_ref(),
                    );
                }

                let mut out =
                    serialize_reasoning_chunk(&pending.delta, pending.provider_metadata.as_ref())?;
                out.extend_from_slice(&serialize_reasoning_chunk(
                    delta,
                    active_provider_metadata.as_ref(),
                )?);
                return Ok(out);
            }

            serialize_reasoning_chunk(delta, active_provider_metadata.as_ref())
        }
        ChatStreamEvent::UsageUpdate { usage } => {
            let mut out = flush_pending_reasoning_chunk(this)?;
            let payload = serde_json::json!({
                "usageMetadata": usage_metadata_payload(usage)
            });
            out.extend_from_slice(&sse_data_frame(&payload)?);
            Ok(out)
        }
        ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            ..
        } => {
            let mut out = flush_pending_reasoning_chunk(this)?;
            let mut state = this.serialize_state.lock().map_err(|_| {
                LlmError::InternalError("serialize_state lock poisoned".to_string())
            })?;

            let call = state.function_calls_by_id.entry(id.clone()).or_default();

            if let Some(name) = function_name.clone()
                && !name.trim().is_empty()
            {
                call.name = Some(name);
            }

            if let Some(delta) = arguments_delta.clone() {
                call.arguments.push_str(&delta);
            }

            let Some(name) = call.name.clone() else {
                return Ok(Vec::new());
            };

            if call.arguments.trim().is_empty() {
                return Ok(Vec::new());
            }

            let parsed: serde_json::Value =
                crate::streaming::parse_json_with_repair(&call.arguments).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Gemini tool call arguments as JSON object: {e}"
                    ))
                })?;

            let Some(obj) = parsed.as_object() else {
                return Ok(Vec::new());
            };

            let args_json = serde_json::to_string(obj).map_err(|e| {
                LlmError::ParseError(format!(
                    "Failed to serialize Gemini tool call args JSON object: {e}"
                ))
            })?;

            if call
                .last_emitted_args_json
                .as_ref()
                .is_some_and(|v| v == &args_json)
            {
                return Ok(Vec::new());
            }
            call.last_emitted_args_json = Some(args_json);

            let payload = serde_json::json!({
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": name,
                                        "args": serde_json::Value::Object(obj.clone())
                                    }
                                }
                            ]
                        }
                    }
                ]
            });
            out.extend_from_slice(&sse_data_frame(&payload)?);
            Ok(out)
        }
        ChatStreamEvent::StreamEnd { response } => {
            let reason = response
                .finish_reason
                .as_ref()
                .unwrap_or(&FinishReason::Stop);
            serialize_terminal_chunk(
                this,
                reason,
                response.raw_finish_reason.as_deref(),
                response.usage.as_ref(),
            )
        }
        ChatStreamEvent::Error { error } => {
            let mut out = flush_pending_reasoning_chunk(this)?;
            let payload = serialize_error_payload(&serde_json::Value::String(error.clone()));
            out.extend_from_slice(&sse_data_frame(&payload)?);
            Ok(out)
        }
        ChatStreamEvent::Part { part } | ChatStreamEvent::PartWithReplay { part, .. } => {
            match &part {
                ChatStreamPart::TextDelta {
                    delta,
                    provider_metadata,
                    ..
                } => {
                    let mut out = flush_pending_reasoning_chunk(this)?;
                    out.extend_from_slice(&serialize_text_chunk(
                        delta,
                        provider_metadata.as_ref(),
                    )?);
                    return Ok(out);
                }
                ChatStreamPart::File(file) => {
                    let mut out = flush_pending_reasoning_chunk(this)?;
                    out.extend_from_slice(&serialize_inline_data_chunk(file, false)?);
                    return Ok(out);
                }
                ChatStreamPart::ReasoningFile(file) => {
                    let mut out = flush_pending_reasoning_chunk(this)?;
                    out.extend_from_slice(&serialize_inline_data_chunk(file, true)?);
                    return Ok(out);
                }
                ChatStreamPart::Finish {
                    usage,
                    finish_reason,
                    ..
                } => {
                    return serialize_terminal_chunk(
                        this,
                        &finish_reason.unified,
                        finish_reason.raw.as_deref(),
                        Some(usage),
                    );
                }
                ChatStreamPart::Error { error } => {
                    let mut out = flush_pending_reasoning_chunk(this)?;
                    let payload = serialize_error_payload(error);
                    out.extend_from_slice(&sse_data_frame(&payload)?);
                    return Ok(out);
                }
                ChatStreamPart::StreamStart { .. }
                | ChatStreamPart::TextStart { .. }
                | ChatStreamPart::TextEnd { .. }
                | ChatStreamPart::ResponseMetadata(_) => {
                    return flush_pending_reasoning_chunk(this);
                }
                _ => {}
            }

            let part = LanguageModelV3StreamPart::from_runtime_part(part.clone());
            if matches!(part, LanguageModelV3StreamPart::ReasoningDelta { .. }) {
                let mut state = this.serialize_state.lock().map_err(|_| {
                    LlmError::InternalError("serialize_state lock poisoned".to_string())
                })?;
                state.expect_reasoning_delta_custom_duplicate = true;
            }
            let Some(custom_event) =
                part.to_protocol_custom_event(crate::streaming::StreamPartNamespace::Gemini)
            else {
                if this.v3_unsupported_part_behavior == V3UnsupportedPartBehavior::AsText
                    && let Some(text) = part.to_lossy_text()
                {
                    return serialize_event(
                        this,
                        &ChatStreamEvent::ContentDelta {
                            delta: text,
                            index: None,
                        },
                    );
                }
                return Ok(Vec::new());
            };
            serialize_event(this, &custom_event)
        }
        ChatStreamEvent::Custom { data, .. } => {
            let Some(part) = LanguageModelV3StreamPart::parse_loose_json(data) else {
                return Ok(Vec::new());
            };

            match part {
                LanguageModelV3StreamPart::ReasoningStart {
                    provider_metadata, ..
                } => {
                    let out = flush_pending_reasoning_chunk(this)?;
                    let mut state = this.serialize_state.lock().map_err(|_| {
                        LlmError::InternalError("serialize_state lock poisoned".to_string())
                    })?;
                    state.active_reasoning_provider_metadata = provider_metadata;
                    Ok(out)
                }
                LanguageModelV3StreamPart::ReasoningDelta {
                    delta,
                    provider_metadata,
                    ..
                } => {
                    let (previous, ignore_duplicate) = {
                        let mut state = this.serialize_state.lock().map_err(|_| {
                            LlmError::InternalError("serialize_state lock poisoned".to_string())
                        })?;

                        let carry_provider_metadata = provider_metadata
                            .clone()
                            .or_else(|| state.active_reasoning_provider_metadata.clone());

                        if state.expect_reasoning_delta_custom_duplicate
                            && state
                                .pending_reasoning_chunk
                                .as_ref()
                                .is_some_and(|pending| {
                                    pending.delta == delta
                                        && pending.provider_metadata == carry_provider_metadata
                                })
                        {
                            state.expect_reasoning_delta_custom_duplicate = false;
                            (None, true)
                        } else {
                            if let Some(provider_metadata) = provider_metadata {
                                state.active_reasoning_provider_metadata = Some(provider_metadata);
                            }

                            (
                                state.pending_reasoning_chunk.replace(
                                    GeminiPendingReasoningSerializeState {
                                        delta,
                                        provider_metadata: carry_provider_metadata,
                                    },
                                ),
                                false,
                            )
                        }
                    };

                    if ignore_duplicate {
                        return Ok(Vec::new());
                    }

                    let Some(previous) = previous else {
                        return Ok(Vec::new());
                    };

                    serialize_reasoning_chunk(&previous.delta, previous.provider_metadata.as_ref())
                }
                LanguageModelV3StreamPart::ReasoningEnd { .. } => {
                    let out = flush_pending_reasoning_chunk(this)?;
                    let mut state = this.serialize_state.lock().map_err(|_| {
                        LlmError::InternalError("serialize_state lock poisoned".to_string())
                    })?;
                    state.active_reasoning_provider_metadata = None;
                    Ok(out)
                }
                LanguageModelV3StreamPart::Source(source) => {
                    let mut out = flush_pending_reasoning_chunk(this)?;
                    let chunk = match source {
                        LanguageModelV3Source::Url { url, title, .. } => {
                            let mut web = serde_json::Map::new();
                            web.insert("uri".to_string(), serde_json::Value::String(url));
                            if let Some(title) = title {
                                web.insert("title".to_string(), serde_json::Value::String(title));
                            }
                            serde_json::json!({ "web": serde_json::Value::Object(web) })
                        }
                        LanguageModelV3Source::Document {
                            title,
                            filename,
                            media_type: _,
                            ..
                        } => {
                            let mut retrieved = serde_json::Map::new();
                            retrieved.insert(
                                "title".to_string(),
                                serde_json::Value::String(match filename {
                                    Some(f) => format!("{title} ({f})"),
                                    None => title,
                                }),
                            );
                            serde_json::json!({
                                "retrievedContext": serde_json::Value::Object(retrieved)
                            })
                        }
                    };

                    let payload = serde_json::json!({
                        "candidates": [
                            {
                                "groundingMetadata": {
                                    "groundingChunks": [chunk]
                                }
                            }
                        ]
                    });
                    out.extend_from_slice(&sse_data_frame(&payload)?);
                    Ok(out)
                }
                LanguageModelV3StreamPart::Finish {
                    usage,
                    finish_reason,
                    ..
                } => {
                    let mut out = flush_pending_reasoning_chunk(this)?;
                    let reason = finish_reason
                        .raw
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                        .unwrap_or_else(|| {
                            let unified = finish_reason.unified.to_ascii_lowercase();
                            if unified.contains("length") || unified.contains("max") {
                                "MAX_TOKENS".to_string()
                            } else if unified.contains("safety") || unified.contains("content") {
                                "SAFETY".to_string()
                            } else {
                                "STOP".to_string()
                            }
                        });

                    let prompt = usage.input_tokens.total.unwrap_or(0).min(u32::MAX as u64) as u32;
                    let completion =
                        usage.output_tokens.total.unwrap_or(0).min(u32::MAX as u64) as u32;
                    let total = prompt.saturating_add(completion);

                    let payload = serde_json::json!({
                        "candidates": [
                            { "finishReason": reason }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": prompt,
                            "candidatesTokenCount": completion,
                            "totalTokenCount": total,
                            "thoughtsTokenCount": usage.output_tokens.reasoning.map(|v| v.min(u32::MAX as u64) as u32),
                        }
                    });
                    out.extend_from_slice(&sse_data_frame(&payload)?);
                    Ok(out)
                }
                LanguageModelV3StreamPart::ToolCall(call) => {
                    let mut out = flush_pending_reasoning_chunk(this)?;
                    let parsed: serde_json::Value =
                        crate::streaming::parse_json_with_repair(&call.input).map_err(|e| {
                            LlmError::ParseError(format!(
                                "Failed to parse Gemini tool call input JSON: {e}"
                            ))
                        })?;

                    let mut part = serde_json::Map::new();
                    if call.provider_executed.unwrap_or(false) && call.tool_name == "code_execution"
                    {
                        let language = parsed
                            .get("language")
                            .and_then(|value| value.as_str())
                            .unwrap_or("PYTHON");
                        let code = parsed
                            .get("code")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default();

                        part.insert(
                            "executableCode".to_string(),
                            serde_json::json!({
                                "language": language,
                                "code": code,
                            }),
                        );
                    } else {
                        let Some(obj) = parsed.as_object() else {
                            return Ok(out);
                        };

                        part.insert(
                            "functionCall".to_string(),
                            serde_json::json!({
                                "name": call.tool_name,
                                "args": serde_json::Value::Object(obj.clone()),
                            }),
                        );
                    }

                    if let Some(sig) =
                        thought_signature_from_provider_metadata(call.provider_metadata.as_ref())
                    {
                        part.insert(
                            "thoughtSignature".to_string(),
                            serde_json::Value::String(sig),
                        );
                    }

                    let payload = serde_json::json!({
                        "candidates": [
                            {
                                "content": {
                                    "parts": [serde_json::Value::Object(part)]
                                }
                            }
                        ]
                    });
                    out.extend_from_slice(&sse_data_frame(&payload)?);
                    Ok(out)
                }
                LanguageModelV3StreamPart::ToolResult(tr) => {
                    let mut out = flush_pending_reasoning_chunk(this)?;
                    if tr.tool_name == "code_execution" {
                        let outcome = tr
                            .result
                            .get("outcome")
                            .and_then(|v| v.as_str())
                            .unwrap_or("OUTCOME_OK");
                        let output = tr.result.get("output").and_then(|v| v.as_str());

                        let mut res = serde_json::Map::new();
                        res.insert(
                            "outcome".to_string(),
                            serde_json::Value::String(outcome.to_string()),
                        );
                        if let Some(out) = output {
                            res.insert(
                                "output".to_string(),
                                serde_json::Value::String(out.to_string()),
                            );
                        }

                        let mut part = serde_json::Map::new();
                        part.insert(
                            "codeExecutionResult".to_string(),
                            serde_json::Value::Object(res),
                        );
                        if let Some(sig) =
                            thought_signature_from_provider_metadata(tr.provider_metadata.as_ref())
                        {
                            part.insert(
                                "thoughtSignature".to_string(),
                                serde_json::Value::String(sig),
                            );
                        }

                        let payload = serde_json::json!({
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [serde_json::Value::Object(part)]
                                    }
                                }
                            ]
                        });
                        out.extend_from_slice(&sse_data_frame(&payload)?);
                        return Ok(out);
                    }

                    if this.emit_function_response_tool_results {
                        let mut part = serde_json::Map::new();
                        part.insert(
                            "functionResponse".to_string(),
                            serde_json::json!({
                                "name": tr.tool_name,
                                "response": tr.result
                            }),
                        );
                        if let Some(sig) =
                            thought_signature_from_provider_metadata(tr.provider_metadata.as_ref())
                        {
                            part.insert(
                                "thoughtSignature".to_string(),
                                serde_json::Value::String(sig),
                            );
                        }

                        let mut payload = serde_json::Map::new();
                        payload.insert(
                            "candidates".to_string(),
                            serde_json::json!([{
                                "content": { "parts": [serde_json::Value::Object(part)] }
                            }]),
                        );
                        payload.insert(
                            "siumai".to_string(),
                            serde_json::json!({ "toolCallId": tr.tool_call_id }),
                        );

                        out.extend_from_slice(&sse_data_frame(&serde_json::Value::Object(
                            payload,
                        ))?);
                        return Ok(out);
                    }

                    if this.v3_unsupported_part_behavior == V3UnsupportedPartBehavior::AsText
                        && let Some(text) =
                            LanguageModelV3StreamPart::ToolResult(tr).to_lossy_text()
                    {
                        return this.serialize_event(&ChatStreamEvent::ContentDelta {
                            delta: text,
                            index: None,
                        });
                    }

                    Ok(out)
                }
                other => {
                    let mut out = flush_pending_reasoning_chunk(this)?;
                    for ev in other.to_best_effort_chat_events() {
                        out.extend_from_slice(&this.serialize_event(&ev)?);
                    }

                    if out.is_empty()
                        && this.v3_unsupported_part_behavior == V3UnsupportedPartBehavior::AsText
                        && let Some(text) = other.to_lossy_text()
                    {
                        out.extend_from_slice(&this.serialize_event(
                            &ChatStreamEvent::ContentDelta {
                                delta: text,
                                index: None,
                            },
                        )?);
                    }

                    Ok(out)
                }
            }
        }
    }
}
