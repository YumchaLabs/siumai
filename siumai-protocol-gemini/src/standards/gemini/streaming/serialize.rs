use super::*;

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

    match event {
        // Gemini streaming does not have an explicit "start" frame; the first chunk carries data.
        ChatStreamEvent::StreamStart { .. } => Ok(Vec::new()),
        ChatStreamEvent::ContentDelta { delta, .. } => {
            let payload = serde_json::json!({
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                { "text": delta }
                            ]
                        }
                    }
                ]
            });
            sse_data_frame(&payload)
        }
        ChatStreamEvent::ThinkingDelta { delta } => {
            // Gemini thinking chunks use `thought: true` on the part.
            let payload = serde_json::json!({
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                { "text": delta, "thought": true }
                            ]
                        }
                    }
                ]
            });
            sse_data_frame(&payload)
        }
        ChatStreamEvent::UsageUpdate { usage } => {
            let thoughts = usage
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens)
                .or({
                    #[allow(deprecated)]
                    {
                        usage.reasoning_tokens
                    }
                });

            let payload = serde_json::json!({
                "usageMetadata": {
                    "promptTokenCount": usage.prompt_tokens,
                    "candidatesTokenCount": usage.completion_tokens,
                    "totalTokenCount": usage.total_tokens,
                    "thoughtsTokenCount": thoughts,
                }
            });
            sse_data_frame(&payload)
        }
        ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            ..
        } => {
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
            sse_data_frame(&payload)
        }
        ChatStreamEvent::StreamEnd { response } => {
            let reason = response
                .finish_reason
                .as_ref()
                .map(map_finish_reason)
                .unwrap_or("STOP");

            let thoughts = response
                .usage
                .as_ref()
                .and_then(|u| u.completion_tokens_details.as_ref())
                .and_then(|d| d.reasoning_tokens)
                .or_else(|| {
                    #[allow(deprecated)]
                    {
                        response.usage.as_ref().and_then(|u| u.reasoning_tokens)
                    }
                });

            let usage = response.usage.as_ref().map(|u| {
                serde_json::json!({
                    "promptTokenCount": u.prompt_tokens,
                    "candidatesTokenCount": u.completion_tokens,
                    "totalTokenCount": u.total_tokens,
                    "thoughtsTokenCount": thoughts,
                })
            });

            let mut out = Vec::new();

            // Flush pending tool calls (best-effort) before finish chunk.
            if let Ok(mut state) = this.serialize_state.lock() {
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
            }

            let payload = serde_json::json!({
                "candidates": [
                    { "finishReason": reason }
                ],
                "usageMetadata": usage.unwrap_or(serde_json::Value::Null),
            });
            out.extend_from_slice(&sse_data_frame(&payload)?);
            Ok(out)
        }
        ChatStreamEvent::Error { error } => {
            // Gemini SSE errors do not have a stable in-band frame; emit a best-effort JSON payload.
            let payload = serde_json::json!({
                "error": { "message": error }
            });
            sse_data_frame(&payload)
        }
        ChatStreamEvent::Custom { data, .. } => {
            let Some(part) = LanguageModelV3StreamPart::parse_loose_json(data) else {
                return Ok(Vec::new());
            };

            match part {
                LanguageModelV3StreamPart::Source(source) => {
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
                    sse_data_frame(&payload)
                }
                LanguageModelV3StreamPart::Finish {
                    usage,
                    finish_reason,
                    ..
                } => {
                    let unified = finish_reason.unified.to_ascii_lowercase();
                    let reason = if unified.contains("length") || unified.contains("max") {
                        "MAX_TOKENS"
                    } else if unified.contains("safety") || unified.contains("content") {
                        "SAFETY"
                    } else {
                        "STOP"
                    };

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
                    sse_data_frame(&payload)
                }
                LanguageModelV3StreamPart::ToolResult(tr) => {
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

                        let payload = serde_json::json!({
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            { "codeExecutionResult": serde_json::Value::Object(res) }
                                        ]
                                    }
                                }
                            ]
                        });
                        return sse_data_frame(&payload);
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

                        return sse_data_frame(&serde_json::Value::Object(payload));
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

                    Ok(Vec::new())
                }
                other => {
                    let mut out = Vec::new();
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
