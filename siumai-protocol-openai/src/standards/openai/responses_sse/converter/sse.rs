use super::OpenAiResponsesEventConverter;

impl crate::streaming::SseEventConverter for OpenAiResponsesEventConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>>,
                > + Send
                + Sync
                + '_,
        >,
    > {
        Box::pin(async move {
            let data_raw = event.data.trim();
            if data_raw.is_empty() {
                return vec![];
            }
            // Consider explicit completed events
            let event_name = event.event.as_str();

            if data_raw == "[DONE]" {
                // [DONE] events should not generate any events in the new architecture
                return vec![];
            }

            // Parse JSON once; most Responses API SSE chunks use `data: {...}` with a `type` field.
            let json = match serde_json::from_str::<serde_json::Value>(data_raw) {
                Ok(v) => v,
                Err(e) => {
                    return vec![Err(crate::error::LlmError::ParseError(format!(
                        "Failed to parse SSE JSON: {e}"
                    )))];
                }
            };

            self.update_provider_tool_names(&json);

            // Some SSE clients (and our JSON fallback wrapper) default to `event: message`.
            // Treat that as "no explicit event name" and rely on the JSON `type` field instead.
            let chunk_type = if !event_name.is_empty() && event_name != "message" {
                event_name
            } else {
                json.get("type").and_then(|t| t.as_str()).unwrap_or("")
            };

            if chunk_type == "response.created" {
                // A new response in the same SSE connection means any previously buffered
                // StreamEnd is not terminal for the overall stream.
                self.clear_pending_stream_end_events();

                if let Some(resp) = json.get("response") {
                    let response_id = resp.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let model_id = resp.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    let created_at = resp.get("created_at").and_then(|v| v.as_i64()).unwrap_or(0);
                    self.record_created_response_metadata(response_id, model_id, created_at);
                }

                let mut out: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

                if self.mark_stream_start_emitted() {
                    out.push(crate::streaming::ChatStreamEvent::Part {
                        part: crate::types::ChatStreamPart::StreamStart { warnings: vec![] },
                    });
                }

                if let (Some(id), Some(model_id), Some(created)) = (
                    self.created_response_id(),
                    self.created_model_id(),
                    self.created_timestamp(),
                ) && self.mark_response_metadata_emitted(&id)
                {
                    out.push(crate::streaming::ChatStreamEvent::Part {
                        part: crate::types::ChatStreamPart::ResponseMetadata(
                            crate::types::ResponseMetadata {
                                id: Some(id),
                                model: Some(model_id),
                                created: Some(created),
                                provider: self.provider_metadata_key.clone(),
                                request_id: None,
                                headers: None,
                                body: None,
                            },
                        ),
                    });
                }

                return out.into_iter().map(Ok).collect();
            }

            if chunk_type == "response.failed" {
                // Vercel alignment: failed responses should not error out the converter;
                // instead they emit an error part (from the preceding `error` chunk) and
                // then a finish part with `unified: "other"` and `providerMetadata.responseId`.
                //
                // The OpenAI Responses API does not provide stable usage totals on failures.
                // Mirror the Vercel stream shape by emitting `null` token fields.
                let extra_events = self.convert_mcp_items_from_completed(&json);

                let response_id = self.created_response_id().or_else(|| {
                    json.get("response")?
                        .get("id")?
                        .as_str()
                        .map(|s| s.to_string())
                });

                let mut provider_metadata = serde_json::Map::new();
                if let Some(id) = response_id {
                    provider_metadata
                        .insert("responseId".to_string(), serde_json::Value::String(id));
                }

                let finish_evt = crate::streaming::LanguageModelV3StreamPart::Finish {
                    usage: crate::streaming::LanguageModelV3Usage {
                        input_tokens: crate::streaming::LanguageModelV3InputTokens {
                            total: None,
                            no_cache: None,
                            cache_read: None,
                            cache_write: None,
                        },
                        output_tokens: crate::streaming::LanguageModelV3OutputTokens {
                            total: None,
                            text: None,
                            reasoning: None,
                        },
                        raw: None,
                    },
                    finish_reason: crate::streaming::LanguageModelV3FinishReason {
                        unified: "other".to_string(),
                        raw: None,
                    },
                    provider_metadata: if provider_metadata.is_empty() {
                        None
                    } else {
                        Some(std::collections::HashMap::from([(
                            self.provider_metadata_key.clone(),
                            serde_json::Value::Object(provider_metadata),
                        )]))
                    },
                }
                .to_part_event();

                self.replace_pending_stream_end_events(vec![finish_evt]);

                return extra_events.into_iter().map(Ok).collect();
            }

            if chunk_type == "response.completed" || chunk_type == "response.incomplete" {
                let extra_events = self.convert_mcp_items_from_completed(&json);

                // The completed event often contains the full response payload.
                // Delegate to centralized ResponseTransformer for final ChatResponse.
                let resp_tx =
                    crate::standards::openai::transformers::OpenAiResponsesResponseTransformer::new()
                    .with_style(self.responses_transform_style)
                    .with_provider_metadata_key(self.provider_metadata_key.clone());
                match crate::execution::transformers::response::ResponseTransformer::transform_chat_response(
                    &resp_tx, &json,
                ) {
                    Ok(response) => {
                        // Buffer the final finish + StreamEnd until the stream ends.
                        // OpenAI Responses can emit multiple `response.created` / `response.completed`
                        // pairs on a single SSE connection (e.g., built-in tools), and only the last
                        // completed response should terminate the stream.
                        let mut pending: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
                        if let Some(finish_evt) = self.convert_finish_event(&json, &response) {
                            pending.push(finish_evt);
                        }
                        pending.push(crate::streaming::ChatStreamEvent::StreamEnd { response });
                        self.replace_pending_stream_end_events(pending);

                        return extra_events.into_iter().map(Ok).collect();
                    }
                    Err(err) => return vec![Err(err)],
                }
            }

            // Route by event name first
            match chunk_type {
                "response.output_text.delta" => {
                    let mut out: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

                    if let Some(mut extra) = self.convert_output_text_delta_events(&json) {
                        out.append(&mut extra);
                    }
                    if out.is_empty()
                        && let Some(mut events) = self.convert_responses_event(&json)
                    {
                        out.append(&mut events);
                    }

                    return out.into_iter().map(Ok).collect();
                }
                "response.tool_call.delta" | "response.function_call.delta" | "response.usage" => {
                    if let Some(events) = self.convert_responses_event(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.output_text.annotation.added" => {
                    if let Some(evt) = self.convert_output_text_annotation_added(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.error" | "error" => {
                    // Vercel alignment: some fixtures use a legacy top-level `"type": "error"` chunk,
                    // while newer streams use `"type": "response.error"`.
                    //
                    // Emit both:
                    // - a structured custom error part (matches Vercel stream parts)
                    // - a standard ChatStreamEvent::Error for generic consumers
                    let msg = json
                        .get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("Unknown error")
                        .to_string();

                    return vec![
                        Ok(crate::streaming::ChatStreamEvent::Part {
                            part: crate::types::ChatStreamPart::Error {
                                error: json.get("error").cloned().unwrap_or_else(|| json.clone()),
                            },
                        }),
                        Ok(crate::streaming::ChatStreamEvent::Error { error: msg }),
                    ];
                }
                "response.function_call_arguments.delta" => {
                    if let Some(evt) = self.convert_function_call_arguments_delta_tool_input(&json)
                    {
                        return vec![Ok(evt)];
                    }
                }
                "response.function_call_arguments.done" => {
                    if let Some(events) =
                        self.convert_function_call_arguments_done_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.apply_patch_call_operation_diff.delta" => {
                    if let Some(evt) = self.convert_apply_patch_operation_diff_delta(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.apply_patch_call_operation_diff.done" => {
                    if let Some(events) = self.convert_apply_patch_operation_diff_done(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.image_generation_call.partial_image" => {
                    if let Some(evt) = self.convert_image_generation_partial_image(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.code_interpreter_call_code.delta" => {
                    if let Some(events) = self.convert_code_interpreter_code_delta_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.code_interpreter_call_code.done" => {
                    if let Some(events) = self.convert_code_interpreter_code_done_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.mcp_call_arguments.delta" => {
                    if let Some(item_id) = json.get("item_id").and_then(|v| v.as_str())
                        && let Some(delta) = json.get("delta").and_then(|v| v.as_str())
                    {
                        self.record_mcp_call_args(item_id, delta);
                    }
                }
                "response.mcp_call_arguments.done" => {
                    if let Some(evt) = self.convert_mcp_call_arguments_done(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.custom_tool_call_input.delta" => {
                    if let Some(events) = self.convert_custom_tool_call_input_delta(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.custom_tool_call_input.done" => {
                    if let Some(events) = self.convert_custom_tool_call_input_done(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.output_item.added" => {
                    if let Some(evt) = self.convert_message_output_item_added(&json) {
                        return vec![Ok(evt)];
                    }
                    if let Some(evt) = self.convert_reasoning_output_item_added(&json) {
                        return vec![Ok(evt)];
                    }
                    if let Some(events) =
                        self.convert_apply_patch_output_item_added_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                    if let Some(events) =
                        self.convert_code_interpreter_output_item_added_tool_input(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                    if let Some(events) = self.convert_provider_tool_output_item_added(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }

                    if let Some(evt) =
                        self.convert_function_call_output_item_added_tool_input(&json)
                    {
                        return vec![Ok(evt)];
                    }
                }
                "response.output_item.done" => {
                    if let Some(events) = self.convert_reasoning_output_item_done(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                    if let Some(events) = self.convert_provider_tool_output_item_done(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                    if let Some(evt) = self.convert_message_output_item_done(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.reasoning_summary_part.added" => {
                    if let Some(events) = self.convert_reasoning_summary_part_added(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
                "response.reasoning_summary_text.delta" => {
                    if let Some(evt) = self.convert_reasoning_summary_text_delta(&json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.reasoning_summary_part.done" => {
                    if let Some(evt) = self.convert_reasoning_summary_part_done(&json) {
                        return vec![Ok(evt)];
                    }
                }
                _ => {
                    if let Some(events) = self.convert_responses_event(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
                    }
                }
            }

            vec![]
        })
    }

    fn handle_stream_end(
        &self,
    ) -> Option<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>> {
        self.pop_pending_stream_end_event().map(Ok)
    }

    fn handle_stream_end_events(
        &self,
    ) -> Vec<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>> {
        let Ok(mut q) = self.pending_stream_end_events.lock() else {
            return Vec::new();
        };
        q.drain(..).map(Ok).collect()
    }

    fn finalize_on_disconnect(&self) -> bool {
        true
    }

    fn serialize_event(
        &self,
        event: &crate::streaming::ChatStreamEvent,
    ) -> Result<Vec<u8>, crate::error::LlmError> {
        self.serialize_event_impl(event)
    }
}
