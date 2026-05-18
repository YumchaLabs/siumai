use super::*;

impl OpenAiResponsesEventConverter {
    fn part_provider_metadata(
        &self,
        value: serde_json::Value,
    ) -> Option<crate::types::StreamProviderMetadata> {
        let serde_json::Value::Object(obj) = value else {
            return None;
        };

        let mut provider_metadata = crate::types::StreamProviderMetadata::new();
        provider_metadata.insert(
            self.provider_metadata_key.clone(),
            serde_json::Value::Object(obj),
        );
        Some(provider_metadata)
    }

    fn style_part_provider_metadata(
        &self,
        value: serde_json::Value,
    ) -> Option<crate::types::StreamProviderMetadata> {
        match self.stream_parts_style {
            StreamPartsStyle::OpenAi => self.part_provider_metadata(value),
            StreamPartsStyle::Xai => None,
        }
    }

    fn reasoning_part_provider_metadata(
        &self,
        value: serde_json::Value,
    ) -> Option<crate::types::StreamProviderMetadata> {
        self.part_provider_metadata(value)
    }

    fn xai_file_search_queries_from_object(
        item: &serde_json::Map<String, serde_json::Value>,
    ) -> serde_json::Value {
        item.get("queries")
            .filter(|value| !value.is_null())
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Array(Vec::new()))
    }

    fn escape_json_delta(delta: &str) -> String {
        serde_json::to_string(delta)
            .ok()
            .and_then(|value| {
                value
                    .strip_prefix('"')
                    .and_then(|value| value.strip_suffix('"'))
                    .map(str::to_string)
            })
            .unwrap_or_else(|| delta.to_string())
    }

    fn file_search_results_from_object(
        item: &serde_json::Map<String, serde_json::Value>,
    ) -> serde_json::Value {
        let Some(results) = item.get("results") else {
            return serde_json::Value::Null;
        };
        let Some(results) = results.as_array() else {
            return results.clone();
        };

        serde_json::Value::Array(
            results
                .iter()
                .map(|result| {
                    let mut out = serde_json::Map::new();
                    if let Some(file_id) = result.get("file_id").or_else(|| result.get("fileId"))
                        && !file_id.is_null()
                    {
                        out.insert("fileId".to_string(), file_id.clone());
                    }
                    if let Some(filename) = result.get("filename")
                        && !filename.is_null()
                    {
                        out.insert("filename".to_string(), filename.clone());
                    }
                    if let Some(attributes) = result.get("attributes")
                        && !attributes.is_null()
                    {
                        out.insert("attributes".to_string(), attributes.clone());
                    }
                    if let Some(score) = result.get("score")
                        && !score.is_null()
                    {
                        out.insert("score".to_string(), score.clone());
                    }
                    if let Some(text) = result.get("text")
                        && !text.is_null()
                    {
                        out.insert("text".to_string(), text.clone());
                    }
                    serde_json::Value::Object(out)
                })
                .collect(),
        )
    }

    fn web_search_result_from_object(
        item: &serde_json::Map<String, serde_json::Value>,
    ) -> serde_json::Value {
        let action = item.get("action").unwrap_or(&serde_json::Value::Null);
        if action.is_null() && item.get("results").is_none() {
            return serde_json::json!({});
        }

        let action_type_raw = action
            .get("type")
            .and_then(|value| value.as_str())
            .unwrap_or("search");
        let action_type = match action_type_raw {
            "open_page" => "openPage",
            "find_in_page" => "findInPage",
            other => other,
        };

        let mut action_obj = serde_json::Map::new();
        action_obj.insert(
            "type".to_string(),
            serde_json::Value::String(action_type.to_string()),
        );

        if action_type_raw == "search"
            && let Some(query) = action.get("query").or_else(|| item.get("query"))
            && !query.is_null()
        {
            action_obj.insert("query".to_string(), query.clone());
        }

        if matches!(action_type_raw, "open_page" | "find_in_page")
            && let Some(url) = action.get("url").or_else(|| item.get("url"))
            && !url.is_null()
        {
            action_obj.insert("url".to_string(), url.clone());
        }

        if action_type_raw == "find_in_page"
            && let Some(pattern) = action.get("pattern")
            && !pattern.is_null()
        {
            action_obj.insert("pattern".to_string(), pattern.clone());
        }

        let mut result = serde_json::Map::new();
        result.insert("action".to_string(), serde_json::Value::Object(action_obj));

        if let Some(sources) = action.get("sources").and_then(|value| value.as_array())
            && !sources.is_empty()
        {
            result.insert(
                "sources".to_string(),
                serde_json::Value::Array(sources.to_vec()),
            );
        }

        if !result.contains_key("sources")
            && let Some(results) = item.get("results").and_then(|value| value.as_array())
        {
            let sources = results
                .iter()
                .filter_map(|entry| {
                    let url = entry.get("url").and_then(|value| value.as_str())?;
                    Some(serde_json::json!({ "type": "url", "url": url }))
                })
                .collect::<Vec<_>>();
            if !sources.is_empty() {
                result.insert("sources".to_string(), serde_json::Value::Array(sources));
            }
        }

        serde_json::Value::Object(result)
    }

    fn local_shell_input_from_object(
        item: &serde_json::Map<String, serde_json::Value>,
    ) -> serde_json::Value {
        let action = item.get("action").unwrap_or(&serde_json::Value::Null);
        let mut action_obj = serde_json::Map::new();
        action_obj.insert("type".to_string(), serde_json::json!("exec"));

        if let Some(command) = action.get("command")
            && !command.is_null()
        {
            action_obj.insert("command".to_string(), command.clone());
        }
        if let Some(timeout_ms) = action.get("timeout_ms").or_else(|| action.get("timeoutMs"))
            && !timeout_ms.is_null()
        {
            action_obj.insert("timeoutMs".to_string(), timeout_ms.clone());
        }
        if let Some(user) = action.get("user")
            && !user.is_null()
        {
            action_obj.insert("user".to_string(), user.clone());
        }
        if let Some(working_directory) = action
            .get("working_directory")
            .or_else(|| action.get("workingDirectory"))
            && !working_directory.is_null()
        {
            action_obj.insert("workingDirectory".to_string(), working_directory.clone());
        }
        if let Some(env) = action.get("env")
            && !env.is_null()
        {
            action_obj.insert("env".to_string(), env.clone());
        }

        serde_json::json!({ "action": action_obj })
    }

    pub(super) fn convert_responses_event(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        // Handle delta as plain text or delta.content
        if let Some(delta) = json.get("delta") {
            // Case 1: delta is a plain string (response.output_text.delta)
            if let Some(s) = delta.as_str() {
                return Some(vec![crate::streaming::ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::TextDelta {
                        id: "text".to_string(),
                        delta: s.to_string(),
                        provider_metadata: None,
                    },
                }]);
            }
            // Case 2: delta.content is a string (message.delta simplified)
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                return Some(vec![crate::streaming::ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::TextDelta {
                        id: "text".to_string(),
                        delta: content.to_string(),
                        provider_metadata: None,
                    },
                }]);
            }

            // Handle tool_calls delta (first item only; downstream can coalesce)
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array())
                && let Some((_index, tool_call)) = tool_calls.iter().enumerate().next()
            {
                let id = tool_call
                    .get("id")
                    .and_then(|id| id.as_str())
                    .unwrap_or("")
                    .to_string();

                let function_name = tool_call
                    .get("function")
                    .and_then(|func| func.get("name"))
                    .and_then(|n| n.as_str())
                    .map(std::string::ToString::to_string);

                let arguments_delta = tool_call
                    .get("function")
                    .and_then(|func| func.get("arguments"))
                    .and_then(|a| a.as_str())
                    .map(std::string::ToString::to_string);

                let mut events = Vec::new();
                if let Some(name) = function_name.as_ref() {
                    events.push(crate::streaming::ChatStreamEvent::Part {
                        part: crate::types::ChatStreamPart::ToolInputStart {
                            id: id.clone(),
                            tool_name: name.clone(),
                            provider_metadata: None,
                            provider_executed: None,
                            dynamic: None,
                            title: None,
                        },
                    });
                }
                if let Some(arguments_delta) = arguments_delta.as_ref() {
                    events.push(crate::streaming::ChatStreamEvent::Part {
                        part: crate::types::ChatStreamPart::ToolInputDelta {
                            id: id.clone(),
                            delta: arguments_delta.clone(),
                            provider_metadata: None,
                        },
                    });
                }
                if let (Some(name), Some(input)) =
                    (function_name.as_ref(), arguments_delta.as_ref())
                    && serde_json::from_str::<serde_json::Value>(input).is_ok()
                {
                    events.push(crate::streaming::ChatStreamEvent::Part {
                        part: crate::types::ChatStreamPart::ToolInputEnd {
                            id: id.clone(),
                            provider_metadata: None,
                        },
                    });
                    events.push(crate::streaming::ChatStreamEvent::Part {
                        part: crate::types::ChatStreamPart::ToolCall(
                            crate::types::ChatStreamToolCall {
                                tool_call_id: id,
                                tool_name: name.clone(),
                                input: input.clone(),
                                provider_executed: None,
                                dynamic: None,
                                provider_metadata: None,
                            },
                        ),
                    });
                }
                return (!events.is_empty()).then_some(events);
            }
        }

        // Handle usage updates with both snake_case and camelCase fields
        if let Some(usage) = json
            .get("usage")
            .or_else(|| json.get("response")?.get("usage"))
        {
            return self.parse_responses_usage_value(usage).map(|usage| {
                vec![crate::streaming::ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::Finish {
                        usage,
                        finish_reason: crate::types::ChatStreamFinishInfo {
                            unified: crate::types::FinishReason::Unknown,
                            raw: None,
                        },
                        provider_metadata: None,
                    },
                }]
            });
        }

        None
    }

    pub(super) fn convert_message_output_item_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // response.output_item.added (message)
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("message") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let output_index = json
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        self.record_message_item_id(output_index, item_id);

        let id = self.text_stream_part_id(item_id);

        if self.has_emitted_text_start(&id) {
            return None;
        }
        self.mark_text_start_emitted(&id);

        Some(crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextStart {
                id,
                provider_metadata: self
                    .style_part_provider_metadata(serde_json::json!({ "itemId": item_id })),
            },
        })
    }

    pub(super) fn convert_output_text_delta_events(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        // response.output_text.delta
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() {
            return None;
        }
        let delta = json.get("delta").and_then(|v| v.as_str())?;

        let id = self.text_stream_part_id(item_id);

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if !self.has_emitted_text_start(&id) {
            self.mark_text_start_emitted(&id);
            events.push(crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::TextStart {
                    id: id.clone(),
                    provider_metadata: self
                        .style_part_provider_metadata(serde_json::json!({ "itemId": item_id })),
                },
            });
        }

        events.push(crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextDelta {
                id,
                delta: delta.to_string(),
                provider_metadata: None,
            },
        });

        Some(events)
    }

    pub(super) fn convert_message_output_item_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // response.output_item.done (message)
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("message") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let id = self.text_stream_part_id(item_id);

        if self.has_emitted_text_end(&id) {
            return None;
        }
        self.mark_text_end_emitted(&id);

        if self.stream_parts_style == StreamPartsStyle::Xai {
            // xAI Vercel-aligned stream parts omit providerMetadata and annotations.
            return Some(crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::TextEnd {
                    id,
                    provider_metadata: None,
                },
            });
        }

        let mut annotations = self.take_text_annotations(item_id);

        // If the message id changes between added/deltas and done, try to carry over annotations
        // captured under the original output_index message id.
        if annotations.is_empty()
            && let Some(output_index) = json.get("output_index").and_then(|v| v.as_u64())
            && let Some(original_id) = self.message_item_id_for_output_index(output_index)
            && original_id != item_id
        {
            annotations = self.take_text_annotations(&original_id);
        }

        if annotations.is_empty() {
            // Best-effort fallback: extract final annotations from completed message content.
            if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                for part in content {
                    if let Some(arr) = part.get("annotations").and_then(|v| v.as_array())
                        && !arr.is_empty()
                    {
                        annotations.extend(arr.iter().cloned());
                    }
                }
            }
        }

        let provider_metadata_openai = if annotations.is_empty() {
            serde_json::json!({
                "itemId": item_id,
            })
        } else {
            serde_json::json!({
                "itemId": item_id,
                "annotations": annotations,
            })
        };

        Some(crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextEnd {
                id,
                provider_metadata: self.style_part_provider_metadata(provider_metadata_openai),
            },
        })
    }

    pub(super) fn convert_finish_event(
        &self,
        completed_json: &serde_json::Value,
        response: &crate::types::ChatResponse,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let raw_usage = completed_json
            .get("response")
            .and_then(|r| r.get("usage"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let usage = self.parse_responses_usage_value(&raw_usage);
        let normalized_input = usage
            .as_ref()
            .map(crate::types::Usage::normalized_input_tokens)
            .unwrap_or_default();
        let normalized_output = usage
            .as_ref()
            .map(crate::types::Usage::normalized_output_tokens)
            .unwrap_or_default();

        let input_tokens = normalized_input.total.unwrap_or(0) as u64;
        let input_cache_read = normalized_input.cache_read.unwrap_or(0) as u64;
        let input_cache_write = normalized_input.cache_write.unwrap_or(0) as u64;
        let input_no_cache = normalized_input.no_cache.unwrap_or(0) as u64;
        let output_tokens = normalized_output.total.unwrap_or(0) as u64;
        let output_reasoning = normalized_output.reasoning.unwrap_or(0) as u64;
        let output_text = normalized_output.text.unwrap_or(0) as u64;
        let raw_usage_value = usage
            .as_ref()
            .and_then(crate::types::Usage::raw_usage_value)
            .unwrap_or(raw_usage.clone());
        let finish_usage = usage.clone().unwrap_or_else(|| {
            if self.responses_transform_style
                == crate::standards::openai::transformers::ResponsesTransformStyle::Xai
            {
                crate::standards::openai::compat::usage::xai_responses_zero_usage()
            } else {
                crate::types::Usage::builder()
                    .with_raw_usage_value(raw_usage_value.clone())
                    .build()
            }
        });

        let unified = response
            .finish_reason
            .clone()
            .unwrap_or(crate::types::FinishReason::Unknown);

        if self.stream_parts_style == StreamPartsStyle::Xai {
            let raw_finish = completed_json
                .get("response")
                .and_then(|r| r.get("status"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let provider_metadata =
                crate::standards::openai::compat::usage::xai_responses_usage_provider_metadata_value(
                    &raw_usage,
                )
                .and_then(|metadata| self.part_provider_metadata(metadata));

            return Some(crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::Finish {
                    usage: finish_usage,
                    finish_reason: crate::types::ChatStreamFinishInfo {
                        unified,
                        raw: raw_finish,
                    },
                    provider_metadata,
                },
            });
        }

        let response_id = self.created_response_id().or_else(|| {
            completed_json
                .get("response")?
                .get("id")?
                .as_str()
                .map(|s| s.to_string())
        });

        let service_tier = completed_json
            .get("response")
            .and_then(|r| r.get("service_tier"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let raw_finish_reason = completed_json
            .get("response")
            .and_then(|r| r.get("incomplete_details"))
            .and_then(|d| d.get("reason"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let mut provider_metadata = serde_json::Map::new();
        if let Some(id) = response_id {
            provider_metadata.insert("responseId".to_string(), serde_json::Value::String(id));
        }
        if let Some(tier) = service_tier {
            provider_metadata.insert("serviceTier".to_string(), serde_json::Value::String(tier));
        }
        if let Some(resp) = completed_json.get("response")
            && let Some(logprobs) =
                crate::standards::openai::transformers::response::extract_responses_output_text_logprobs(resp)
        {
            provider_metadata.insert("logprobs".to_string(), logprobs);
        }
        let provider_metadata = serde_json::Value::Object(provider_metadata);

        let _ = (
            input_tokens,
            input_cache_read,
            input_cache_write,
            input_no_cache,
            output_tokens,
            output_reasoning,
            output_text,
        );

        Some(crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::Finish {
                usage: finish_usage,
                finish_reason: crate::types::ChatStreamFinishInfo {
                    unified,
                    raw: raw_finish_reason,
                },
                provider_metadata: self.part_provider_metadata(provider_metadata),
            },
        })
    }

    fn reasoning_end_event(
        &self,
        id: String,
        provider_metadata: serde_json::Value,
    ) -> crate::streaming::ChatStreamEvent {
        crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ReasoningEnd {
                id,
                provider_metadata: self.reasoning_part_provider_metadata(provider_metadata),
            },
        }
    }

    pub(super) fn convert_reasoning_output_item_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("reasoning") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let encrypted_content = item
            .get("encrypted_content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        self.record_reasoning_encrypted_content(item_id, encrypted_content.clone());

        let id = match self.stream_parts_style {
            // Vercel alignment: a reasoning item implies at least one block (`:0`), even when summary is empty.
            StreamPartsStyle::OpenAi => format!("{item_id}:0"),
            // xAI Vercel stream parts use a single id without block suffix.
            StreamPartsStyle::Xai => self.reasoning_stream_part_id(item_id),
        };
        if self.has_emitted_reasoning_start(&id) {
            return None;
        }
        self.mark_reasoning_start_emitted(&id);
        self.record_reasoning_part_id(item_id, &id);

        Some(crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ReasoningStart {
                id,
                provider_metadata: match self.stream_parts_style {
                    StreamPartsStyle::OpenAi => {
                        self.reasoning_part_provider_metadata(serde_json::json!({
                            "itemId": item_id,
                            "reasoningEncryptedContent": encrypted_content,
                        }))
                    }
                    StreamPartsStyle::Xai => {
                        self.reasoning_part_provider_metadata(serde_json::json!({
                            "itemId": item_id,
                        }))
                    }
                },
            },
        })
    }

    pub(super) fn convert_reasoning_summary_part_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        // response.reasoning_summary_part.added
        let item_id = json.get("item_id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        if self.stream_parts_style == StreamPartsStyle::Xai {
            // xAI Vercel stream parts do not expose block indices for reasoning summaries.
            // Ensure a start event exists and otherwise ignore this event.
            let id = self.reasoning_stream_part_id(item_id);
            if self.has_emitted_reasoning_start(&id) {
                return None;
            }
            self.mark_reasoning_start_emitted(&id);
            self.record_reasoning_part_id(item_id, &id);
            return Some(vec![crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ReasoningStart {
                    id,
                    provider_metadata: self.reasoning_part_provider_metadata(serde_json::json!({
                        "itemId": item_id,
                    })),
                },
            }]);
        }
        let summary_index = json
            .get("summary_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
        if summary_index > 0 {
            for id in self.take_reasoning_parts_can_conclude(item_id) {
                if self.has_emitted_reasoning_end(&id) {
                    continue;
                }
                self.mark_reasoning_end_emitted(&id);
                events.push(self.reasoning_end_event(
                    id,
                    serde_json::json!({
                        "itemId": item_id,
                    }),
                ));
            }
        }

        let id = format!("{item_id}:{summary_index}");
        if !self.has_emitted_reasoning_start(&id) {
            self.mark_reasoning_start_emitted(&id);
            self.record_reasoning_part_id(item_id, &id);

            let encrypted_content = self.reasoning_encrypted_content(item_id);

            events.push(crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ReasoningStart {
                    id,
                    provider_metadata: self.reasoning_part_provider_metadata(serde_json::json!({
                        "itemId": item_id,
                        "reasoningEncryptedContent": encrypted_content,
                    })),
                },
            });
        }

        if events.is_empty() {
            None
        } else {
            Some(events)
        }
    }

    pub(super) fn convert_reasoning_summary_text_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // response.reasoning_summary_text.delta
        let item_id = json.get("item_id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }
        let delta = json.get("delta").and_then(|v| v.as_str())?;

        match self.stream_parts_style {
            StreamPartsStyle::OpenAi => {
                let summary_index = json
                    .get("summary_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let id = format!("{item_id}:{summary_index}");

                // Ensure a start event exists for this block.
                if !self.has_emitted_reasoning_start(&id) {
                    self.mark_reasoning_start_emitted(&id);
                    self.record_reasoning_part_id(item_id, &id);
                }

                Some(crate::streaming::ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::ReasoningDelta {
                        id,
                        delta: delta.to_string(),
                        provider_metadata: self.reasoning_part_provider_metadata(
                            serde_json::json!({ "itemId": item_id }),
                        ),
                    },
                })
            }
            StreamPartsStyle::Xai => {
                let id = self.reasoning_stream_part_id(item_id);
                if !self.has_emitted_reasoning_start(&id) {
                    self.mark_reasoning_start_emitted(&id);
                    self.record_reasoning_part_id(item_id, &id);
                }

                Some(crate::streaming::ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::ReasoningDelta {
                        id,
                        delta: delta.to_string(),
                        provider_metadata: self.reasoning_part_provider_metadata(
                            serde_json::json!({ "itemId": item_id }),
                        ),
                    },
                })
            }
        }
    }

    pub(super) fn convert_reasoning_summary_part_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // response.reasoning_summary_part.done
        let item_id = json.get("item_id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let id = match self.stream_parts_style {
            StreamPartsStyle::OpenAi => {
                let summary_index = json
                    .get("summary_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                format!("{item_id}:{summary_index}")
            }
            StreamPartsStyle::Xai => self.reasoning_stream_part_id(item_id),
        };

        if self.requested_store != Some(true) {
            self.mark_reasoning_part_can_conclude(item_id, &id);
            return None;
        }

        if self.has_emitted_reasoning_end(&id) {
            return None;
        }
        self.mark_reasoning_end_emitted(&id);

        Some(self.reasoning_end_event(
            id,
            serde_json::json!({
                "itemId": item_id,
            }),
        ))
    }

    pub(super) fn convert_reasoning_output_item_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("reasoning") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let encrypted_content = item
            .get("encrypted_content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        self.record_reasoning_encrypted_content(item_id, encrypted_content.clone());

        if self.stream_parts_style == StreamPartsStyle::Xai {
            let id = self.reasoning_stream_part_id(item_id);
            if self.has_emitted_reasoning_end(&id) {
                return None;
            }
            let mut events = Vec::new();
            if !self.has_emitted_reasoning_start(&id) {
                self.mark_reasoning_start_emitted(&id);
                self.record_reasoning_part_id(item_id, &id);
                events.push(crate::streaming::ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::ReasoningStart {
                        id: id.clone(),
                        provider_metadata: self.reasoning_part_provider_metadata(
                            serde_json::json!({
                                "itemId": item_id,
                            }),
                        ),
                    },
                });
            }
            self.mark_reasoning_end_emitted(&id);
            let provider_metadata = if let Some(enc) = encrypted_content.as_ref() {
                serde_json::json!({
                    "itemId": item_id,
                    "reasoningEncryptedContent": enc,
                })
            } else {
                serde_json::json!({
                    "itemId": item_id,
                })
            };
            events.push(crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ReasoningEnd {
                    id,
                    provider_metadata: self.reasoning_part_provider_metadata(provider_metadata),
                },
            });

            return Some(events);
        }

        let summary_len = item
            .get("summary")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let blocks = std::cmp::max(1, summary_len);

        let mut ids: Vec<String> = (0..blocks).map(|i| format!("{item_id}:{i}")).collect();
        ids.extend(self.take_reasoning_parts_can_conclude(item_id));
        ids.extend(self.take_reasoning_part_ids(item_id));
        ids.sort();
        ids.dedup();

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
        for id in ids {
            if self.has_emitted_reasoning_end(&id) {
                continue;
            }
            self.mark_reasoning_end_emitted(&id);

            // Vercel alignment: omit reasoningEncryptedContent when it is null/absent.
            let provider_metadata = if let Some(enc) = encrypted_content.as_ref() {
                serde_json::json!({
                    "itemId": item_id,
                    "reasoningEncryptedContent": enc,
                })
            } else {
                serde_json::json!({
                    "itemId": item_id,
                })
            };

            events.push(crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ReasoningEnd {
                    id,
                    provider_metadata: self.reasoning_part_provider_metadata(provider_metadata),
                },
            });
        }

        Some(events)
    }

    pub(super) fn convert_output_text_annotation_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let annotation = json.get("annotation")?;

        if !item_id.is_empty() {
            self.record_text_annotation(item_id, annotation.clone());
        }

        let ann_type = annotation.get("type")?.as_str()?;

        if ann_type == "url_citation" {
            let url = annotation.get("url")?.as_str()?;
            let title = annotation.get("title").and_then(|v| v.as_str());
            let start_index = annotation.get("start_index").and_then(|v| v.as_u64());
            let id = start_index
                .map(|s| format!("ann:url:{s}"))
                .unwrap_or_else(|| format!("ann:url:{url}"));

            return Some(crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::Source {
                    id,
                    source: crate::types::SourcePart::Url {
                        url: url.to_string(),
                        title: title.map(ToString::to_string),
                    },
                    provider_metadata: None,
                },
            });
        }

        if matches!(
            ann_type,
            "file_citation" | "container_file_citation" | "file_path"
        ) {
            let file_id = annotation.get("file_id")?.as_str()?;
            let filename = annotation
                .get("filename")
                .and_then(|v| v.as_str())
                .unwrap_or(file_id);
            let quote = annotation.get("quote").and_then(|v| v.as_str());

            let media_type = if ann_type == "file_path" {
                "application/octet-stream"
            } else {
                "text/plain"
            };

            let title = quote.unwrap_or(filename);
            let start_index = annotation.get("start_index").and_then(|v| v.as_u64());
            let annotation_index = json.get("annotation_index").and_then(|v| v.as_u64());
            let id = start_index
                .map(|s| format!("ann:doc:{s}"))
                .or_else(|| annotation_index.map(|idx| format!("ann:doc:{file_id}:{idx}")))
                .unwrap_or_else(|| format!("ann:doc:{file_id}"));

            let provider_metadata = match ann_type {
                "file_citation" => self.part_provider_metadata(serde_json::json!({
                    "type": "file_citation",
                    "fileId": file_id,
                    "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
                })),
                "container_file_citation" => self.part_provider_metadata(serde_json::json!({
                    "type": "container_file_citation",
                    "fileId": file_id,
                    "containerId": annotation.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
                    "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
                })),
                "file_path" => self.part_provider_metadata(serde_json::json!({
                    "type": "file_path",
                    "fileId": file_id,
                    "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
                })),
                _ => None,
            };

            return Some(crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::Source {
                    id,
                    source: crate::types::SourcePart::Document {
                        media_type: media_type.to_string(),
                        title: title.to_string(),
                        filename: Some(filename.to_string()),
                    },
                    provider_metadata,
                },
            });
        }

        None
    }

    pub(super) fn convert_function_call_output_item_added_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item = json.get("item")?;
        if item.get("type").and_then(|t| t.as_str()) != Some("function_call") {
            return None;
        }

        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
        let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");

        if !item_id.is_empty() && !call_id.is_empty() && !tool_name.is_empty() {
            self.record_function_call_meta(item_id, call_id, tool_name);
        }

        if call_id.is_empty() || tool_name.is_empty() {
            return None;
        }

        if !self.mark_function_tool_input_start_emitted(call_id) {
            return None;
        }

        Some(self.openai_tool_input_start_event(call_id, tool_name, None, None, None, None))
    }

    pub(super) fn convert_function_call_arguments_delta_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let delta = json.get("delta").and_then(|d| d.as_str())?;
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");

        let (call_id, _tool_name) = self.function_call_meta(item_id)?;
        if call_id.is_empty() {
            return None;
        }

        Some(self.openai_tool_input_delta_event(&call_id, delta))
    }

    pub(super) fn convert_function_call_arguments_done_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let args = json.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() {
            return None;
        }

        let (call_id, tool_name) = self.function_call_meta(item_id)?;
        if call_id.is_empty() || tool_name.is_empty() {
            return None;
        }

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if !self.has_emitted_function_tool_input_end(call_id.as_str())
            && self.mark_function_tool_input_end_emitted(call_id.as_str())
        {
            events.push(self.openai_tool_input_end_event(&call_id, None));
        }

        events.push(self.openai_tool_call_event(
            &call_id,
            &tool_name,
            serde_json::json!(args),
            None,
            None,
            Some(self.provider_metadata_json(serde_json::json!({
                "itemId": item_id,
            }))),
            OpenAiResponsesEventExtras::default(),
        ));

        Some(events)
    }

    pub(super) fn convert_apply_patch_output_item_added_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("apply_patch_call") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
        let operation = item
            .get("operation")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        if call_id.is_empty() {
            return None;
        }
        self.record_apply_patch_call(item_id, call_id, operation.clone());

        if !self.mark_apply_patch_tool_input_start_emitted(call_id) {
            return None;
        }

        let tool_name = self
            .provider_tool_name_for_item_type("apply_patch_call")
            .unwrap_or_else(|| "apply_patch".to_string());

        let mut events: Vec<crate::streaming::ChatStreamEvent> =
            vec![self.openai_tool_input_start_event(call_id, &tool_name, None, None, None, None)];

        let op_type = operation.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let path = operation.get("path").and_then(|v| v.as_str());

        let call_id_json = serde_json::to_string(call_id).unwrap_or_else(|_| "\"\"".to_string());
        let op_type_json = serde_json::to_string(op_type).unwrap_or_else(|_| "\"\"".to_string());
        let path_json = path
            .and_then(|p| serde_json::to_string(p).ok())
            .unwrap_or_else(|| "null".to_string());

        if op_type == "delete_file" {
            let input = format!(
                "{{\"callId\":{call_id_json},\"operation\":{{\"type\":{op_type_json},\"path\":{path_json}}}}}"
            );

            events.push(self.openai_tool_input_delta_event(call_id, &input));

            if self.mark_apply_patch_tool_input_end_emitted(call_id) {
                events.push(self.openai_tool_input_end_event(call_id, None));
            }

            return Some(events);
        }

        let prefix = format!(
            "{{\"callId\":{call_id_json},\"operation\":{{\"type\":{op_type_json},\"path\":{path_json},\"diff\":\""
        );
        events.push(self.openai_tool_input_delta_event(call_id, &prefix));

        Some(events)
    }

    pub(super) fn convert_apply_patch_operation_diff_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let delta = json.get("delta").and_then(|v| v.as_str())?;
        if item_id.is_empty() {
            return None;
        }
        let call_id = self.apply_patch_call_id(item_id)?;
        if call_id.is_empty() {
            return None;
        }
        if !self.has_emitted_apply_patch_tool_input_start(call_id.as_str()) {
            return None;
        }

        self.mark_apply_patch_diff_seen(&call_id);
        let escaped_delta = Self::escape_json_delta(delta);
        Some(self.openai_tool_input_delta_event(&call_id, &escaped_delta))
    }

    pub(super) fn convert_apply_patch_operation_diff_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() {
            return None;
        }
        let call_id = self.apply_patch_call_id(item_id)?;
        if call_id.is_empty() || self.has_emitted_apply_patch_tool_input_end(call_id.as_str()) {
            return None;
        }

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if !self.has_seen_apply_patch_diff(&call_id)
            && let Some(diff) = json.get("diff").and_then(|v| v.as_str())
            && !diff.is_empty()
        {
            let escaped_diff = Self::escape_json_delta(diff);
            events.push(self.openai_tool_input_delta_event(&call_id, &escaped_diff));
            self.mark_apply_patch_diff_seen(&call_id);
        }

        // Close the open `diff` string and the surrounding objects: `"}}`
        events.push(self.openai_tool_input_delta_event(&call_id, "\"}}"));

        if self.mark_apply_patch_tool_input_end_emitted(call_id.as_str()) {
            events.push(self.openai_tool_input_end_event(&call_id, None));
        }

        Some(events)
    }

    pub(super) fn convert_code_interpreter_output_item_added_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        if item.get("type").and_then(|t| t.as_str()) != Some("code_interpreter_call") {
            return None;
        }

        let item_id = item.get("id")?.as_str()?;
        if item_id.is_empty() {
            return None;
        }

        let tool_name = self
            .provider_tool_name_for_item_type("code_interpreter_call")
            .unwrap_or_else(|| "code_interpreter".to_string());

        let container_id = item
            .get("container_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !container_id.is_empty() {
            self.record_code_interpreter_container_id(item_id, container_id);
        }

        if !self.mark_code_interpreter_tool_input_start_emitted(item_id) {
            return None;
        }

        let container_id_json =
            serde_json::to_string(container_id).unwrap_or_else(|_| "\"\"".to_string());
        let prefix = format!("{{\"containerId\":{container_id_json},\"code\":\"");

        Some(vec![
            self.openai_tool_input_start_event(item_id, &tool_name, Some(true), None, None, None),
            self.openai_tool_input_delta_event(item_id, &prefix),
        ])
    }

    pub(super) fn convert_code_interpreter_code_delta_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let delta = json.get("delta").and_then(|v| v.as_str())?;
        if item_id.is_empty() {
            return None;
        }

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if !self.has_emitted_code_interpreter_tool_input_start(item_id)
            && self.mark_code_interpreter_tool_input_start_emitted(item_id)
        {
            let tool_name = self
                .provider_tool_name_for_item_type("code_interpreter_call")
                .unwrap_or_else(|| "code_interpreter".to_string());
            let container_id = self
                .code_interpreter_container_id(item_id)
                .unwrap_or_default();
            let container_id_json =
                serde_json::to_string(container_id.as_str()).unwrap_or_else(|_| "\"\"".to_string());
            let prefix = format!("{{\"containerId\":{container_id_json},\"code\":\"");

            events.push(self.openai_tool_input_start_event(
                item_id,
                &tool_name,
                Some(true),
                None,
                None,
                None,
            ));
            events.push(self.openai_tool_input_delta_event(item_id, &prefix));
        }

        let escaped_delta = Self::escape_json_delta(delta);
        events.push(self.openai_tool_input_delta_event(item_id, &escaped_delta));

        Some(events)
    }

    pub(super) fn convert_code_interpreter_code_done_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() || self.has_emitted_code_interpreter_tool_input_end(item_id) {
            return None;
        }

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        // Close the open `code` string and the object: `"}`
        events.push(self.openai_tool_input_delta_event(item_id, "\"}"));

        if self.mark_code_interpreter_tool_input_end_emitted(item_id) {
            events.push(self.openai_tool_input_end_event(item_id, None));
        }

        if self.mark_code_interpreter_tool_call_emitted(item_id) {
            let tool_name = self
                .provider_tool_name_for_item_type("code_interpreter_call")
                .unwrap_or_else(|| "code_interpreter".to_string());
            let code = json.get("code").and_then(|v| v.as_str()).unwrap_or("");
            let container_id = self
                .code_interpreter_container_id(item_id)
                .unwrap_or_default();
            let code_json = serde_json::to_string(code).unwrap_or_else(|_| "\"\"".to_string());
            let container_id_json =
                serde_json::to_string(container_id.as_str()).unwrap_or_else(|_| "\"\"".to_string());
            let input = format!("{{\"code\":{code_json},\"containerId\":{container_id_json}}}");

            events.push(self.openai_tool_call_event(
                item_id,
                &tool_name,
                serde_json::Value::String(input),
                Some(true),
                None,
                None,
                OpenAiResponsesEventExtras {
                    output_index: json.get("output_index").and_then(|value| value.as_u64()),
                    raw_item: None,
                },
            ));
        }

        Some(events)
    }

    pub(super) fn convert_provider_tool_output_item_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        let item_type = item.get("type")?.as_str()?;
        let output_index = json.get("output_index").and_then(|v| v.as_u64());

        let (default_tool_name, input) = match item_type {
            "mcp_call" => {
                // MCP tool-call/result parts are emitted on output_item.done, matching AI SDK.
                return None;
            }
            "mcp_approval_request" => {
                // MCP approval requests are emitted on output_item.done, matching AI SDK.
                return None;
            }
            "custom_tool_call" => {
                // xAI x_search emits internal custom tool calls (e.g. `x_keyword_search`) and streams
                // their input via `response.custom_tool_call_input.*` events.
                let tool_call_id = item.get("id")?.as_str()?;
                let call_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let tool_name = self.custom_tool_name_for_call_name(call_name);
                self.record_custom_tool_item(tool_call_id, call_name, &tool_name);

                if self.stream_parts_style == StreamPartsStyle::Xai {
                    // AI SDK xAI alignment: defer tool-input-* and tool-call emission until
                    // `response.output_item.done`, using the finalized input string.
                    return None;
                }

                if self.mark_custom_tool_input_start_emitted(tool_call_id) {
                    return Some(vec![self.openai_tool_input_start_event(
                        tool_call_id,
                        &tool_name,
                        None,
                        None,
                        None,
                        None,
                    )]);
                }

                return None;
            }
            "web_search_call" => {
                // xAI streams `arguments` on the output item; Vercel aligns web_search tool input
                // to this JSON string when present. OpenAI does not always include arguments.
                let args = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
                if args.is_empty() {
                    (
                        "web_search",
                        serde_json::Value::String(self.web_search_default_input.clone()),
                    )
                } else {
                    ("web_search", serde_json::Value::String(args.to_string()))
                }
            }
            "file_search_call" => {
                let input = match self.stream_parts_style {
                    StreamPartsStyle::OpenAi => serde_json::json!("{}"),
                    StreamPartsStyle::Xai => serde_json::json!(""),
                };
                ("file_search", input)
            }
            "computer_call" => {
                let tool_call_id = item.get("id")?.as_str()?;
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "computer_use".to_string());

                return Some(vec![self.openai_tool_input_start_event(
                    tool_call_id,
                    &tool_name,
                    Some(true),
                    None,
                    None,
                    None,
                )]);
            }
            "code_interpreter_call" => {
                let container_id = item.get("container_id").and_then(|v| v.as_str());
                let tool_call_id = item.get("id")?.as_str()?;
                if let Some(cid) = container_id {
                    self.record_code_interpreter_container_id(tool_call_id, cid);
                }

                // Vercel alignment: code interpreter tool-call is emitted after tool-input-end,
                // once the full code is known.
                return None;
            }
            "tool_search_call" => {
                let is_hosted = item.get("execution").and_then(|v| v.as_str()) == Some("server");
                if is_hosted {
                    let tool_call_id = item.get("id")?.as_str()?;
                    let tool_name = self
                        .provider_tool_name_for_item_type(item_type)
                        .unwrap_or_else(|| "toolSearch".to_string());
                    let should_emit = self
                        .emitted_tool_search_input_start_ids
                        .lock()
                        .map(|mut ids| ids.insert(tool_call_id.to_string()))
                        .unwrap_or(false);
                    if should_emit {
                        return Some(vec![self.openai_tool_input_start_event(
                            tool_call_id,
                            &tool_name,
                            Some(true),
                            None,
                            None,
                            None,
                        )]);
                    }
                }

                return None;
            }
            "tool_search_output" => {
                // Tool search output is paired with the call at output_item.done.
                return None;
            }
            "image_generation_call" => ("image_generation", serde_json::json!("{}")),
            _ => return None,
        };

        let tool_name = self
            .provider_tool_name_for_item_type(item_type)
            .unwrap_or_else(|| default_tool_name.to_string());

        let tool_call_id = item.get("id")?.as_str()?;

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        let emit_provider_tool_input = match item_type {
            "web_search_call" => self.mark_web_search_tool_input_emitted(tool_call_id),
            "file_search_call" if self.stream_parts_style == StreamPartsStyle::Xai => {
                self.mark_web_search_tool_input_emitted(tool_call_id)
            }
            _ => false,
        };

        // Vercel alignment:
        // - webSearch emits tool-input-start/end even with empty input
        // - xAI file_search also emits empty tool-input-* lifecycle events before tool-call
        if emit_provider_tool_input {
            let provider_executed = if item_type == "web_search_call" {
                self.include_web_search_provider_executed_in_tool_input
                    .then_some(true)
            } else {
                None
            };
            events.push(self.openai_tool_input_start_event(
                tool_call_id,
                &tool_name,
                provider_executed,
                None,
                None,
                None,
            ));

            let should_emit_delta = match item_type {
                "web_search_call" => self.emit_web_search_tool_input_delta,
                "file_search_call" if self.stream_parts_style == StreamPartsStyle::Xai => true,
                _ => false,
            };
            if should_emit_delta && let serde_json::Value::String(delta) = &input {
                events.push(self.openai_tool_input_delta_event(tool_call_id, delta));
            }

            events.push(self.openai_tool_input_end_event(tool_call_id, None));
        }

        events.push(self.openai_tool_call_event(
            tool_call_id,
            &tool_name,
            input,
            Some(true),
            None,
            None,
            OpenAiResponsesEventExtras {
                output_index,
                raw_item: Some(serde_json::Value::Object(item.clone())),
            },
        ));

        Some(events)
    }

    pub(super) fn convert_image_generation_partial_image(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let tool_call_id = json.get("item_id")?.as_str()?;
        if tool_call_id.is_empty() {
            return None;
        }
        let partial_image_b64 = json.get("partial_image_b64")?.as_str()?;
        let output_index = json.get("output_index").and_then(|value| value.as_u64());

        let tool_name = self
            .provider_tool_name_for_item_type("image_generation_call")
            .unwrap_or_else(|| "image_generation".to_string());

        Some(self.openai_tool_result_event_with_preliminary(
            tool_call_id,
            &tool_name,
            serde_json::json!({
                "result": partial_image_b64,
            }),
            None,
            None,
            Some(true),
            None,
            OpenAiResponsesEventExtras {
                output_index,
                raw_item: None,
            },
        ))
    }

    pub(super) fn convert_provider_tool_output_item_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        let item_type = item.get("type")?.as_str()?;
        let output_index = json.get("output_index").and_then(|v| v.as_u64());

        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let tool_call_id = item
            .get("call_id")
            .and_then(|v| v.as_str())
            .filter(|id| !id.is_empty())
            .unwrap_or(item_id);
        if tool_call_id.is_empty() {
            return None;
        }

        let mut extra_events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if item_type == "custom_tool_call" {
            let custom_item_id = if item_id.is_empty() {
                tool_call_id
            } else {
                item_id
            };

            if !self.mark_custom_tool_call_emitted(custom_item_id) {
                return None;
            }

            let call_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let tool_name = self.custom_tool_name_for_call_name(call_name);
            let input = item
                .get("input")
                .and_then(|v| v.as_str())
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .or_else(|| self.take_custom_tool_input_by_item_id(custom_item_id))
                .unwrap_or_default();

            if self.stream_parts_style == StreamPartsStyle::Xai {
                return Some(vec![
                    self.openai_tool_input_start_event(
                        tool_call_id,
                        &tool_name,
                        None,
                        None,
                        None,
                        None,
                    ),
                    self.openai_tool_input_delta_event(tool_call_id, &input),
                    self.openai_tool_input_end_event(tool_call_id, None),
                    self.openai_tool_call_event(
                        tool_call_id,
                        &tool_name,
                        serde_json::json!(input),
                        Some(true),
                        None,
                        None,
                        OpenAiResponsesEventExtras {
                            output_index,
                            raw_item: Some(serde_json::Value::Object(item.clone())),
                        },
                    ),
                ]);
            }

            let mut events = vec![self.openai_tool_call_event(
                tool_call_id,
                &tool_name,
                serde_json::json!(input.as_str()),
                Some(true),
                None,
                None,
                OpenAiResponsesEventExtras {
                    output_index,
                    raw_item: Some(serde_json::Value::Object(item.clone())),
                },
            )];

            if let Some(output) = item.get("output") {
                events.push(self.openai_tool_result_event(
                    tool_call_id,
                    &tool_name,
                    output.clone(),
                    Some(true),
                    None,
                    item.get("is_error").and_then(|v| v.as_bool()),
                    Some(self.provider_metadata_json(serde_json::json!({
                        "itemId": tool_call_id,
                    }))),
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                ));
            }

            return Some(events);
        }

        let (default_tool_name, result) = match item_type {
            "mcp_approval_request" => {
                let approval_id = item.get("id")?.as_str()?;
                if self.has_emitted_mcp_approval_request(approval_id) {
                    return None;
                }

                let tool_call_id = self.mcp_approval_tool_call_id(approval_id);
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let tool_name = format!("mcp.{name}");

                extra_events.push(self.openai_tool_call_event(
                    &tool_call_id,
                    &tool_name,
                    serde_json::json!(args),
                    Some(true),
                    Some(true),
                    None,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                ));
                extra_events.push(self.openai_tool_approval_request_event(
                    approval_id,
                    &tool_call_id,
                    None,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                ));

                self.mark_mcp_approval_request_emitted(approval_id);
                return Some(extra_events);
            }
            "mcp_call" => {
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let server_label = item
                    .get("server_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| self.mcp_call_args(tool_call_id))
                    .unwrap_or_else(|| "{}".to_string());
                let output = item
                    .get("output")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let error = item
                    .get("error")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                let tool_name = format!("mcp.{name}");
                let tool_name_for_result = tool_name.clone();
                let args_for_result = args.clone();

                if !self.has_emitted_mcp_call(tool_call_id) {
                    extra_events.push(self.openai_tool_call_event(
                        tool_call_id,
                        &tool_name,
                        serde_json::json!(args),
                        Some(true),
                        Some(true),
                        None,
                        OpenAiResponsesEventExtras {
                            output_index,
                            raw_item: Some(serde_json::Value::Object(item.clone())),
                        },
                    ));
                    self.mark_mcp_call_emitted(tool_call_id);
                }

                if self.has_emitted_mcp_result(tool_call_id) {
                    return Some(extra_events);
                }
                self.mark_mcp_result_emitted(tool_call_id);

                let mut result = serde_json::Map::new();
                result.insert("type".to_string(), serde_json::json!("call"));
                result.insert(
                    "serverLabel".to_string(),
                    serde_json::Value::String(server_label.to_string()),
                );
                result.insert(
                    "name".to_string(),
                    serde_json::Value::String(name.to_string()),
                );
                result.insert(
                    "arguments".to_string(),
                    serde_json::Value::String(args_for_result),
                );
                if !output.is_null() {
                    result.insert("output".to_string(), output);
                }
                if !error.is_null() {
                    result.insert("error".to_string(), error);
                }
                let provider_metadata = (!item_id.is_empty())
                    .then(|| self.provider_metadata_json(serde_json::json!({ "itemId": item_id })));

                extra_events.push(self.openai_tool_result_event(
                    tool_call_id,
                    &tool_name_for_result,
                    serde_json::Value::Object(result),
                    Some(true),
                    Some(true),
                    None,
                    provider_metadata,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                ));
                return Some(extra_events);
            }
            "web_search_call" => {
                if !self.emit_web_search_tool_result {
                    return None;
                }

                // xAI web search fixtures do not include `action`/`results` payloads in `output_item.done`.
                // When there is no result payload, Vercel does not emit a `tool-result` stream part.
                if item.get("action").is_none() && item.get("results").is_none() {
                    return None;
                }

                // Emit Vercel-aligned sources for web search results.
                if let Some(arr) = item.get("results").and_then(|v| v.as_array()) {
                    for (i, r) in arr.iter().enumerate() {
                        let Some(obj) = r.as_object() else {
                            continue;
                        };
                        let Some(url) = obj.get("url").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let title = obj.get("title").and_then(|v| v.as_str());

                        extra_events.push(
                            crate::streaming::TypedStreamPart::Source(
                                crate::streaming::TypedStreamSource::Url {
                                    id: format!("{tool_call_id}:{i}"),
                                    url: url.to_string(),
                                    title: title.map(ToString::to_string),
                                    provider_metadata: None,
                                },
                            )
                            .to_part_event(),
                        );
                    }
                }

                ("web_search", Self::web_search_result_from_object(item))
            }
            "file_search_call" => {
                let result = match self.stream_parts_style {
                    StreamPartsStyle::OpenAi => serde_json::json!({
                        "queries": item.get("queries").cloned().unwrap_or(serde_json::Value::Null),
                        "results": Self::file_search_results_from_object(item),
                    }),
                    StreamPartsStyle::Xai => serde_json::json!({
                        "queries": Self::xai_file_search_queries_from_object(item),
                        "results": Self::file_search_results_from_object(item),
                    }),
                };
                ("file_search", result)
            }
            "code_interpreter_call" => {
                // Vercel alignment: codeExecution streams tool input, then emits tool-call, then tool-result.
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "code_interpreter".to_string());

                let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

                if !self.has_emitted_code_interpreter_tool_input_end(tool_call_id)
                    && self.mark_code_interpreter_tool_input_end_emitted(tool_call_id)
                {
                    events.push(self.openai_tool_input_delta_event(tool_call_id, "\"}"));
                    events.push(self.openai_tool_input_end_event(tool_call_id, None));
                }

                let container_id = item.get("container_id").and_then(|v| v.as_str());
                let code = item.get("code").and_then(|v| v.as_str()).unwrap_or("");

                let code_json = serde_json::to_string(code).unwrap_or_else(|_| "\"\"".to_string());
                let container_id_json = container_id
                    .and_then(|cid| serde_json::to_string(cid).ok())
                    .unwrap_or_else(|| "null".to_string());

                let input = format!("{{\"code\":{code_json},\"containerId\":{container_id_json}}}");

                if self.mark_code_interpreter_tool_call_emitted(tool_call_id) {
                    events.push(self.openai_tool_call_event(
                        tool_call_id,
                        &tool_name,
                        serde_json::Value::String(input),
                        Some(true),
                        None,
                        None,
                        OpenAiResponsesEventExtras {
                            output_index,
                            raw_item: Some(serde_json::Value::Object(item.clone())),
                        },
                    ));
                }

                events.push(self.openai_tool_result_event(
                    tool_call_id,
                    &tool_name,
                    serde_json::json!({
                        "outputs": item.get("outputs").cloned().unwrap_or_else(|| serde_json::json!([])),
                    }),
                    Some(true),
                    None,
                    None,
                    None,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                ));

                return Some(events);
            }
            "image_generation_call" => (
                "image_generation",
                serde_json::json!({
                    "result": item.get("result").cloned().unwrap_or(serde_json::Value::Null),
                }),
            ),
            "tool_search_call" => {
                let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let call_id = item.get("call_id").and_then(|v| v.as_str());
                let is_hosted = item.get("execution").and_then(|v| v.as_str()) == Some("server");
                let tool_call_id = if is_hosted {
                    item_id
                } else {
                    call_id.filter(|id| !id.is_empty()).unwrap_or(item_id)
                };
                if tool_call_id.is_empty() {
                    return None;
                }

                if is_hosted && let Ok(mut ids) = self.hosted_tool_search_call_ids.lock() {
                    ids.push_back(tool_call_id.to_string());
                }

                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "toolSearch".to_string());
                let arguments_json = serde_json::to_string(
                    item.get("arguments").unwrap_or(&serde_json::Value::Null),
                )
                .unwrap_or_else(|_| "null".to_string());
                let call_id_json = if is_hosted {
                    "null".to_string()
                } else {
                    serde_json::to_string(tool_call_id).unwrap_or_else(|_| "null".to_string())
                };
                let input =
                    format!("{{\"arguments\":{arguments_json},\"call_id\":{call_id_json}}}");
                let provider_metadata = (!item_id.is_empty())
                    .then(|| self.provider_metadata_json(serde_json::json!({ "itemId": item_id })));

                let mut events = Vec::new();
                if is_hosted {
                    let has_started = self
                        .emitted_tool_search_input_start_ids
                        .lock()
                        .map(|ids| ids.contains(tool_call_id))
                        .unwrap_or(false);
                    if has_started {
                        events.push(self.openai_tool_input_end_event(tool_call_id, None));
                    }
                } else {
                    events.push(self.openai_tool_input_start_event(
                        tool_call_id,
                        &tool_name,
                        None,
                        None,
                        None,
                        None,
                    ));
                    events.push(self.openai_tool_input_end_event(tool_call_id, None));
                }
                events.push(self.openai_tool_call_event(
                    tool_call_id,
                    &tool_name,
                    serde_json::Value::String(input),
                    is_hosted.then_some(true),
                    None,
                    provider_metadata,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                ));
                return Some(events);
            }
            "tool_search_output" => {
                let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let call_id = item.get("call_id").and_then(|v| v.as_str());
                let tool_call_id = if let Some(call_id) = call_id.filter(|id| !id.is_empty()) {
                    call_id.to_string()
                } else {
                    self.hosted_tool_search_call_ids
                        .lock()
                        .ok()
                        .and_then(|mut ids| ids.pop_front())
                        .unwrap_or_else(|| item_id.to_string())
                };
                if tool_call_id.is_empty() {
                    return None;
                }

                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "toolSearch".to_string());
                let provider_metadata = (!item_id.is_empty())
                    .then(|| self.provider_metadata_json(serde_json::json!({ "itemId": item_id })));

                return Some(vec![self.openai_tool_result_event(
                    &tool_call_id,
                    &tool_name,
                    serde_json::json!({
                        "tools": item.get("tools").cloned().unwrap_or_else(|| {
                            serde_json::Value::Array(Vec::new())
                        }),
                    }),
                    None,
                    None,
                    None,
                    provider_metadata,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                )]);
            }
            "local_shell_call" => {
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "shell".to_string());

                let input = Self::local_shell_input_from_object(item).to_string();
                let provider_metadata = (!item_id.is_empty())
                    .then(|| self.provider_metadata_json(serde_json::json!({ "itemId": item_id })));

                return Some(vec![self.openai_tool_call_event(
                    tool_call_id,
                    &tool_name,
                    serde_json::Value::String(input),
                    None,
                    None,
                    provider_metadata,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                )]);
            }
            "shell_call" => {
                let action = item
                    .get("action")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "shell".to_string());

                // Vercel alignment: only expose the commands list to the shell executor.
                let commands = action
                    .as_object()
                    .and_then(|m| m.get("commands"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let input = serde_json::json!({
                    "action": {
                        "commands": commands,
                    }
                })
                .to_string();
                let provider_metadata = (!item_id.is_empty())
                    .then(|| self.provider_metadata_json(serde_json::json!({ "itemId": item_id })));

                return Some(vec![self.openai_tool_call_event(
                    tool_call_id,
                    &tool_name,
                    serde_json::Value::String(input),
                    self.shell_call_provider_executed().then_some(true),
                    None,
                    provider_metadata,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                )]);
            }
            "apply_patch_call" => {
                let operation = item
                    .get("operation")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "apply_patch".to_string());

                let input = serde_json::json!({
                    "callId": tool_call_id,
                    "operation": operation,
                })
                .to_string();
                let provider_metadata = (!item_id.is_empty())
                    .then(|| self.provider_metadata_json(serde_json::json!({ "itemId": item_id })));

                return Some(vec![self.openai_tool_call_event(
                    tool_call_id,
                    &tool_name,
                    serde_json::Value::String(input),
                    None,
                    None,
                    provider_metadata,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                )]);
            }
            "local_shell_call_output" => {
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "shell".to_string());
                let result = serde_json::json!({
                    "output": item.get("output").cloned().unwrap_or_else(|| serde_json::json!("")),
                });

                return Some(vec![self.openai_tool_result_event(
                    tool_call_id,
                    &tool_name,
                    result,
                    None,
                    None,
                    None,
                    None,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                )]);
            }
            "shell_call_output" => {
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "shell".to_string());
                let output = item
                    .get("output")
                    .and_then(|value| value.as_array())
                    .map(|items| {
                        items
                            .iter()
                            .map(|entry| {
                                let stdout = entry
                                    .get("stdout")
                                    .cloned()
                                    .unwrap_or_else(|| serde_json::json!(""));
                                let stderr = entry
                                    .get("stderr")
                                    .cloned()
                                    .unwrap_or_else(|| serde_json::json!(""));
                                let outcome =
                                    entry.get("outcome").and_then(|value| value.as_object());
                                let mapped_outcome = outcome
                                    .map(|outcome| {
                                        match outcome
                                            .get("type")
                                            .and_then(|value| value.as_str())
                                            .unwrap_or("")
                                        {
                                            "timeout" => serde_json::json!({ "type": "timeout" }),
                                            "exit" => serde_json::json!({
                                                "type": "exit",
                                                "exitCode": outcome
                                                    .get("exit_code")
                                                    .or_else(|| outcome.get("exitCode"))
                                                    .cloned()
                                                    .unwrap_or_else(|| serde_json::json!(0)),
                                            }),
                                            _ => serde_json::Value::Object(outcome.clone()),
                                        }
                                    })
                                    .unwrap_or_else(
                                        || serde_json::json!({ "type": "exit", "exitCode": 0 }),
                                    );

                                serde_json::json!({
                                    "stdout": stdout,
                                    "stderr": stderr,
                                    "outcome": mapped_outcome,
                                })
                            })
                            .collect::<Vec<serde_json::Value>>()
                    })
                    .unwrap_or_default();
                let result = serde_json::json!({ "output": output });

                return Some(vec![self.openai_tool_result_event(
                    tool_call_id,
                    &tool_name,
                    result,
                    None,
                    None,
                    None,
                    None,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                )]);
            }
            "apply_patch_call_output" => {
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "apply_patch".to_string());
                let mut result = serde_json::Map::new();
                result.insert(
                    "status".to_string(),
                    item.get("status")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!("completed")),
                );
                if let Some(output) = item.get("output").cloned() {
                    result.insert("output".to_string(), output);
                }

                return Some(vec![self.openai_tool_result_event(
                    tool_call_id,
                    &tool_name,
                    serde_json::Value::Object(result),
                    None,
                    None,
                    None,
                    None,
                    OpenAiResponsesEventExtras {
                        output_index,
                        raw_item: Some(serde_json::Value::Object(item.clone())),
                    },
                )]);
            }
            "computer_call" => {
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "computer_use".to_string());
                let status = item
                    .get("status")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!("completed"));

                return Some(vec![
                    self.openai_tool_input_end_event(tool_call_id, None),
                    self.openai_tool_call_event(
                        tool_call_id,
                        &tool_name,
                        serde_json::json!(""),
                        Some(true),
                        None,
                        None,
                        OpenAiResponsesEventExtras {
                            output_index,
                            raw_item: Some(serde_json::Value::Object(item.clone())),
                        },
                    ),
                    self.openai_tool_result_event(
                        tool_call_id,
                        &tool_name,
                        serde_json::json!({
                            "type": "computer_use_tool_result",
                            "status": status,
                        }),
                        Some(true),
                        None,
                        None,
                        None,
                        OpenAiResponsesEventExtras {
                            output_index,
                            raw_item: Some(serde_json::Value::Object(item.clone())),
                        },
                    ),
                ]);
            }
            _ => return None,
        };

        let tool_name = self
            .provider_tool_name_for_item_type(item_type)
            .unwrap_or_else(|| default_tool_name.to_string());

        let mut events = vec![self.openai_tool_result_event(
            tool_call_id,
            &tool_name,
            result,
            Some(true),
            None,
            None,
            None,
            OpenAiResponsesEventExtras {
                output_index,
                raw_item: Some(serde_json::Value::Object(item.clone())),
            },
        )];

        events.extend(extra_events);
        Some(events)
    }

    pub(super) fn convert_mcp_call_arguments_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item_id = json.get("item_id").and_then(|v| v.as_str())?;
        let args = json.get("arguments").and_then(|v| v.as_str())?;
        self.record_mcp_call_args(item_id, args);
        None
    }

    pub(super) fn convert_mcp_items_from_completed(
        &self,
        json: &serde_json::Value,
    ) -> Vec<crate::streaming::ChatStreamEvent> {
        let Some(output) = json
            .get("response")
            .and_then(|r| r.get("output"))
            .and_then(|v| v.as_array())
        else {
            return Vec::new();
        };

        let mut events = Vec::new();

        for item in output {
            let Some(item_type) = item.get("type").and_then(|v| v.as_str()) else {
                continue;
            };

            match item_type {
                "mcp_call" => {
                    let Some(tool_call_id) = item.get("id").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    if self.has_emitted_mcp_result(tool_call_id) {
                        continue;
                    }

                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let server_label = item
                        .get("server_label")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let args = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let output = item
                        .get("output")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let error = item
                        .get("error")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let tool_name = format!("mcp.{name}");
                    let tool_name_for_result = tool_name.clone();

                    let mut result = serde_json::Map::new();
                    result.insert("type".to_string(), serde_json::json!("call"));
                    result.insert(
                        "serverLabel".to_string(),
                        serde_json::Value::String(server_label.to_string()),
                    );
                    result.insert(
                        "name".to_string(),
                        serde_json::Value::String(name.to_string()),
                    );
                    result.insert(
                        "arguments".to_string(),
                        serde_json::Value::String(args.to_string()),
                    );
                    if !output.is_null() {
                        result.insert("output".to_string(), output);
                    }
                    if !error.is_null() {
                        result.insert("error".to_string(), error);
                    }

                    if !self.has_emitted_mcp_call(tool_call_id) {
                        events.push(self.openai_tool_call_event(
                            tool_call_id,
                            &tool_name,
                            serde_json::json!(args),
                            Some(true),
                            Some(true),
                            None,
                            OpenAiResponsesEventExtras::default(),
                        ));
                        self.mark_mcp_call_emitted(tool_call_id);
                    }

                    events.push(self.openai_tool_result_event(
                        tool_call_id,
                        &tool_name_for_result,
                        serde_json::Value::Object(result),
                        Some(true),
                        Some(true),
                        None,
                        Some(self.provider_metadata_json(serde_json::json!({
                            "itemId": tool_call_id,
                        }))),
                        OpenAiResponsesEventExtras::default(),
                    ));
                    self.mark_mcp_result_emitted(tool_call_id);
                }
                "mcp_approval_request" => {
                    let Some(approval_id) = item.get("id").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    if self.has_emitted_mcp_approval_request(approval_id) {
                        continue;
                    }
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let tool_call_id = self.mcp_approval_tool_call_id(approval_id);
                    let tool_call_id_for_approval = tool_call_id.clone();
                    let tool_name = format!("mcp.{name}");

                    events.push(self.openai_tool_call_event(
                        &tool_call_id,
                        &tool_name,
                        serde_json::json!(args),
                        Some(true),
                        Some(true),
                        None,
                        OpenAiResponsesEventExtras::default(),
                    ));
                    events.push(self.openai_tool_approval_request_event(
                        approval_id,
                        &tool_call_id_for_approval,
                        None,
                        OpenAiResponsesEventExtras::default(),
                    ));

                    self.mark_mcp_approval_request_emitted(approval_id);
                }
                _ => {}
            }
        }

        events
    }
}
