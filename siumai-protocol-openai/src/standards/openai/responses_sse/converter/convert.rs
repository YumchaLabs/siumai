use super::*;

impl OpenAiResponsesEventConverter {
    pub(super) fn convert_responses_event(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // Handle delta as plain text or delta.content
        if let Some(delta) = json.get("delta") {
            // Case 1: delta is a plain string (response.output_text.delta)
            if let Some(s) = delta.as_str()
                && !s.is_empty()
            {
                return Some(crate::streaming::ChatStreamEvent::ContentDelta {
                    delta: s.to_string(),
                    index: None,
                });
            }
            // Case 2: delta.content is a string (message.delta simplified)
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                return Some(crate::streaming::ChatStreamEvent::ContentDelta {
                    delta: content.to_string(),
                    index: None,
                });
            }

            // Handle tool_calls delta (first item only; downstream can coalesce)
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array())
                && let Some((index, tool_call)) = tool_calls.iter().enumerate().next()
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

                return Some(crate::streaming::ChatStreamEvent::ToolCallDelta {
                    id,
                    function_name,
                    arguments_delta,
                    index: Some(index),
                });
            }
        }

        // Handle usage updates with both snake_case and camelCase fields
        if let Some(usage) = json
            .get("usage")
            .or_else(|| json.get("response")?.get("usage"))
        {
            let prompt_tokens = usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
                .or_else(|| usage.get("inputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let completion_tokens = usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
                .or_else(|| usage.get("outputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let total_tokens = usage
                .get("total_tokens")
                .or_else(|| usage.get("totalTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let reasoning_tokens = usage
                .get("reasoning_tokens")
                .or_else(|| usage.get("reasoningTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32);

            let usage_info = crate::types::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                #[allow(deprecated)]
                reasoning_tokens,
                #[allow(deprecated)]
                cached_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: reasoning_tokens.map(|r| {
                    crate::types::CompletionTokensDetails {
                        reasoning_tokens: Some(r),
                        audio_tokens: None,
                        accepted_prediction_tokens: None,
                        rejected_prediction_tokens: None,
                    }
                }),
            };
            return Some(crate::streaming::ChatStreamEvent::UsageUpdate { usage: usage_info });
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

        match self.stream_parts_style {
            StreamPartsStyle::OpenAi => Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:text-start".to_string(),
                data: serde_json::json!({
                    "type": "text-start",
                    "id": id,
                    "providerMetadata": self.provider_metadata_json(serde_json::json!({
                        "itemId": item_id,
                    })),
                }),
            }),
            StreamPartsStyle::Xai => Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:text-start".to_string(),
                data: serde_json::json!({
                    "type": "text-start",
                    "id": id,
                }),
            }),
        }
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
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if delta.is_empty() {
            return None;
        }

        let id = self.text_stream_part_id(item_id);

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if !self.has_emitted_text_start(&id) {
            self.mark_text_start_emitted(&id);
            match self.stream_parts_style {
                StreamPartsStyle::OpenAi => {
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:text-start".to_string(),
                        data: serde_json::json!({
                            "type": "text-start",
                            "id": id,
                            "providerMetadata": self.provider_metadata_json(serde_json::json!({
                                "itemId": item_id,
                            })),
                        }),
                    });
                }
                StreamPartsStyle::Xai => {
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:text-start".to_string(),
                        data: serde_json::json!({
                            "type": "text-start",
                            "id": id,
                        }),
                    });
                }
            }
        }

        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": id,
                "delta": delta,
            }),
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
            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:text-end".to_string(),
                data: serde_json::json!({
                    "type": "text-end",
                    "id": id,
                }),
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

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-end".to_string(),
            data: serde_json::json!({
                "type": "text-end",
                "id": id,
                "providerMetadata": self.provider_metadata_json(provider_metadata_openai),
            }),
        })
    }

    pub(super) fn convert_finish_event(
        &self,
        completed_json: &serde_json::Value,
        response: &crate::types::ChatResponse,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let usage = completed_json
            .get("response")
            .and_then(|r| r.get("usage"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let input_tokens = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cached_tokens = usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let reasoning_tokens = usage
            .get("output_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let input_cache_read = cached_tokens.min(input_tokens);
        let input_no_cache = input_tokens.saturating_sub(input_cache_read);
        let output_reasoning = reasoning_tokens.min(output_tokens);
        let output_text = output_tokens.saturating_sub(output_reasoning);

        let unified = response.finish_reason.as_ref().map(|r| match r {
            crate::types::FinishReason::Stop => "stop".to_string(),
            crate::types::FinishReason::StopSequence => "stop".to_string(),
            crate::types::FinishReason::Length => "length".to_string(),
            crate::types::FinishReason::ToolCalls => "tool-calls".to_string(),
            crate::types::FinishReason::ContentFilter => "content-filter".to_string(),
            crate::types::FinishReason::Error => "error".to_string(),
            crate::types::FinishReason::Unknown => "unknown".to_string(),
            crate::types::FinishReason::Other(s) => s.clone(),
        });

        if self.stream_parts_style == StreamPartsStyle::Xai {
            let raw_finish = completed_json
                .get("response")
                .and_then(|r| r.get("status"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "completed".to_string());

            let input_tokens_obj = serde_json::json!({
                "total": input_tokens,
                "cacheRead": input_cache_read,
                "noCache": input_no_cache,
            });
            let output_tokens_obj = serde_json::json!({
                "total": output_tokens,
                "reasoning": output_reasoning,
                "text": output_text,
            });

            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:finish".to_string(),
                data: serde_json::json!({
                    "type": "finish",
                    "finishReason": {
                        "raw": raw_finish,
                        "unified": unified,
                    },
                    "usage": {
                        "inputTokens": input_tokens_obj,
                        "outputTokens": output_tokens_obj,
                        "raw": usage,
                    },
                }),
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
            .map(|s| serde_json::Value::String(s.to_string()))
            .unwrap_or(serde_json::Value::Null);

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

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:finish".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "finishReason": {
                    "raw": raw_finish_reason,
                    "unified": unified,
                },
                "providerMetadata": self.provider_metadata_json(provider_metadata),
                "usage": {
                    "inputTokens": {
                        "total": input_tokens,
                        "cacheRead": input_cache_read,
                        "cacheWrite": serde_json::Value::Null,
                        "noCache": input_no_cache,
                    },
                    "outputTokens": {
                        "total": output_tokens,
                        "reasoning": output_reasoning,
                        "text": output_text,
                    },
                    "raw": usage,
                },
            }),
        })
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

        match self.stream_parts_style {
            StreamPartsStyle::OpenAi => Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-start".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-start",
                    "id": id,
                    "providerMetadata": self.provider_metadata_json(serde_json::json!({
                        "itemId": item_id,
                        // Vercel alignment: always include the key for start events.
                        "reasoningEncryptedContent": encrypted_content,
                    })),
                }),
            }),
            StreamPartsStyle::Xai => Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-start".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-start",
                    "id": id,
                }),
            }),
        }
    }

    pub(super) fn convert_reasoning_summary_part_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
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
            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-start".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-start",
                    "id": id,
                }),
            });
        }
        let summary_index = json
            .get("summary_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let id = format!("{item_id}:{summary_index}");
        if self.has_emitted_reasoning_start(&id) {
            return None;
        }
        self.mark_reasoning_start_emitted(&id);

        let encrypted_content = self.reasoning_encrypted_content(item_id);

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:reasoning-start".to_string(),
            data: serde_json::json!({
                "type": "reasoning-start",
                "id": id,
                "providerMetadata": self.provider_metadata_json(serde_json::json!({
                    "itemId": item_id,
                    // Vercel alignment: always include the key for start events.
                    "reasoningEncryptedContent": encrypted_content,
                })),
            }),
        })
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
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if delta.is_empty() {
            return None;
        }

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
                }

                Some(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:reasoning-delta".to_string(),
                    data: serde_json::json!({
                        "type": "reasoning-delta",
                        "id": id,
                        "delta": delta,
                        "providerMetadata": self.provider_metadata_json(serde_json::json!({
                            "itemId": item_id,
                        })),
                    }),
                })
            }
            StreamPartsStyle::Xai => {
                let id = self.reasoning_stream_part_id(item_id);
                if !self.has_emitted_reasoning_start(&id) {
                    self.mark_reasoning_start_emitted(&id);
                }

                Some(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:reasoning-delta".to_string(),
                    data: serde_json::json!({
                        "type": "reasoning-delta",
                        "id": id,
                        "delta": delta,
                    }),
                })
            }
        }
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
            self.mark_reasoning_end_emitted(&id);

            return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-end".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-end",
                    "id": id,
                }),
            }]);
        }

        let summary_len = item
            .get("summary")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let blocks = std::cmp::max(1, summary_len);

        let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();
        for i in 0..blocks {
            let id = format!("{item_id}:{i}");
            if self.has_emitted_reasoning_end(&id) {
                continue;
            }
            self.mark_reasoning_end_emitted(&id);

            // Vercel alignment: omit reasoningEncryptedContent when it is null/absent.
            let provider_metadata = if let Some(enc) = encrypted_content.as_ref() {
                self.provider_metadata_json(serde_json::json!({
                    "itemId": item_id,
                    "reasoningEncryptedContent": enc,
                }))
            } else {
                self.provider_metadata_json(serde_json::json!({
                    "itemId": item_id,
                }))
            };

            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:reasoning-end".to_string(),
                data: serde_json::json!({
                    "type": "reasoning-end",
                    "id": id,
                    "providerMetadata": provider_metadata,
                }),
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

            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:source".to_string(),
                data: serde_json::json!({
                    "type": "source",
                    "sourceType": "url",
                    "id": id,
                    "url": url,
                    "title": title,
                }),
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
            let id = start_index
                .map(|s| format!("ann:doc:{s}"))
                .unwrap_or_else(|| format!("ann:doc:{file_id}"));

            let provider_metadata = match ann_type {
                "file_citation" => {
                    self.provider_metadata_json(serde_json::json!({ "fileId": file_id }))
                }
                "container_file_citation" => self.provider_metadata_json(serde_json::json!({
                    "fileId": file_id,
                    "containerId": annotation.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
                    "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
                })),
                "file_path" => self.provider_metadata_json(serde_json::json!({
                    "fileId": file_id,
                    "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
                })),
                _ => serde_json::Value::Null,
            };

            return Some(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:source".to_string(),
                data: serde_json::json!({
                    "type": "source",
                    "sourceType": "document",
                    "id": id,
                    "url": file_id,
                    "title": title,
                    "mediaType": media_type,
                    "filename": filename,
                    "providerMetadata": provider_metadata,
                }),
            });
        }

        None
    }

    pub(super) fn convert_function_call_arguments_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // Handle response.function_call_arguments.delta events
        let delta = json.get("delta").and_then(|d| d.as_str())?;
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0);

        let id = self
            .function_call_ids_by_output_index
            .lock()
            .ok()
            .and_then(|map| map.get(&output_index).cloned())
            .or_else(|| {
                json.get("item_id")
                    .and_then(|id| id.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_default();

        Some(crate::streaming::ChatStreamEvent::ToolCallDelta {
            id,
            function_name: None, // Function name is set in the initial item.added event
            arguments_delta: Some(delta.to_string()),
            index: Some(output_index as usize),
        })
    }

    pub(super) fn convert_output_item_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // Handle response.output_item.added events for function calls
        let item = json.get("item")?;
        if item.get("type").and_then(|t| t.as_str()) != Some("function_call") {
            return None;
        }

        let id = item.get("call_id").and_then(|id| id.as_str()).unwrap_or("");
        let function_name = item.get("name").and_then(|name| name.as_str());
        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0);

        if !item_id.is_empty()
            && !id.is_empty()
            && let Some(name) = function_name
        {
            self.record_function_call_meta(item_id, id, name);
        }

        if !id.is_empty()
            && let Ok(mut map) = self.function_call_ids_by_output_index.lock()
        {
            map.insert(output_index, id.to_string());
        }

        Some(crate::streaming::ChatStreamEvent::ToolCallDelta {
            id: id.to_string(),
            function_name: function_name.map(|s| s.to_string()),
            arguments_delta: None, // Arguments will come in subsequent delta events
            index: Some(output_index as usize),
        })
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

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-start".to_string(),
            data: serde_json::json!({
                "type": "tool-input-start",
                "id": call_id,
                "toolName": tool_name,
            }),
        })
    }

    pub(super) fn convert_function_call_arguments_delta_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let delta = json.get("delta").and_then(|d| d.as_str())?;
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");

        let (call_id, _tool_name) = self.function_call_meta(item_id)?;
        if call_id.is_empty() || delta.is_empty() {
            return None;
        }

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": call_id,
                "delta": delta,
            }),
        })
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
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-end".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-end",
                    "id": call_id,
                }),
            });
        }

        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": call_id,
                "toolName": tool_name,
                "input": args,
                "providerMetadata": self.provider_metadata_json(serde_json::json!({
                    "itemId": item_id,
                })),
            }),
        });

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
            vec![crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-start".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-start",
                    "id": call_id,
                    "toolName": tool_name,
                }),
            }];

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

            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-delta".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-delta",
                    "id": call_id,
                    "delta": input,
                }),
            });

            if self.mark_apply_patch_tool_input_end_emitted(call_id) {
                events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-input-end".to_string(),
                    data: serde_json::json!({
                        "type": "tool-input-end",
                        "id": call_id,
                    }),
                });
            }

            return Some(events);
        }

        let prefix = format!(
            "{{\"callId\":{call_id_json},\"operation\":{{\"type\":{op_type_json},\"path\":{path_json},\"diff\":\""
        );
        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": call_id,
                "delta": prefix,
            }),
        });

        Some(events)
    }

    pub(super) fn convert_apply_patch_operation_diff_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() || delta.is_empty() {
            return None;
        }
        let call_id = self.apply_patch_call_id(item_id)?;
        if call_id.is_empty() {
            return None;
        }
        if !self.has_emitted_apply_patch_tool_input_start(call_id.as_str()) {
            return None;
        }

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": call_id,
                "delta": delta,
            }),
        })
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

        // Close the open `diff` string and the surrounding objects: `"}}`
        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": call_id,
                "delta": "\"}}",
            }),
        });

        if self.mark_apply_patch_tool_input_end_emitted(call_id.as_str()) {
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-end".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-end",
                    "id": call_id,
                }),
            });
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
            crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-start".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-start",
                    "id": item_id,
                    "toolName": tool_name,
                    "providerExecuted": true,
                }),
            },
            crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-delta".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-delta",
                    "id": item_id,
                    "delta": prefix,
                }),
            },
        ])
    }

    pub(super) fn convert_code_interpreter_code_delta_tool_input(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item_id = json.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
        let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() || delta.is_empty() {
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

            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-start".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-start",
                    "id": item_id,
                    "toolName": tool_name,
                    "providerExecuted": true,
                }),
            });
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-delta".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-delta",
                    "id": item_id,
                    "delta": prefix,
                }),
            });
        }

        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": item_id,
                "delta": delta,
            }),
        });

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
        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": item_id,
                "delta": "\"}",
            }),
        });

        if self.mark_code_interpreter_tool_input_end_emitted(item_id) {
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-end".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-end",
                    "id": item_id,
                }),
            });
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
                // MCP tool calls stream arguments separately. Record metadata here,
                // emit tool-call when arguments are available.
                let item_id = item.get("id")?.as_str()?;
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let server_label = item
                    .get("server_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                self.record_mcp_call_added(item_id, name, server_label);
                return None;
            }
            "mcp_approval_request" => {
                // Vercel alignment: represent approval request as a dynamic tool-call
                // followed by a tool-approval-request (emitted on output_item.done).
                let approval_id = item.get("id")?.as_str()?;
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let tool_call_id = self.mcp_approval_tool_call_id(approval_id);
                let tool_name = format!("mcp.{name}");

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "dynamic": true,
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "input": args,
                        "providerExecuted": true,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
            }
            "custom_tool_call" => {
                // xAI x_search emits internal custom tool calls (e.g. `x_keyword_search`) and streams
                // their input via `response.custom_tool_call_input.*` events.
                let tool_call_id = item.get("id")?.as_str()?;
                let call_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let tool_name = self.custom_tool_name_for_call_name(call_name);
                self.record_custom_tool_item(tool_call_id, call_name, &tool_name);

                if self.mark_custom_tool_input_start_emitted(tool_call_id) {
                    return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-input-start".to_string(),
                        data: serde_json::json!({
                            "type": "tool-input-start",
                            "id": tool_call_id,
                            "toolName": tool_name,
                        }),
                    }]);
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
            "file_search_call" => ("file_search", serde_json::json!("{}")),
            "computer_call" => ("computer_use", serde_json::json!("")),
            "code_interpreter_call" => {
                let container_id = item.get("container_id").and_then(|v| v.as_str());
                let tool_call_id = item.get("id")?.as_str()?;
                if let Some(cid) = container_id {
                    self.record_code_interpreter_container_id(tool_call_id, cid);
                }

                // Vercel alignment: code interpreter tool-call is emitted after tool-input-end,
                // once the full code is known (at output_item.done).
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

        // Vercel alignment: webSearch emits tool-input-start/end even with empty input.
        if item_type == "web_search_call" && self.mark_web_search_tool_input_emitted(tool_call_id) {
            let mut data = serde_json::json!({
                "type": "tool-input-start",
                "id": tool_call_id,
                "toolName": tool_name,
            });
            if self.include_web_search_provider_executed_in_tool_input {
                data["providerExecuted"] = serde_json::json!(true);
            }
            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-start".to_string(),
                data,
            });

            if self.emit_web_search_tool_input_delta
                && let serde_json::Value::String(delta) = &input
            {
                events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-input-delta".to_string(),
                    data: serde_json::json!({
                        "type": "tool-input-delta",
                        "id": tool_call_id,
                        "delta": delta,
                    }),
                });
            }

            events.push(crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-input-end".to_string(),
                data: serde_json::json!({
                    "type": "tool-input-end",
                    "id": tool_call_id,
                }),
            });
        }

        events.push(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "input": input,
                "providerExecuted": true,
                "outputIndex": output_index,
                "rawItem": serde_json::Value::Object(item.clone()),
            }),
        });

        Some(events)
    }

    pub(super) fn convert_provider_tool_output_item_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        let item_type = item.get("type")?.as_str()?;
        let output_index = json.get("output_index").and_then(|v| v.as_u64());

        let tool_call_id = item.get("id")?.as_str()?;

        let mut extra_events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        if item_type == "custom_tool_call" {
            if !self.mark_custom_tool_call_emitted(tool_call_id) {
                return None;
            }

            let call_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let tool_name = self.custom_tool_name_for_call_name(call_name);
            let input = item.get("input").and_then(|v| v.as_str()).unwrap_or("");

            let mut events = vec![crate::streaming::ChatStreamEvent::Custom {
                event_type: "openai:tool-call".to_string(),
                data: serde_json::json!({
                    "type": "tool-call",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "input": input,
                    "providerExecuted": true,
                    "outputIndex": output_index,
                    "rawItem": serde_json::Value::Object(item.clone()),
                }),
            }];

            if let Some(output) = item.get("output") {
                let mut tool_result = serde_json::json!({
                    "type": "tool-result",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "result": output,
                    "providerExecuted": true,
                    "outputIndex": output_index,
                    "providerMetadata": self.provider_metadata_json(serde_json::json!({
                        "itemId": tool_call_id,
                    })),
                    "rawItem": serde_json::Value::Object(item.clone()),
                });

                if let Some(is_error) = item.get("is_error").and_then(|v| v.as_bool())
                    && let Some(obj) = tool_result.as_object_mut()
                {
                    obj.insert("isError".to_string(), serde_json::Value::Bool(is_error));
                }

                events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-result".to_string(),
                    data: tool_result,
                });
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
                extra_events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-approval-request".to_string(),
                    data: serde_json::json!({
                        "type": "tool-approval-request",
                        "approvalId": approval_id,
                        "toolCallId": tool_call_id,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                });

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

                let tool_name = format!("mcp.{name}");
                let tool_name_for_result = tool_name.clone();
                let args_for_result = args.clone();

                if !self.has_emitted_mcp_call(tool_call_id) {
                    extra_events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-call".to_string(),
                        data: serde_json::json!({
                            "type": "tool-call",
                            "dynamic": true,
                            "toolCallId": tool_call_id,
                            "toolName": tool_name,
                            "input": args,
                            "providerExecuted": true,
                            "outputIndex": output_index,
                            "rawItem": serde_json::Value::Object(item.clone()),
                        }),
                    });
                    self.mark_mcp_call_emitted(tool_call_id);
                }

                if self.has_emitted_mcp_result(tool_call_id) {
                    return Some(extra_events);
                }
                self.mark_mcp_result_emitted(tool_call_id);

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-result".to_string(),
                    data: serde_json::json!({
                        "type": "tool-result",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name_for_result,
                        "result": {
                            "type": "call",
                            "serverLabel": server_label,
                            "name": name,
                            "arguments": args_for_result,
                            "output": output,
                        },
                        "providerExecuted": true,
                        "outputIndex": output_index,
                        "providerMetadata": self.provider_metadata_json(serde_json::json!({
                            "itemId": tool_call_id,
                        })),
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
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

                // Include results if present (align with non-streaming transformer).
                let results = item
                    .get("results")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

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

                        extra_events.push(crate::streaming::ChatStreamEvent::Custom {
                            event_type: "openai:source".to_string(),
                            data: serde_json::json!({
                                "type": "source",
                                "sourceType": "url",
                                "id": format!("{tool_call_id}:{i}"),
                                "url": url,
                                "title": title,
                                "toolCallId": tool_call_id,
                            }),
                        });
                    }
                }

                (
                    "web_search",
                    serde_json::json!({
                        "action": item.get("action").cloned().unwrap_or(serde_json::Value::Null),
                        "results": results,
                        "status": item.get("status").cloned().unwrap_or_else(|| serde_json::json!("completed")),
                    }),
                )
            }
            "file_search_call" => (
                "file_search",
                serde_json::json!({
                    "results": item.get("results").cloned().unwrap_or(serde_json::Value::Null),
                    "status": item.get("status").cloned().unwrap_or_else(|| serde_json::json!("completed")),
                }),
            ),
            "code_interpreter_call" => {
                // Vercel alignment: codeExecution streams tool input, then emits tool-call, then tool-result.
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "code_interpreter".to_string());

                let mut events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

                if !self.has_emitted_code_interpreter_tool_input_end(tool_call_id)
                    && self.mark_code_interpreter_tool_input_end_emitted(tool_call_id)
                {
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-input-delta".to_string(),
                        data: serde_json::json!({
                            "type": "tool-input-delta",
                            "id": tool_call_id,
                            "delta": "\"}",
                        }),
                    });
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-input-end".to_string(),
                        data: serde_json::json!({
                            "type": "tool-input-end",
                            "id": tool_call_id,
                        }),
                    });
                }

                let container_id = item.get("container_id").and_then(|v| v.as_str());
                let code = item.get("code").and_then(|v| v.as_str()).unwrap_or("");

                let code_json = serde_json::to_string(code).unwrap_or_else(|_| "\"\"".to_string());
                let container_id_json = container_id
                    .and_then(|cid| serde_json::to_string(cid).ok())
                    .unwrap_or_else(|| "null".to_string());

                let input = format!("{{\"code\":{code_json},\"containerId\":{container_id_json}}}");

                events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "input": input,
                        "providerExecuted": true,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                });

                events.push(crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-result".to_string(),
                    data: serde_json::json!({
                        "type": "tool-result",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "result": {
                            "outputs": item.get("outputs").cloned().unwrap_or_else(|| serde_json::json!([])),
                        },
                        "providerExecuted": true,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                });

                return Some(events);
            }
            "image_generation_call" => (
                "image_generation",
                serde_json::json!({
                    "result": item.get("result").cloned().unwrap_or(serde_json::Value::Null),
                }),
            ),
            "local_shell_call" => {
                let call_id = item.get("call_id").and_then(|v| v.as_str())?;
                let action = item
                    .get("action")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "shell".to_string());

                let input = serde_json::json!({ "action": action }).to_string();

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "toolCallId": call_id,
                        "toolName": tool_name,
                        "input": input,
                        "providerExecuted": false,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
            }
            "shell_call" => {
                let call_id = item.get("call_id").and_then(|v| v.as_str())?;
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

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "toolCallId": call_id,
                        "toolName": tool_name,
                        "input": input,
                        "providerExecuted": false,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
            }
            "apply_patch_call" => {
                let call_id = item.get("call_id").and_then(|v| v.as_str())?;
                let operation = item
                    .get("operation")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let tool_name = self
                    .provider_tool_name_for_item_type(item_type)
                    .unwrap_or_else(|| "apply_patch".to_string());

                let input = serde_json::json!({
                    "callId": call_id,
                    "operation": operation,
                })
                .to_string();

                return Some(vec![crate::streaming::ChatStreamEvent::Custom {
                    event_type: "openai:tool-call".to_string(),
                    data: serde_json::json!({
                        "type": "tool-call",
                        "toolCallId": call_id,
                        "toolName": tool_name,
                        "input": input,
                        "providerExecuted": false,
                        "outputIndex": output_index,
                        "rawItem": serde_json::Value::Object(item.clone()),
                    }),
                }]);
            }
            "computer_call" => (
                "computer_use",
                serde_json::json!({
                    "action": item.get("action").cloned().unwrap_or(serde_json::Value::Null),
                    "status": item.get("status").cloned().unwrap_or_else(|| serde_json::json!("completed")),
                }),
            ),
            _ => return None,
        };

        let tool_name = self
            .provider_tool_name_for_item_type(item_type)
            .unwrap_or_else(|| default_tool_name.to_string());

        let mut events = vec![crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "result": result,
                "providerExecuted": true,
                "outputIndex": output_index,
                "rawItem": serde_json::Value::Object(item.clone()),
            }),
        }];

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

        if self.has_emitted_mcp_call(item_id) {
            return None;
        }

        let (name, _server_label) = self.mcp_call_meta(item_id)?;
        self.mark_mcp_call_emitted(item_id);

        let tool_name = format!("mcp.{name}");
        let output_index = json.get("output_index").and_then(|v| v.as_u64());

        Some(crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "dynamic": true,
                "toolCallId": item_id,
                "toolName": tool_name,
                "input": args,
                "providerExecuted": true,
                "outputIndex": output_index,
            }),
        })
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
                    let tool_name = format!("mcp.{name}");
                    let tool_name_for_result = tool_name.clone();

                    if !self.has_emitted_mcp_call(tool_call_id) {
                        events.push(crate::streaming::ChatStreamEvent::Custom {
                            event_type: "openai:tool-call".to_string(),
                            data: serde_json::json!({
                                "type": "tool-call",
                                "dynamic": true,
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                                "input": args,
                                "providerExecuted": true,
                            }),
                        });
                        self.mark_mcp_call_emitted(tool_call_id);
                    }

                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-result".to_string(),
                        data: serde_json::json!({
                            "type": "tool-result",
                            "toolCallId": tool_call_id,
                            "toolName": tool_name_for_result,
                            "result": {
                                "type": "call",
                                "serverLabel": server_label,
                                "name": name,
                                "arguments": args,
                                "output": output,
                            },
                            "providerExecuted": true,
                            "providerMetadata": self.provider_metadata_json(serde_json::json!({
                                "itemId": tool_call_id,
                            })),
                        }),
                    });
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

                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-call".to_string(),
                        data: serde_json::json!({
                            "type": "tool-call",
                            "dynamic": true,
                            "toolCallId": tool_call_id,
                            "toolName": tool_name,
                            "input": args,
                            "providerExecuted": true,
                        }),
                    });
                    events.push(crate::streaming::ChatStreamEvent::Custom {
                        event_type: "openai:tool-approval-request".to_string(),
                        data: serde_json::json!({
                            "type": "tool-approval-request",
                            "approvalId": approval_id,
                            "toolCallId": tool_call_id_for_approval,
                        }),
                    });

                    self.mark_mcp_approval_request_emitted(approval_id);
                }
                _ => {}
            }
        }

        events
    }
}
