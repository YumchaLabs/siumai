//! OpenAI Responses SSE Event Converter
//!
//! This module keeps only the SSE event converter for the OpenAI Responses API.
//! All request/response building and HTTP execution have been migrated to
//! Transformers + Executors. The converter normalizes provider-specific SSE
//! events into Siumai's unified `ChatStreamEvent` sequence.

/// OpenAI Responses SSE event converter using unified streaming utilities
#[derive(Clone)]
pub struct OpenAiResponsesEventConverter;

impl Default for OpenAiResponsesEventConverter {
    fn default() -> Self {
        Self
    }
}

impl OpenAiResponsesEventConverter {
    pub fn new() -> Self {
        Self
    }

    fn convert_responses_event(
        &self,
        json: serde_json::Value,
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

    fn convert_function_call_arguments_delta(
        &self,
        json: serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // Handle response.function_call_arguments.delta events
        let delta = json.get("delta").and_then(|d| d.as_str())?;
        let item_id = json.get("item_id").and_then(|id| id.as_str()).unwrap_or("");
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0);

        Some(crate::streaming::ChatStreamEvent::ToolCallDelta {
            id: item_id.to_string(),
            function_name: None, // Function name is set in the initial item.added event
            arguments_delta: Some(delta.to_string()),
            index: Some(output_index as usize),
        })
    }

    fn convert_output_item_added(
        &self,
        json: serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        // Handle response.output_item.added events for function calls
        let item = json.get("item")?;
        if item.get("type").and_then(|t| t.as_str()) != Some("function_call") {
            return None;
        }

        let id = item.get("call_id").and_then(|id| id.as_str()).unwrap_or("");
        let function_name = item.get("name").and_then(|name| name.as_str());
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0);

        Some(crate::streaming::ChatStreamEvent::ToolCallDelta {
            id: id.to_string(),
            function_name: function_name.map(|s| s.to_string()),
            arguments_delta: None, // Arguments will come in subsequent delta events
            index: Some(output_index as usize),
        })
    }
}

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
            if event_name == "response.completed" {
                // The completed event often contains the full response payload.
                // Delegate to centralized ResponseTransformer for final ChatResponse.
                let json = match serde_json::from_str::<serde_json::Value>(data_raw) {
                    Ok(v) => v,
                    Err(e) => {
                        return vec![Err(crate::error::LlmError::ParseError(format!(
                            "Failed to parse completed event JSON: {e}"
                        )))];
                    }
                };

                let resp_tx = super::transformers::OpenAiResponsesResponseTransformer;
                match crate::execution::transformers::response::ResponseTransformer::transform_chat_response(
                    &resp_tx, &json,
                ) {
                    Ok(response) => {
                        return vec![Ok(crate::streaming::ChatStreamEvent::StreamEnd {
                            response,
                        })];
                    }
                    Err(err) => return vec![Err(err)],
                }
            }

            // Parse JSON (fallback)
            let json = match serde_json::from_str::<serde_json::Value>(data_raw) {
                Ok(v) => v,
                Err(e) => {
                    return vec![Err(crate::error::LlmError::ParseError(format!(
                        "Failed to parse SSE JSON: {e}"
                    )))];
                }
            };

            // Route by event name first
            match event_name {
                "response.output_text.delta"
                | "response.tool_call.delta"
                | "response.function_call.delta"
                | "response.usage" => {
                    if let Some(evt) = self.convert_responses_event(json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.error" => {
                    // Normalize provider error into ChatStreamEvent::Error
                    let msg = json
                        .get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("Unknown error")
                        .to_string();
                    return vec![Ok(crate::streaming::ChatStreamEvent::Error { error: msg })];
                }
                "response.function_call_arguments.delta" => {
                    if let Some(evt) = self.convert_function_call_arguments_delta(json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.output_item.added" => {
                    if let Some(evt) = self.convert_output_item_added(json) {
                        return vec![Ok(evt)];
                    }
                }
                _ => {
                    if let Some(evt) = self.convert_responses_event(json) {
                        return vec![Ok(evt)];
                    }
                }
            }

            vec![]
        })
    }

    fn handle_stream_end(
        &self,
    ) -> Option<Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>> {
        // Do not emit a StreamEnd on [DONE]. The Responses API emits a
        // `response.completed` event that we already convert into the final
        // ChatResponse via the centralized ResponseTransformer. Returning None
        // here avoids duplicate StreamEnd events and matches Cherry's behavior.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::SseEventConverter;

    #[test]
    fn test_responses_event_converter_content_delta() {
        let conv = OpenAiResponsesEventConverter::new();
        let event = eventsource_stream::Event {
            event: "message".to_string(),
            data: r#"{"delta":{"content":"hello"}}"#.to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let fut = conv.convert_event(event);
        let events = futures::executor::block_on(fut);
        assert!(!events.is_empty());
        let ev = events.first().unwrap().as_ref().unwrap();
        match ev {
            crate::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
                assert_eq!(delta, "hello")
            }
            _ => panic!("expected ContentDelta"),
        }
    }

    #[test]
    fn test_responses_event_converter_tool_call_delta() {
        let conv = OpenAiResponsesEventConverter::new();
        let event = eventsource_stream::Event {
            event: "message".to_string(),
            data: r#"{"delta":{"tool_calls":[{"id":"t1","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]}}"#.to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let fut = conv.convert_event(event);
        let events = futures::executor::block_on(fut);
        assert!(!events.is_empty());
        let ev = events.first().unwrap().as_ref().unwrap();
        match ev {
            crate::streaming::ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            } => {
                assert_eq!(id, "t1");
                assert_eq!(function_name.clone().unwrap(), "lookup");
                assert_eq!(arguments_delta.clone().unwrap(), "{\"q\":\"x\"}");
            }
            _ => panic!("expected ToolCallDelta"),
        }
    }

    #[test]
    fn test_responses_event_converter_usage_update() {
        let conv = OpenAiResponsesEventConverter::new();
        let event = eventsource_stream::Event {
            event: "message".to_string(),
            data: r#"{"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}"#
                .to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let fut = conv.convert_event(event);
        let events = futures::executor::block_on(fut);
        assert!(!events.is_empty());
        let ev = events.first().unwrap().as_ref().unwrap();
        match ev {
            crate::streaming::ChatStreamEvent::UsageUpdate { usage } => {
                assert_eq!(usage.prompt_tokens, 3);
                assert_eq!(usage.completion_tokens, 5);
                assert_eq!(usage.total_tokens, 8);
            }
            _ => panic!("expected UsageUpdate"),
        }
    }

    #[test]
    fn test_responses_event_converter_done() {
        let conv = OpenAiResponsesEventConverter::new();
        let event = eventsource_stream::Event {
            event: "message".to_string(),
            data: "[DONE]".to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let fut = conv.convert_event(event);
        let events = futures::executor::block_on(fut);
        // [DONE] events should not generate any events in our new architecture
        assert!(events.is_empty());
    }

    #[test]
    fn test_sse_named_events_routing() {
        let conv = OpenAiResponsesEventConverter::new();
        use crate::streaming::SseEventConverter;

        // content delta via named event
        let ev1 = eventsource_stream::Event {
            event: "response.output_text.delta".to_string(),
            data: r#"{"delta":{"content":"abc"}}"#.to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let events1 = futures::executor::block_on(conv.convert_event(ev1));
        assert!(!events1.is_empty());
        let out1 = events1.first().unwrap().as_ref().unwrap();
        match out1 {
            crate::streaming::ChatStreamEvent::ContentDelta { delta, .. } => {
                assert_eq!(delta, "abc")
            }
            _ => panic!("expected ContentDelta"),
        }

        // tool call delta via named event
        let ev2 = eventsource_stream::Event {
            event: "response.tool_call.delta".to_string(),
            data: r#"{"delta":{"tool_calls":[{"id":"t1","function":{"name":"fn","arguments":"{}"}}]}}"#.to_string(),
            id: "2".to_string(),
            retry: None,
        };
        let events2 = futures::executor::block_on(conv.convert_event(ev2));
        assert!(!events2.is_empty());
        let out2 = events2.first().unwrap().as_ref().unwrap();
        match out2 {
            crate::streaming::ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            } => {
                assert_eq!(id, "t1");
                assert_eq!(function_name.clone().unwrap(), "fn");
                assert_eq!(arguments_delta.clone().unwrap(), "{}");
            }
            _ => panic!("expected ToolCallDelta"),
        }

        // usage via named event camelCase
        let ev3 = eventsource_stream::Event {
            event: "response.usage".to_string(),
            data: r#"{"usage":{"inputTokens":4,"outputTokens":6,"totalTokens":10}}"#.to_string(),
            id: "3".to_string(),
            retry: None,
        };
        let events3 = futures::executor::block_on(conv.convert_event(ev3));
        assert!(!events3.is_empty());
        let out3 = events3.first().unwrap().as_ref().unwrap();
        match out3 {
            crate::streaming::ChatStreamEvent::UsageUpdate { usage } => {
                assert_eq!(usage.prompt_tokens, 4);
                assert_eq!(usage.completion_tokens, 6);
                assert_eq!(usage.total_tokens, 10);
            }
            _ => panic!("expected UsageUpdate"),
        }

        // completed
        let ev4 = eventsource_stream::Event {
            event: "response.completed".to_string(),
            data: "{}".to_string(),
            id: "4".to_string(),
            retry: None,
        };
        let events4 = futures::executor::block_on(conv.convert_event(ev4));
        assert!(!events4.is_empty());
        let out4 = events4.first().unwrap().as_ref().unwrap();
        match out4 {
            crate::streaming::ChatStreamEvent::StreamEnd { .. } => {}
            _ => panic!("expected StreamEnd"),
        }
    }
}
