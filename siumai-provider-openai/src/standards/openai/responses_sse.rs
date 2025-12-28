//! OpenAI Responses SSE Event Converter (protocol layer)
//!
//! This module normalizes OpenAI Responses API SSE events into Siumai's unified
//! `ChatStreamEvent` sequence. It is intentionally part of the `standards::openai`
//! protocol implementation so that providers stay thin.
//!
//! Note: Providers may re-export this converter under historical module paths
//! (e.g. `providers::openai::responses::OpenAiResponsesEventConverter`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// OpenAI Responses SSE event converter using unified streaming utilities
#[derive(Clone)]
pub struct OpenAiResponsesEventConverter {
    function_call_ids_by_output_index: Arc<Mutex<HashMap<u64, String>>>,
}

impl Default for OpenAiResponsesEventConverter {
    fn default() -> Self {
        Self {
            function_call_ids_by_output_index: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl OpenAiResponsesEventConverter {
    pub fn new() -> Self {
        Self::default()
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

    fn convert_output_text_annotation_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let annotation = json.get("annotation")?;
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
                "file_citation" => serde_json::json!({ "openai": { "fileId": file_id } }),
                "container_file_citation" => serde_json::json!({
                    "openai": {
                        "fileId": file_id,
                        "containerId": annotation.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
                        "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
                    }
                }),
                "file_path" => serde_json::json!({
                    "openai": {
                        "fileId": file_id,
                        "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
                    }
                }),
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

    fn convert_function_call_arguments_delta(
        &self,
        json: serde_json::Value,
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
            .or_else(|| json.get("item_id").and_then(|id| id.as_str()).map(|s| s.to_string()))
            .unwrap_or_default();

        Some(crate::streaming::ChatStreamEvent::ToolCallDelta {
            id,
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

    fn convert_provider_tool_output_item_added(
        &self,
        json: &serde_json::Value,
    ) -> Option<crate::streaming::ChatStreamEvent> {
        let item = json.get("item")?.as_object()?;
        let item_type = item.get("type")?.as_str()?;
        let output_index = json.get("output_index").and_then(|v| v.as_u64());

        let (tool_name, input) = match item_type {
            "web_search_call" => ("web_search", serde_json::json!("{}")),
            "file_search_call" => ("file_search", serde_json::json!("{}")),
            "computer_call" => ("computer_use", serde_json::json!("")),
            _ => return None,
        };

        let tool_call_id = item.get("id")?.as_str()?;

        Some(crate::streaming::ChatStreamEvent::Custom {
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
        })
    }

    fn convert_provider_tool_output_item_done(
        &self,
        json: &serde_json::Value,
    ) -> Option<Vec<crate::streaming::ChatStreamEvent>> {
        let item = json.get("item")?.as_object()?;
        let item_type = item.get("type")?.as_str()?;
        let output_index = json.get("output_index").and_then(|v| v.as_u64());

        let tool_call_id = item.get("id")?.as_str()?;

        let mut extra_events: Vec<crate::streaming::ChatStreamEvent> = Vec::new();

        let (tool_name, result) = match item_type {
            "web_search_call" => {
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
            "computer_call" => (
                "computer_use",
                serde_json::json!({
                    "action": item.get("action").cloned().unwrap_or(serde_json::Value::Null),
                    "status": item.get("status").cloned().unwrap_or_else(|| serde_json::json!("completed")),
                }),
            ),
            _ => return None,
        };

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

            // Parse JSON once; most Responses API SSE chunks use `data: {...}` with a `type` field.
            let json = match serde_json::from_str::<serde_json::Value>(data_raw) {
                Ok(v) => v,
                Err(e) => {
                    return vec![Err(crate::error::LlmError::ParseError(format!(
                        "Failed to parse SSE JSON: {e}"
                    )))];
                }
            };

            let chunk_type = if !event_name.is_empty() {
                event_name
            } else {
                json.get("type").and_then(|t| t.as_str()).unwrap_or("")
            };

            if chunk_type == "response.completed" {
                // The completed event often contains the full response payload.
                // Delegate to centralized ResponseTransformer for final ChatResponse.
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

            // Route by event name first
            match chunk_type {
                "response.output_text.delta"
                | "response.tool_call.delta"
                | "response.function_call.delta"
                | "response.usage" => {
                    if let Some(evt) = self.convert_responses_event(json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.output_text.annotation.added" => {
                    if let Some(evt) = self.convert_output_text_annotation_added(&json) {
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
                    if let Some(evt) = self.convert_provider_tool_output_item_added(&json) {
                        return vec![Ok(evt)];
                    }
                    if let Some(evt) = self.convert_output_item_added(json) {
                        return vec![Ok(evt)];
                    }
                }
                "response.output_item.done" => {
                    if let Some(events) = self.convert_provider_tool_output_item_done(&json)
                        && !events.is_empty()
                    {
                        return events.into_iter().map(Ok).collect();
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

        // provider tool output_item.added emits custom tool-call event
        let ev_added = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"}}"#.to_string(),
            id: "1".to_string(),
            retry: None,
        };
        let out_added = futures::executor::block_on(conv.convert_event(ev_added));
        assert_eq!(out_added.len(), 1);
        match out_added[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-call");
                assert_eq!(data["toolCallId"], serde_json::json!("ws_1"));
                assert_eq!(data["toolName"], serde_json::json!("web_search"));
                assert_eq!(data["providerExecuted"], serde_json::json!(true));
            }
            other => panic!("expected Custom tool-call, got {other:?}"),
        }

        let ev_done = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"rust"}}}"#.to_string(),
            id: "2".to_string(),
            retry: None,
        };
        let out_done = futures::executor::block_on(conv.convert_event(ev_done));
        assert_eq!(out_done.len(), 1);
        match out_done[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("ws_1"));
                assert_eq!(data["toolName"], serde_json::json!("web_search"));
                assert_eq!(data["providerExecuted"], serde_json::json!(true));
                assert_eq!(data["result"]["action"]["query"], serde_json::json!("rust"));
            }
            other => panic!("expected Custom tool-result, got {other:?}"),
        }

        // If the payload includes results, we also emit Vercel-aligned sources.
        let ev_done_with_results = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"rust"},"results":[{"url":"https://www.rust-lang.org","title":"Rust"}]}}"#.to_string(),
            id: "3".to_string(),
            retry: None,
        };
        let out_done = futures::executor::block_on(conv.convert_event(ev_done_with_results));
        assert_eq!(out_done.len(), 2);
        match out_done[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:tool-result");
                assert_eq!(data["toolCallId"], serde_json::json!("ws_1"));
            }
            other => panic!("expected Custom tool-result, got {other:?}"),
        }

        match out_done[1].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:source");
                assert_eq!(data["url"], serde_json::json!("https://www.rust-lang.org"));
                assert_eq!(data["sourceType"], serde_json::json!("url"));
            }
            other => panic!("expected Custom source, got {other:?}"),
        }
    }

    #[test]
    fn responses_output_text_annotation_added_emits_source() {
        let conv = OpenAiResponsesEventConverter::new();

        let ev = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","url":"https://www.rust-lang.org","title":"Rust","start_index":1,"end_index":2}}"#
                .to_string(),
            id: "1".to_string(),
            retry: None,
        };

        let out = futures::executor::block_on(conv.convert_event(ev));
        assert_eq!(out.len(), 1);
        match out[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:source");
                assert_eq!(data["sourceType"], serde_json::json!("url"));
                assert_eq!(data["url"], serde_json::json!("https://www.rust-lang.org"));
            }
            other => panic!("expected Custom source, got {other:?}"),
        }

        let ev = eventsource_stream::Event {
            event: "".to_string(),
            data: r#"{"type":"response.output_text.annotation.added","annotation":{"type":"file_citation","file_id":"file_123","filename":"notes.txt","quote":"Document","start_index":10,"end_index":20}}"#
                .to_string(),
            id: "2".to_string(),
            retry: None,
        };

        let out = futures::executor::block_on(conv.convert_event(ev));
        assert_eq!(out.len(), 1);
        match out[0].as_ref().unwrap() {
            crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
                assert_eq!(event_type, "openai:source");
                assert_eq!(data["sourceType"], serde_json::json!("document"));
                assert_eq!(data["url"], serde_json::json!("file_123"));
                assert_eq!(data["filename"], serde_json::json!("notes.txt"));
            }
            other => panic!("expected Custom source, got {other:?}"),
        }
    }
}
