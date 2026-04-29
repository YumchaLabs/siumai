use super::*;
use crate::streaming::SseEventConverter;
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};

fn parse_sse_frames(bytes: &[u8]) -> Vec<(String, serde_json::Value)> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|chunk| {
            let mut event_name: Option<String> = None;
            let mut data_line: Option<String> = None;
            for line in chunk.lines() {
                if let Some(v) = line.strip_prefix("event: ") {
                    event_name = Some(v.trim().to_string());
                } else if let Some(v) = line.strip_prefix("data: ") {
                    data_line = Some(v.trim().to_string());
                }
            }
            let event_name = event_name?;
            let data_line = data_line?;
            if data_line == "[DONE]" {
                return None;
            }
            let json = serde_json::from_str::<serde_json::Value>(&data_line).ok()?;
            Some((event_name, json))
        })
        .collect()
}

fn stream_part(
    event: &Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>,
) -> Option<crate::streaming::LanguageModelV3StreamPart> {
    crate::streaming::LanguageModelV3StreamPart::try_from_chat_event(event.as_ref().ok()?)
}

fn append_tool_input_deltas(
    events: &[Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>],
    out: &mut String,
) {
    for event in events {
        if let Some(crate::streaming::LanguageModelV3StreamPart::ToolInputDelta { delta, .. }) =
            stream_part(event)
        {
            out.push_str(&delta);
        }
    }
}

fn openai_responses_raw_item(
    event: &Result<crate::streaming::ChatStreamEvent, crate::error::LlmError>,
) -> Option<&serde_json::Value> {
    event
        .as_ref()
        .ok()?
        .replay_ref()?
        .openai_responses_ref()?
        .raw_item
        .as_ref()
}

fn openai_provider_metadata(value: serde_json::Value) -> crate::types::StreamProviderMetadata {
    let mut provider_metadata = crate::types::StreamProviderMetadata::new();
    provider_metadata.insert("openai".to_string(), value);
    provider_metadata
}

fn parse_test_timestamp(timestamp: &str) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .expect("rfc3339 timestamp")
        .with_timezone(&chrono::Utc)
}

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
        data: r#"{"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens_details":{"reasoning_tokens":2}}}"#.to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let fut = conv.convert_event(event);
    let events = futures::executor::block_on(fut);
    assert!(!events.is_empty());
    let ev = events.first().unwrap().as_ref().unwrap();
    match ev {
        crate::streaming::ChatStreamEvent::UsageUpdate { usage } => {
            assert_eq!(usage.prompt_tokens(), Some(3));
            assert_eq!(usage.completion_tokens(), Some(5));
            assert_eq!(usage.total_tokens(), Some(8));
            assert_eq!(usage.normalized_input_tokens().cache_read, Some(1));
            assert_eq!(usage.normalized_input_tokens().no_cache, Some(2));
            assert_eq!(usage.normalized_output_tokens().reasoning, Some(2));
            assert_eq!(usage.normalized_output_tokens().text, Some(3));
            assert_eq!(
                usage.raw_usage_value().expect("raw usage")["prompt_tokens"],
                serde_json::json!(3)
            );
        }
        _ => panic!("expected UsageUpdate"),
    }
}

#[test]
fn xai_responses_event_converter_usage_update_uses_xai_semantics() {
    let conv = OpenAiResponsesEventConverter::new().with_responses_transform_style(
        crate::standards::openai::transformers::ResponsesTransformStyle::Xai,
    );
    let event = eventsource_stream::Event {
        event: "response.usage".to_string(),
        data: r#"{"usage":{"input_tokens":4142,"output_tokens":254,"total_tokens":4396,"input_tokens_details":{"cached_tokens":4328},"output_tokens_details":{"reasoning_tokens":10}}}"#.to_string(),
        id: "1".to_string(),
        retry: None,
    };

    let fut = conv.convert_event(event);
    let events = futures::executor::block_on(fut);
    assert!(!events.is_empty());
    let ev = events.first().unwrap().as_ref().unwrap();
    match ev {
        crate::streaming::ChatStreamEvent::UsageUpdate { usage } => {
            assert_eq!(usage.normalized_input_tokens().total, Some(8470));
            assert_eq!(usage.normalized_input_tokens().no_cache, Some(4142));
            assert_eq!(usage.normalized_input_tokens().cache_read, Some(4328));
            assert_eq!(usage.normalized_output_tokens().total, Some(254));
            assert_eq!(usage.normalized_output_tokens().text, Some(244));
            assert_eq!(usage.normalized_output_tokens().reasoning, Some(10));
            assert_eq!(
                usage.raw_usage_value().expect("raw usage")["input_tokens"],
                serde_json::json!(4142)
            );
        }
        _ => panic!("expected UsageUpdate"),
    }
}

#[test]
fn xai_responses_completed_finish_exposes_cost_provider_metadata() {
    let conv = OpenAiResponsesEventConverter::new()
        .with_stream_parts_style(StreamPartsStyle::Xai)
        .with_responses_transform_style(
            crate::standards::openai::transformers::ResponsesTransformStyle::Xai,
        )
        .with_provider_metadata_key("xai");
    let event = eventsource_stream::Event {
        event: "response.completed".to_string(),
        data: r#"{"type":"response.completed","response":{"id":"resp_xai_cost","object":"response","model":"grok-4-fast","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"cost_in_usd_ticks":113500}}}"#.to_string(),
        id: "1".to_string(),
        retry: None,
    };

    let fut = conv.convert_event(event);
    let events = futures::executor::block_on(fut);
    assert!(events.is_empty());

    let pending = conv.handle_stream_end_events();
    let finish = pending
        .iter()
        .find_map(|event| match event {
            Ok(crate::streaming::ChatStreamEvent::Part {
                part:
                    crate::types::ChatStreamPart::Finish {
                        provider_metadata, ..
                    },
            }) => provider_metadata.as_ref(),
            _ => None,
        })
        .expect("finish provider metadata");

    assert_eq!(finish["xai"]["costInUsdTicks"], serde_json::json!(113500));

    let response = pending
        .iter()
        .find_map(|event| match event {
            Ok(crate::streaming::ChatStreamEvent::StreamEnd { response }) => Some(response),
            _ => None,
        })
        .expect("stream end response");
    assert_eq!(
        response
            .provider_metadata
            .as_ref()
            .expect("response provider metadata")["xai"]["costInUsdTicks"],
        serde_json::json!(113500)
    );
}

#[test]
fn xai_responses_completed_finish_defaults_missing_usage_to_zero() {
    let conv = OpenAiResponsesEventConverter::new()
        .with_stream_parts_style(StreamPartsStyle::Xai)
        .with_responses_transform_style(
            crate::standards::openai::transformers::ResponsesTransformStyle::Xai,
        )
        .with_provider_metadata_key("xai");
    let event = eventsource_stream::Event {
        event: "response.completed".to_string(),
        data: r#"{"type":"response.completed","response":{"id":"resp_xai_no_usage","object":"response","model":"grok-4-fast","status":"completed","output":[]}}"#.to_string(),
        id: "1".to_string(),
        retry: None,
    };

    let fut = conv.convert_event(event);
    let events = futures::executor::block_on(fut);
    assert!(events.is_empty());

    let pending = conv.handle_stream_end_events();
    let finish_usage = pending
        .iter()
        .find_map(|event| match event {
            Ok(crate::streaming::ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::Finish { usage, .. },
            }) => Some(usage),
            _ => None,
        })
        .expect("finish usage");

    assert_eq!(finish_usage.normalized_input_tokens().total, Some(0));
    assert_eq!(finish_usage.normalized_input_tokens().no_cache, Some(0));
    assert_eq!(finish_usage.normalized_input_tokens().cache_read, Some(0));
    assert_eq!(finish_usage.normalized_input_tokens().cache_write, Some(0));
    assert_eq!(finish_usage.normalized_output_tokens().total, Some(0));
    assert_eq!(finish_usage.normalized_output_tokens().text, Some(0));
    assert_eq!(finish_usage.normalized_output_tokens().reasoning, Some(0));
    assert!(finish_usage.raw_usage_value().is_none());

    let response_usage = pending
        .iter()
        .find_map(|event| match event {
            Ok(crate::streaming::ChatStreamEvent::StreamEnd { response }) => {
                response.usage.as_ref()
            }
            _ => None,
        })
        .expect("stream end response usage");
    assert_eq!(response_usage.normalized_input_tokens().total, Some(0));
    assert_eq!(response_usage.normalized_output_tokens().total, Some(0));
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
        data: r#"{"delta":{"tool_calls":[{"id":"t1","function":{"name":"fn","arguments":"{}"}}]}}"#
            .to_string(),
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
            assert_eq!(usage.prompt_tokens(), Some(4));
            assert_eq!(usage.completion_tokens(), Some(6));
            assert_eq!(usage.total_tokens(), Some(10));
            assert_eq!(usage.normalized_input_tokens().no_cache, Some(4));
            assert_eq!(usage.normalized_output_tokens().text, Some(6));
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
    assert_eq!(out_added.len(), 3);
    match stream_part(&out_added[0]).expect("tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart {
            id,
            tool_name,
            provider_executed,
            ..
        } => {
            assert_eq!(id, "ws_1");
            assert_eq!(tool_name, "web_search");
            assert_eq!(provider_executed, Some(true));
        }
        other => panic!("expected tool-input-start part, got {other:?}"),
    }
    match stream_part(&out_added[1]).expect("tool-input-end part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputEnd { id, .. } => {
            assert_eq!(id, "ws_1");
        }
        other => panic!("expected tool-input-end part, got {other:?}"),
    }
    match stream_part(&out_added[2]).expect("tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "ws_1");
            assert_eq!(call.tool_name, "web_search");
            assert_eq!(call.provider_executed, Some(true));
        }
        other => panic!("expected tool-call part, got {other:?}"),
    }
    assert_eq!(
        openai_responses_raw_item(&out_added[2])
            .and_then(|raw_item| raw_item.get("type"))
            .and_then(|value| value.as_str()),
        Some("web_search_call")
    );

    let ev_done = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"rust"}}}"#.to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let out_done = futures::executor::block_on(conv.convert_event(ev_done));
    assert_eq!(out_done.len(), 1);
    match stream_part(&out_done[0]).expect("tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "ws_1");
            assert_eq!(result.tool_name, "web_search");
            assert_eq!(result.result["action"]["query"], serde_json::json!("rust"));
        }
        other => panic!("expected tool-result part, got {other:?}"),
    }
    assert_eq!(
        openai_responses_raw_item(&out_done[0])
            .and_then(|raw_item| raw_item.get("type"))
            .and_then(|value| value.as_str()),
        Some("web_search_call")
    );

    // If the payload includes results, we also emit Vercel-aligned sources.
    let ev_done_with_results = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"rust"},"results":[{"url":"https://www.rust-lang.org","title":"Rust"}]}}"#.to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let out_done = futures::executor::block_on(conv.convert_event(ev_done_with_results));
    assert_eq!(out_done.len(), 2);
    match stream_part(&out_done[0]).expect("tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "ws_1");
        }
        other => panic!("expected tool-result part, got {other:?}"),
    }

    match stream_part(&out_done[1]).expect("source part") {
        crate::streaming::LanguageModelV3StreamPart::Source(
            crate::streaming::LanguageModelV3Source::Url { url, .. },
        ) => {
            assert_eq!(url, "https://www.rust-lang.org");
        }
        other => panic!("expected url source part, got {other:?}"),
    }
}

#[test]
fn responses_provider_tool_name_uses_configured_web_search_preview() {
    let conv = OpenAiResponsesEventConverter::new();

    let ev_created = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.created","response":{"tools":[{"type":"web_search_preview","search_context_size":"low","user_location":{"type":"approximate"}}]}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let _ = futures::executor::block_on(conv.convert_event(ev_created));

    let ev_added = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"}}"#
            .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let out_added = futures::executor::block_on(conv.convert_event(ev_added));
    // Vercel alignment: web search emits tool-input-start/end even with empty input.
    assert_eq!(out_added.len(), 3);
    match stream_part(&out_added[0]).expect("tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart { tool_name, .. } => {
            assert_eq!(tool_name, "web_search_preview");
        }
        other => panic!("expected tool-input-start part, got {other:?}"),
    }
    match stream_part(&out_added[1]).expect("tool-input-end part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputEnd { .. } => {}
        other => panic!("expected tool-input-end part, got {other:?}"),
    }
    match stream_part(&out_added[2]).expect("tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_name, "web_search_preview");
        }
        other => panic!("expected tool-call part, got {other:?}"),
    }

    let ev_done = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"rust"}}}"#
            .to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let out_done = futures::executor::block_on(conv.convert_event(ev_done));
    assert_eq!(out_done.len(), 1);
    match stream_part(&out_done[0]).expect("tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_name, "web_search_preview");
        }
        other => panic!("expected tool-result part, got {other:?}"),
    }
}

#[test]
fn responses_image_generation_partial_image_emits_preliminary_tool_result() {
    let tools = [siumai_core::tools::openai::image_generation()];
    let conv = OpenAiResponsesEventConverter::new().with_request_tools(&tools);

    let ev = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.image_generation_call.partial_image","output_index":1,"item_id":"ig_1","partial_image_index":0,"partial_image_b64":"b64_partial"}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let out = futures::executor::block_on(conv.convert_event(ev));
    assert_eq!(out.len(), 1);

    match stream_part(&out[0]).expect("tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "ig_1");
            assert_eq!(result.tool_name, "generateImage");
            assert_eq!(result.result["result"], serde_json::json!("b64_partial"));
            assert_eq!(result.preliminary, Some(true));
        }
        other => panic!("expected tool-result part, got {other:?}"),
    }
}

#[test]
fn responses_computer_call_stream_matches_ai_sdk_lifecycle() {
    let conv = OpenAiResponsesEventConverter::new();

    let ev_added = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"cu_1","type":"computer_call","status":"in_progress","action":{"type":"screenshot"}}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let out_added = futures::executor::block_on(conv.convert_event(ev_added));
    assert_eq!(out_added.len(), 1);
    match stream_part(&out_added[0]).expect("tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart {
            id,
            tool_name,
            provider_executed,
            ..
        } => {
            assert_eq!(id, "cu_1");
            assert_eq!(tool_name, "computer_use");
            assert_eq!(provider_executed, Some(true));
        }
        other => panic!("expected tool-input-start part, got {other:?}"),
    }

    let ev_done = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"cu_1","type":"computer_call","status":"completed","action":{"type":"screenshot"}}}"#
            .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let out_done = futures::executor::block_on(conv.convert_event(ev_done));
    assert_eq!(out_done.len(), 3);

    match stream_part(&out_done[0]).expect("tool-input-end part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputEnd { id, .. } => {
            assert_eq!(id, "cu_1");
        }
        other => panic!("expected tool-input-end part, got {other:?}"),
    }

    match stream_part(&out_done[1]).expect("tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "cu_1");
            assert_eq!(call.tool_name, "computer_use");
            assert_eq!(call.input, "");
            assert_eq!(call.provider_executed, Some(true));
        }
        other => panic!("expected tool-call part, got {other:?}"),
    }

    match stream_part(&out_done[2]).expect("tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "cu_1");
            assert_eq!(result.tool_name, "computer_use");
            assert_eq!(
                result.result["type"],
                serde_json::json!("computer_use_tool_result")
            );
            assert_eq!(result.result["status"], serde_json::json!("completed"));
            assert!(
                result.result.get("action").is_none(),
                "computer result should not expose action"
            );
        }
        other => panic!("expected tool-result part, got {other:?}"),
    }
}

#[test]
fn responses_file_search_stream_maps_ai_sdk_result_shape() {
    let tools = [siumai_core::tools::openai::file_search()];
    let conv = OpenAiResponsesEventConverter::new().with_request_tools(&tools);

    let ev_done = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"fs_1","type":"file_search_call","status":"completed","queries":["rust"],"results":[{"file_id":"file_1","filename":"rust.txt","attributes":{"lang":"rust"},"score":0.98,"text":"Rust text"}]}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let out = futures::executor::block_on(conv.convert_event(ev_done));
    assert_eq!(out.len(), 1);

    match stream_part(&out[0]).expect("tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "fs_1");
            assert_eq!(result.tool_name, "fileSearch");
            assert_eq!(result.result["queries"], serde_json::json!(["rust"]));
            assert_eq!(
                result.result["results"][0]["fileId"],
                serde_json::json!("file_1")
            );
            assert_eq!(
                result.result["results"][0]["attributes"]["lang"],
                serde_json::json!("rust")
            );
            assert!(
                result.result["results"][0].get("file_id").is_none(),
                "file search results should expose fileId"
            );
            assert!(
                result.result.get("status").is_none(),
                "AI SDK file search result does not include status"
            );
        }
        other => panic!("expected tool-result part, got {other:?}"),
    }
}

#[test]
fn responses_code_interpreter_code_delta_escapes_json_string() {
    let conv = OpenAiResponsesEventConverter::new();
    let mut stitched_input = String::new();

    let ev_added = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "ci_1",
                "type": "code_interpreter_call",
                "status": "in_progress",
                "container_id": "cntr_1"
            }
        })
        .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let out_added = futures::executor::block_on(conv.convert_event(ev_added));
    append_tool_input_deltas(&out_added, &mut stitched_input);

    let code_delta = "print(\"x\")\npath = \"C:\\tmp\"";
    let ev_delta = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::json!({
            "type": "response.code_interpreter_call_code.delta",
            "item_id": "ci_1",
            "output_index": 0,
            "delta": code_delta
        })
        .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let out_delta = futures::executor::block_on(conv.convert_event(ev_delta));
    append_tool_input_deltas(&out_delta, &mut stitched_input);

    let ev_done = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::json!({
            "type": "response.code_interpreter_call_code.done",
            "item_id": "ci_1",
            "output_index": 0,
            "code": code_delta
        })
        .to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let out_done = futures::executor::block_on(conv.convert_event(ev_done));
    append_tool_input_deltas(&out_done, &mut stitched_input);

    let parsed: serde_json::Value =
        serde_json::from_str(&stitched_input).expect("stitched code input is valid JSON");
    assert_eq!(parsed["containerId"], serde_json::json!("cntr_1"));
    assert_eq!(parsed["code"], serde_json::json!(code_delta));
}

#[test]
fn responses_apply_patch_diff_delta_escapes_and_done_does_not_duplicate() {
    let conv = OpenAiResponsesEventConverter::new();
    let mut stitched_input = String::new();

    let ev_added = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "ap_1",
                "type": "apply_patch_call",
                "call_id": "call_patch_1",
                "status": "in_progress",
                "operation": {
                    "type": "update_file",
                    "path": "src/lib.rs"
                }
            }
        })
        .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let out_added = futures::executor::block_on(conv.convert_event(ev_added));
    append_tool_input_deltas(&out_added, &mut stitched_input);

    let diff = "@@\n-println!(\"old\");\n+println!(\"new\\\\path\");";
    let ev_delta = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::json!({
            "type": "response.apply_patch_call_operation_diff.delta",
            "item_id": "ap_1",
            "output_index": 0,
            "delta": diff
        })
        .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let out_delta = futures::executor::block_on(conv.convert_event(ev_delta));
    append_tool_input_deltas(&out_delta, &mut stitched_input);

    let ev_done = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::json!({
            "type": "response.apply_patch_call_operation_diff.done",
            "item_id": "ap_1",
            "output_index": 0,
            "diff": diff
        })
        .to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let out_done = futures::executor::block_on(conv.convert_event(ev_done));
    append_tool_input_deltas(&out_done, &mut stitched_input);

    let parsed: serde_json::Value =
        serde_json::from_str(&stitched_input).expect("stitched apply patch input is valid JSON");
    assert_eq!(parsed["callId"], serde_json::json!("call_patch_1"));
    assert_eq!(parsed["operation"]["diff"], serde_json::json!(diff));
}

#[test]
fn responses_apply_patch_diff_done_emits_diff_without_delta() {
    let conv = OpenAiResponsesEventConverter::new();
    let mut stitched_input = String::new();

    let ev_added = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "ap_2",
                "type": "apply_patch_call",
                "call_id": "call_patch_2",
                "status": "in_progress",
                "operation": {
                    "type": "update_file",
                    "path": "src/main.rs"
                }
            }
        })
        .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let out_added = futures::executor::block_on(conv.convert_event(ev_added));
    append_tool_input_deltas(&out_added, &mut stitched_input);

    let diff = "@@\n+let text = \"done-only\";";
    let ev_done = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::json!({
            "type": "response.apply_patch_call_operation_diff.done",
            "item_id": "ap_2",
            "output_index": 0,
            "diff": diff
        })
        .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let out_done = futures::executor::block_on(conv.convert_event(ev_done));
    append_tool_input_deltas(&out_done, &mut stitched_input);

    let parsed: serde_json::Value =
        serde_json::from_str(&stitched_input).expect("stitched apply patch input is valid JSON");
    assert_eq!(parsed["callId"], serde_json::json!("call_patch_2"));
    assert_eq!(parsed["operation"]["diff"], serde_json::json!(diff));
}

#[test]
fn responses_tool_search_stream_maps_call_and_output() {
    let tools = [siumai_core::tools::openai::tool_search()];
    let conv = OpenAiResponsesEventConverter::new().with_request_tools(&tools);

    let call_done = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"tsc_1","type":"tool_search_call","execution":"server","call_id":null,"status":"completed","arguments":{"paths":["get_weather"]}}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let call_events = futures::executor::block_on(conv.convert_event(call_done));
    assert_eq!(call_events.len(), 1);
    match stream_part(&call_events[0]).expect("tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "tsc_1");
            assert_eq!(call.tool_name, "toolSearch");
            assert_eq!(call.provider_executed, Some(true));
            let input: serde_json::Value =
                serde_json::from_str(&call.input).expect("tool-search input json");
            assert_eq!(input["call_id"], serde_json::Value::Null);
            assert_eq!(
                input["arguments"]["paths"],
                serde_json::json!(["get_weather"])
            );
        }
        other => panic!("expected tool-call part, got {other:?}"),
    }

    let output_done = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"tso_1","type":"tool_search_output","execution":"server","call_id":null,"status":"completed","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}]}}"#
            .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let result_events = futures::executor::block_on(conv.convert_event(output_done));
    assert_eq!(result_events.len(), 1);
    match stream_part(&result_events[0]).expect("tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "tsc_1");
            assert_eq!(result.tool_name, "toolSearch");
            assert_eq!(
                result.result["tools"][0]["name"],
                serde_json::json!("get_weather")
            );
        }
        other => panic!("expected tool-result part, got {other:?}"),
    }
}

#[test]
fn responses_client_tool_search_stream_uses_final_call_id() {
    let tools = [siumai_core::tools::openai::tool_search()];
    let conv = OpenAiResponsesEventConverter::new().with_request_tools(&tools);

    let call_done = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"tsc_client_1","type":"tool_search_call","execution":"client","call_id":"call_final","status":"completed","arguments":{"goal":"Find weather"}}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let events = futures::executor::block_on(conv.convert_event(call_done));
    assert_eq!(events.len(), 3);

    match stream_part(&events[0]).expect("tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart {
            id,
            tool_name,
            provider_executed,
            ..
        } => {
            assert_eq!(id, "call_final");
            assert_eq!(tool_name, "toolSearch");
            assert_eq!(provider_executed, None);
        }
        other => panic!("expected tool-input-start part, got {other:?}"),
    }
    match stream_part(&events[1]).expect("tool-input-end part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputEnd { id, .. } => {
            assert_eq!(id, "call_final");
        }
        other => panic!("expected tool-input-end part, got {other:?}"),
    }
    match stream_part(&events[2]).expect("tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "call_final");
            assert_eq!(call.tool_name, "toolSearch");
            assert_eq!(call.provider_executed, None);
            let input: serde_json::Value =
                serde_json::from_str(&call.input).expect("tool-search input json");
            assert_eq!(input["call_id"], serde_json::json!("call_final"));
            assert_eq!(
                input["arguments"]["goal"],
                serde_json::json!("Find weather")
            );
        }
        other => panic!("expected tool-call part, got {other:?}"),
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
    match stream_part(&out[0]).expect("source part") {
        crate::streaming::LanguageModelV3StreamPart::Source(
            crate::streaming::LanguageModelV3Source::Url { url, .. },
        ) => {
            assert_eq!(url, "https://www.rust-lang.org");
        }
        other => panic!("expected url source part, got {other:?}"),
    }

    let ev = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_text.annotation.added","annotation":{"type":"file_citation","file_id":"file_123","filename":"notes.txt","quote":"Document","index":7,"start_index":10,"end_index":20}}"#
            .to_string(),
        id: "2".to_string(),
        retry: None,
    };

    let out = futures::executor::block_on(conv.convert_event(ev));
    assert_eq!(out.len(), 1);
    match stream_part(&out[0]).expect("source part") {
        crate::streaming::LanguageModelV3StreamPart::Source(
            crate::streaming::LanguageModelV3Source::Document {
                media_type,
                title,
                filename,
                provider_metadata,
                ..
            },
        ) => {
            assert_eq!(media_type, "text/plain");
            assert_eq!(title, "Document");
            assert_eq!(filename.as_deref(), Some("notes.txt"));
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("openai"))
                    .and_then(|meta| meta.get("type")),
                Some(&serde_json::json!("file_citation"))
            );
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("openai"))
                    .and_then(|meta| meta.get("fileId")),
                Some(&serde_json::json!("file_123"))
            );
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("openai"))
                    .and_then(|meta| meta.get("index")),
                Some(&serde_json::json!(7))
            );
        }
        other => panic!("expected document source part, got {other:?}"),
    }

    let ev = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_text.annotation.added","annotation":{"type":"container_file_citation","file_id":"file_container_1","container_id":"container_42","index":3,"filename":"bundle.txt","quote":"Bundle","start_index":21,"end_index":30}}"#
            .to_string(),
        id: "3".to_string(),
        retry: None,
    };

    let out = futures::executor::block_on(conv.convert_event(ev));
    assert_eq!(out.len(), 1);
    match stream_part(&out[0]).expect("source part") {
        crate::streaming::LanguageModelV3StreamPart::Source(
            crate::streaming::LanguageModelV3Source::Document {
                media_type,
                filename,
                provider_metadata,
                ..
            },
        ) => {
            assert_eq!(media_type, "text/plain");
            assert_eq!(filename.as_deref(), Some("bundle.txt"));
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("openai"))
                    .and_then(|meta| meta.get("type")),
                Some(&serde_json::json!("container_file_citation"))
            );
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("openai"))
                    .and_then(|meta| meta.get("containerId")),
                Some(&serde_json::json!("container_42"))
            );
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("openai"))
                    .and_then(|meta| meta.get("index")),
                Some(&serde_json::json!(3))
            );
        }
        other => panic!("expected container document source part, got {other:?}"),
    }

    let ev = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_text.annotation.added","annotation":{"type":"file_path","file_id":"file_path_9","index":5,"filename":"artifact.bin","start_index":31,"end_index":40}}"#
            .to_string(),
        id: "4".to_string(),
        retry: None,
    };

    let out = futures::executor::block_on(conv.convert_event(ev));
    assert_eq!(out.len(), 1);
    match stream_part(&out[0]).expect("source part") {
        crate::streaming::LanguageModelV3StreamPart::Source(
            crate::streaming::LanguageModelV3Source::Document {
                media_type,
                filename,
                provider_metadata,
                ..
            },
        ) => {
            assert_eq!(media_type, "application/octet-stream");
            assert_eq!(filename.as_deref(), Some("artifact.bin"));
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("openai"))
                    .and_then(|meta| meta.get("type")),
                Some(&serde_json::json!("file_path"))
            );
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("openai"))
                    .and_then(|meta| meta.get("fileId")),
                Some(&serde_json::json!("file_path_9"))
            );
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("openai"))
                    .and_then(|meta| meta.get("index")),
                Some(&serde_json::json!(5))
            );
        }
        other => panic!("expected file-path source part, got {other:?}"),
    }
}

#[test]
fn responses_reasoning_summary_text_delta_emits_thinking_delta_and_part() {
    let conv = OpenAiResponsesEventConverter::new();

    let ev = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","summary_index":0,"delta":"Let me think."}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };

    let out = futures::executor::block_on(conv.convert_event(ev));
    assert_eq!(out.len(), 2);
    assert!(out.iter().any(|event| {
        matches!(
            event.as_ref().expect("thinking delta event"),
            crate::streaming::ChatStreamEvent::ThinkingDelta { delta }
                if delta == "Let me think."
        )
    }));
    assert!(out.iter().any(|event| {
        matches!(
            stream_part(event),
            Some(crate::streaming::LanguageModelV3StreamPart::ReasoningDelta {
                delta,
                provider_metadata,
                ..
            })
                if delta == "Let me think."
                    && provider_metadata
                        .as_ref()
                        .and_then(|meta| meta.get("openai"))
                        .and_then(|meta| meta.get("itemId"))
                        == Some(&serde_json::json!("rs_1"))
        )
    }));
}

#[test]
fn responses_created_emits_part_stream_start_and_response_metadata() {
    let conv = OpenAiResponsesEventConverter::new();
    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.created","response":{"id":"resp_1","model":"gpt-test","created_at":1735689600}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };

    let out = futures::executor::block_on(conv.convert_event(event));
    assert_eq!(out.len(), 2);
    assert!(out.iter().any(|event| {
        matches!(
            stream_part(event),
            Some(crate::streaming::LanguageModelV3StreamPart::StreamStart { warnings })
                if warnings.is_empty()
        )
    }));
    assert!(out.iter().any(|event| {
        matches!(
            stream_part(event),
            Some(crate::streaming::LanguageModelV3StreamPart::ResponseMetadata(metadata))
                if metadata.id.as_deref() == Some("resp_1")
                    && metadata.model_id.as_deref() == Some("gpt-test")
                    && metadata.timestamp == Some(parse_test_timestamp("2025-01-01T00:00:00Z"))
        )
    }));
}

#[test]
fn responses_message_output_text_events_emit_text_parts() {
    let conv = OpenAiResponsesEventConverter::new();

    let added = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant"}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let added_out = futures::executor::block_on(conv.convert_event(added));
    assert!(matches!(
        stream_part(&added_out[0]),
        Some(crate::streaming::LanguageModelV3StreamPart::TextStart {
            id,
            provider_metadata,
        }) if id == "msg_1"
            && provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("openai"))
                .and_then(|meta| meta.get("itemId"))
                == Some(&serde_json::json!("msg_1"))
    ));

    let delta = eventsource_stream::Event {
        event: "response.output_text.delta".to_string(),
        data: r#"{"type":"response.output_text.delta","item_id":"msg_1","delta":"Hello"}"#
            .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let delta_out = futures::executor::block_on(conv.convert_event(delta));
    assert!(delta_out.iter().any(|event| {
        matches!(
            stream_part(event),
            Some(crate::streaming::LanguageModelV3StreamPart::TextDelta { id, delta, .. })
                if id == "msg_1" && delta == "Hello"
        )
    }));
    assert!(delta_out.iter().any(|event| {
        matches!(
            event.as_ref().expect("content delta event"),
            crate::streaming::ChatStreamEvent::ContentDelta { delta, .. } if delta == "Hello"
        )
    }));

    let done = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello","annotations":[]}]}}"#
            .to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let done_out = futures::executor::block_on(conv.convert_event(done));
    assert!(matches!(
        stream_part(&done_out[0]),
        Some(crate::streaming::LanguageModelV3StreamPart::TextEnd {
            id,
            provider_metadata,
        }) if id == "msg_1"
            && provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("openai"))
                .and_then(|meta| meta.get("itemId"))
                == Some(&serde_json::json!("msg_1"))
    ));
}

#[test]
fn responses_reasoning_summary_part_done_waits_for_final_item_by_default() {
    let conv = OpenAiResponsesEventConverter::new();

    let added = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","encrypted_content":"enc_start"}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let added_out = futures::executor::block_on(conv.convert_event(added));
    assert!(matches!(
        stream_part(&added_out[0]),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningStart { id, .. })
            if id == "rs_1:0"
    ));

    let done0 = eventsource_stream::Event {
        event: "".to_string(),
        data:
            r#"{"type":"response.reasoning_summary_part.done","item_id":"rs_1","summary_index":0}"#
                .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let done0_out = futures::executor::block_on(conv.convert_event(done0));
    assert!(done0_out.is_empty());

    let added1 = eventsource_stream::Event {
        event: "".to_string(),
        data:
            r#"{"type":"response.reasoning_summary_part.added","item_id":"rs_1","summary_index":1}"#
                .to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let added1_out = futures::executor::block_on(conv.convert_event(added1));
    assert_eq!(added1_out.len(), 2);
    assert!(matches!(
        stream_part(&added1_out[0]),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningEnd {
            id,
            provider_metadata,
        }) if id == "rs_1:0"
            && provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("openai"))
                .and_then(|meta| meta.get("reasoningEncryptedContent"))
                .is_none()
    ));
    assert!(matches!(
        stream_part(&added1_out[1]),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningStart { id, .. })
            if id == "rs_1:1"
    ));

    let done1 = eventsource_stream::Event {
        event: "".to_string(),
        data:
            r#"{"type":"response.reasoning_summary_part.done","item_id":"rs_1","summary_index":1}"#
                .to_string(),
        id: "4".to_string(),
        retry: None,
    };
    let done1_out = futures::executor::block_on(conv.convert_event(done1));
    assert!(done1_out.is_empty());

    let final_item = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","encrypted_content":"enc_final"}}"#
            .to_string(),
        id: "5".to_string(),
        retry: None,
    };
    let final_out = futures::executor::block_on(conv.convert_event(final_item));
    assert_eq!(final_out.len(), 1);
    assert!(matches!(
        stream_part(&final_out[0]),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningEnd {
            id,
            provider_metadata,
        }) if id == "rs_1:1"
            && provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("openai"))
                .and_then(|meta| meta.get("reasoningEncryptedContent"))
                == Some(&serde_json::json!("enc_final"))
    ));
}

#[test]
fn responses_reasoning_summary_part_done_ends_immediately_when_store_requested() {
    let conv = OpenAiResponsesEventConverter::new().with_requested_store(Some(true));

    let added = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning"}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let _ = futures::executor::block_on(conv.convert_event(added));

    let done = eventsource_stream::Event {
        event: "".to_string(),
        data:
            r#"{"type":"response.reasoning_summary_part.done","item_id":"rs_1","summary_index":0}"#
                .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let done_out = futures::executor::block_on(conv.convert_event(done));
    assert_eq!(done_out.len(), 1);
    assert!(matches!(
        stream_part(&done_out[0]),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningEnd {
            id,
            provider_metadata,
        }) if id == "rs_1:0"
            && provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("openai"))
                .and_then(|meta| meta.get("itemId"))
                == Some(&serde_json::json!("rs_1"))
            && provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("openai"))
                .and_then(|meta| meta.get("reasoningEncryptedContent"))
                .is_none()
    ));

    let final_item = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","encrypted_content":"enc_final"}}"#
            .to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let final_out = futures::executor::block_on(conv.convert_event(final_item));
    assert!(final_out.is_empty());
}

#[test]
fn responses_reasoning_item_events_emit_reasoning_parts() {
    let conv = OpenAiResponsesEventConverter::new();

    let added = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"in_progress","summary":[],"encrypted_content":"enc_1"}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let added_out = futures::executor::block_on(conv.convert_event(added));
    assert!(matches!(
        stream_part(&added_out[0]),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningStart {
            id,
            provider_metadata,
        }) if id == "rs_1:0"
            && provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("openai"))
                .and_then(|meta| meta.get("reasoningEncryptedContent"))
                == Some(&serde_json::json!("enc_1"))
    ));

    let delta = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","summary_index":0,"delta":"Let me think."}"#
            .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let delta_out = futures::executor::block_on(conv.convert_event(delta));
    assert!(delta_out.iter().any(|event| {
        matches!(
            stream_part(event),
            Some(crate::streaming::LanguageModelV3StreamPart::ReasoningDelta { id, delta, .. })
                if id == "rs_1:0" && delta == "Let me think."
        )
    }));
    assert!(delta_out.iter().any(|event| {
        matches!(
            event.as_ref().expect("thinking delta event"),
            crate::streaming::ChatStreamEvent::ThinkingDelta { delta } if delta == "Let me think."
        )
    }));

    let done = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"completed","summary":[],"encrypted_content":"enc_2"}}"#
            .to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let done_out = futures::executor::block_on(conv.convert_event(done));
    assert!(matches!(
        stream_part(&done_out[0]),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningEnd {
            id,
            provider_metadata,
        }) if id == "rs_1:0"
            && provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("openai"))
                .and_then(|meta| meta.get("reasoningEncryptedContent"))
                == Some(&serde_json::json!("enc_2"))
    ));
}

#[test]
fn responses_stream_proxy_serializes_basic_text_deltas() {
    let conv = OpenAiResponsesEventConverter::new();

    let start_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");
    let start_frames = parse_sse_frames(&start_bytes);
    assert_eq!(start_frames.len(), 1);
    assert_eq!(start_frames[0].0, "response.created");
    assert_eq!(
        start_frames[0].1["type"],
        serde_json::json!("response.created")
    );

    let delta_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        })
        .expect("serialize delta");
    let delta_frames = parse_sse_frames(&delta_bytes);
    assert!(
        delta_frames
            .iter()
            .any(|(ev, v)| ev == "response.output_text.delta"
                && v["delta"] == serde_json::json!("Hello"))
    );

    let end_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                content: MessageContent::Text("Hello".to_string()),
                usage: Some(
                    Usage::builder()
                        .prompt_tokens(3)
                        .completion_tokens(5)
                        .total_tokens(8)
                        .build(),
                ),
                finish_reason: Some(FinishReason::Stop),
                raw_finish_reason: None,
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            },
        })
        .expect("serialize end");
    let end_frames = parse_sse_frames(&end_bytes);
    let completed = end_frames
        .iter()
        .find(|(ev, _)| ev == "response.completed")
        .map(|(_, value)| value)
        .expect("response.completed frame");
    assert_eq!(completed["type"], serde_json::json!("response.completed"));
    assert_eq!(
        completed["response"]["finish_reason"],
        serde_json::json!("stop")
    );
    assert_eq!(
        completed["response"]["status"],
        serde_json::json!("completed")
    );
    assert_eq!(
        completed["response"]["usage"]["input_tokens"],
        serde_json::json!(3)
    );
    assert_eq!(
        completed["response"]["usage"]["output_tokens"],
        serde_json::json!(5)
    );
}

#[test]
fn responses_stream_proxy_serializes_openai_text_stream_part_delta() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": "msg_1",
                "delta": "Hello",
            }),
        })
        .expect("serialize stream part");

    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.output_text.delta" && v["delta"] == serde_json::json!("Hello")
    }));
}

#[test]
fn responses_stream_proxy_serializes_v3_text_delta_even_with_non_openai_event_type() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "gemini:tool".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": "msg_1",
                "delta": "Hello",
            }),
        })
        .expect("serialize stream part");

    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.output_text.delta" && v["delta"] == serde_json::json!("Hello")
    }));
}

#[test]
fn responses_stream_proxy_serializes_openai_reasoning_stream_part_delta() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:reasoning-delta".to_string(),
            data: serde_json::json!({
                "type": "reasoning-delta",
                "id": "rs_1:0",
                "delta": "think",
            }),
        })
        .expect("serialize reasoning stream part");

    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.reasoning_summary_text.delta" && v["delta"] == serde_json::json!("think")
    }));
}

#[test]
fn responses_stream_proxy_serializes_v3_reasoning_delta_even_with_non_openai_event_type() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "gemini:reasoning".to_string(),
            data: serde_json::json!({
                "type": "reasoning-delta",
                "id": "rs_1:0",
                "delta": "think",
            }),
        })
        .expect("serialize reasoning stream part");

    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.reasoning_summary_text.delta" && v["delta"] == serde_json::json!("think")
    }));
}

#[test]
fn responses_stream_proxy_serializes_openai_source_stream_part_as_annotation_added() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": "msg_1",
                "delta": "Hello",
            }),
        })
        .expect("serialize text delta");

    let url_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "url",
                "id": "ann:url:https://www.rust-lang.org",
                "url": "https://www.rust-lang.org",
                "title": "Rust",
            }),
        })
        .expect("serialize url source");
    let url_frames = parse_sse_frames(&url_bytes);
    assert!(url_frames.iter().any(|(ev, v)| {
        ev == "response.output_text.annotation.added"
            && v["annotation"]["type"] == serde_json::json!("url_citation")
            && v["annotation"]["url"] == serde_json::json!("https://www.rust-lang.org")
    }));

    let doc_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "document",
                "id": "ann:doc:file_123",
                "url": "file_123",
                "title": "Document",
                "mediaType": "text/plain",
                "filename": "notes.txt",
                "providerMetadata": { "openai": { "type": "file_citation", "fileId": "file_123", "index": 7 } },
            }),
        })
        .expect("serialize doc source");
    let doc_frames = parse_sse_frames(&doc_bytes);
    assert!(doc_frames.iter().any(|(ev, v)| {
        ev == "response.output_text.annotation.added"
            && v["annotation"]["type"] == serde_json::json!("file_citation")
            && v["annotation"]["file_id"] == serde_json::json!("file_123")
            && v["annotation"]["index"] == serde_json::json!(7)
            && v["annotation"]["filename"] == serde_json::json!("notes.txt")
    }));

    let container_doc_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "document",
                "id": "ann:doc:file_container_1",
                "url": "file_container_1",
                "title": "Bundle",
                "mediaType": "text/plain",
                "filename": "bundle.txt",
                "providerMetadata": { "openai": { "type": "container_file_citation", "fileId": "file_container_1", "containerId": "container_42", "index": 3 } },
            }),
        })
        .expect("serialize container doc source");
    let container_doc_frames = parse_sse_frames(&container_doc_bytes);
    assert!(container_doc_frames.iter().any(|(ev, v)| {
        ev == "response.output_text.annotation.added"
            && v["annotation"]["type"] == serde_json::json!("container_file_citation")
            && v["annotation"]["file_id"] == serde_json::json!("file_container_1")
            && v["annotation"]["container_id"] == serde_json::json!("container_42")
            && v["annotation"]["index"] == serde_json::json!(3)
            && v["annotation"]["filename"] == serde_json::json!("bundle.txt")
    }));

    let file_path_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "document",
                "id": "ann:doc:file_path_9",
                "url": "file_path_9",
                "title": "artifact.bin",
                "mediaType": "application/octet-stream",
                "filename": "artifact.bin",
                "providerMetadata": { "openai": { "type": "file_path", "fileId": "file_path_9", "index": 5 } },
            }),
        })
        .expect("serialize file path source");
    let file_path_frames = parse_sse_frames(&file_path_bytes);
    assert!(file_path_frames.iter().any(|(ev, v)| {
        ev == "response.output_text.annotation.added"
            && v["annotation"]["type"] == serde_json::json!("file_path")
            && v["annotation"]["file_id"] == serde_json::json!("file_path_9")
            && v["annotation"]["index"] == serde_json::json!(5)
            && v["annotation"]["filename"] == serde_json::json!("artifact.bin")
    }));
}

#[test]
fn responses_stream_proxy_serializes_openai_source_before_text_with_message_scaffold() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:response-metadata".to_string(),
            data: serde_json::json!({
                "type": "response-metadata",
                "id": "resp_test",
                "modelId": "gpt-test",
                "timestamp": "2025-01-01T00:00:00.000Z",
            }),
        })
        .expect("serialize response-metadata");

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "url",
                "id": "ann:url:https://www.rust-lang.org",
                "url": "https://www.rust-lang.org",
                "title": "Rust",
            }),
        })
        .expect("serialize url source before text");
    let frames = parse_sse_frames(&bytes);

    let added = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_item.added")
        .map(|(_, value)| value.clone())
        .expect("message scaffold output_item.added");
    let part_added = frames
        .iter()
        .find(|(ev, _)| ev == "response.content_part.added")
        .map(|(_, value)| value.clone())
        .expect("message scaffold content_part.added");
    let annotation = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_text.annotation.added")
        .map(|(_, value)| value.clone())
        .expect("annotation added");

    assert_eq!(added["item"]["type"], serde_json::json!("message"));
    assert_eq!(part_added["part"]["type"], serde_json::json!("output_text"));
    assert_eq!(
        annotation["annotation"]["type"],
        serde_json::json!("url_citation")
    );
    assert_eq!(added["item"]["id"], part_added["item_id"]);
    assert_eq!(part_added["item_id"], annotation["item_id"]);
}

#[test]
fn responses_stream_proxy_serializes_tool_input_stream_parts_as_function_call_arguments() {
    let conv = OpenAiResponsesEventConverter::new();

    let start_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");
    let _ = parse_sse_frames(&start_bytes);

    let start_tool_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-start".to_string(),
            data: serde_json::json!({
                "type": "tool-input-start",
                "id": "call_1",
                "toolName": "lookup",
            }),
        })
        .expect("serialize tool-input-start");
    let start_tool_frames = parse_sse_frames(&start_tool_bytes);
    assert!(start_tool_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.added"
            && v["item"]["type"] == serde_json::json!("function_call")
            && v["item"]["call_id"] == serde_json::json!("call_1")
            && v["item"]["name"] == serde_json::json!("lookup")
    }));

    let delta_tool_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-delta".to_string(),
            data: serde_json::json!({
                "type": "tool-input-delta",
                "id": "call_1",
                "delta": "{\"q\":\"rust\"}",
            }),
        })
        .expect("serialize tool-input-delta");
    let delta_tool_frames = parse_sse_frames(&delta_tool_bytes);
    assert!(delta_tool_frames.iter().any(|(ev, v)| {
        ev == "response.function_call_arguments.delta"
            && v["delta"] == serde_json::json!("{\"q\":\"rust\"}")
    }));

    let end_tool_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-input-end".to_string(),
            data: serde_json::json!({
                "type": "tool-input-end",
                "id": "call_1",
            }),
        })
        .expect("serialize tool-input-end");
    let end_tool_frames = parse_sse_frames(&end_tool_bytes);
    assert!(end_tool_frames.iter().any(|(ev, v)| {
        ev == "response.function_call_arguments.done"
            && v["arguments"] == serde_json::json!("{\"q\":\"rust\"}")
    }));
}

#[test]
fn responses_stream_proxy_serializes_provider_tool_call_stream_part_raw_item() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "ws_1",
                "toolName": "web_search_preview",
                "providerExecuted": true,
                "outputIndex": 1,
                "rawItem": {
                    "id": "ws_1",
                    "type": "web_search_call",
                    "status": "in_progress"
                }
            }),
        })
        .expect("serialize provider tool-call");

    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.output_item.added"
            && v["output_index"] == serde_json::json!(1)
            && v["item"]["type"] == serde_json::json!("web_search_call")
    }));
}

#[test]
fn responses_stream_proxy_serializes_provider_tool_result_stream_part_raw_item() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "ws_1",
                "toolName": "web_search_preview",
                "providerExecuted": true,
                "outputIndex": 1,
                "rawItem": {
                    "id": "ws_1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": { "type": "search", "query": "rust" },
                    "results": []
                }
            }),
        })
        .expect("serialize provider tool-result");

    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done"
            && v["output_index"] == serde_json::json!(1)
            && v["item"]["type"] == serde_json::json!("web_search_call")
            && v["item"]["status"] == serde_json::json!("completed")
    }));
}

#[test]
fn responses_stream_proxy_serializes_image_generation_preliminary_result_as_partial_image() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "ig_1",
                "toolName": "generateImage",
                "providerExecuted": true,
                "outputIndex": 1,
                "rawItem": {
                    "id": "ig_1",
                    "type": "image_generation_call",
                    "status": "in_progress"
                }
            }),
        })
        .expect("serialize image generation tool-call");

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "ig_1",
                "toolName": "generateImage",
                "result": { "result": "b64_partial" },
                "preliminary": true,
                "outputIndex": 1
            }),
        })
        .expect("serialize image generation partial image");

    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.image_generation_call.partial_image"
            && v["output_index"] == serde_json::json!(1)
            && v["item_id"] == serde_json::json!("ig_1")
            && v["partial_image_b64"] == serde_json::json!("b64_partial")
    }));
}

#[test]
fn responses_stream_proxy_reuses_output_index_for_tool_parts_without_explicit_output_index() {
    let conv = OpenAiResponsesEventConverter::new();

    let call_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "ct_1",
                "toolName": "custom_tool",
                "providerExecuted": true,
                "rawItem": {
                    "id": "ct_1",
                    "type": "custom_tool_call",
                    "status": "in_progress",
                    "name": "custom_tool",
                    "input": "{}"
                }
            }),
        })
        .expect("serialize tool-call without outputIndex");

    let call_frames = parse_sse_frames(&call_bytes);
    let call_output_index = call_frames
        .iter()
        .find_map(|(ev, v)| {
            if ev == "response.output_item.added" {
                v.get("output_index").and_then(|v| v.as_u64())
            } else {
                None
            }
        })
        .expect("output_item.added frame must exist");

    let result_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "ct_1",
                "toolName": "custom_tool",
                "providerExecuted": true,
                "rawItem": {
                    "id": "ct_1",
                    "type": "custom_tool_call",
                    "status": "completed",
                    "name": "custom_tool",
                    "input": "{}",
                    "output": { "ok": true }
                }
            }),
        })
        .expect("serialize tool-result without outputIndex");

    let result_frames = parse_sse_frames(&result_bytes);
    let result_output_index = result_frames
        .iter()
        .find_map(|(ev, v)| {
            if ev == "response.output_item.done" {
                v.get("output_index").and_then(|v| v.as_u64())
            } else {
                None
            }
        })
        .expect("output_item.done frame must exist");

    assert_eq!(call_output_index, result_output_index);
}

#[test]
fn responses_stream_proxy_serializes_tool_call_and_result_even_with_non_openai_event_type() {
    let conv = OpenAiResponsesEventConverter::new();

    let tool_call_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "gemini:tool".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "ct_1",
                "toolName": "code_execution",
                "providerExecuted": true,
                "input": { "language": "PYTHON", "code": "print(1)" }
            }),
        })
        .expect("serialize tool-call");

    let call_frames = parse_sse_frames(&tool_call_bytes);
    let call_output_index = call_frames
        .iter()
        .find_map(|(ev, v)| {
            if ev == "response.output_item.added" {
                v.get("output_index").and_then(|v| v.as_u64())
            } else {
                None
            }
        })
        .expect("output_item.added frame must exist");

    assert!(
        call_frames
            .iter()
            .any(|(ev, v)| ev == "response.function_call_arguments.done"
                && v.get("arguments")
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| s.contains("\"language\""))),
        "expected function_call_arguments.done with JSON arguments: {call_frames:?}"
    );

    let tool_result_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "gemini:tool".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "ct_1",
                "toolName": "code_execution",
                "providerExecuted": true,
                "result": { "outcome": "OUTCOME_OK", "output": "1" }
            }),
        })
        .expect("serialize tool-result");

    let result_frames = parse_sse_frames(&tool_result_bytes);
    let result_output_index = result_frames
        .iter()
        .find_map(|(ev, v)| {
            if ev == "response.output_item.done" {
                v.get("output_index").and_then(|v| v.as_u64())
            } else {
                None
            }
        })
        .expect("output_item.done frame must exist");

    assert_eq!(call_output_index, result_output_index);

    assert!(
        result_frames
            .iter()
            .any(|(ev, v)| ev == "response.output_item.done"
                && v["item"]["type"] == serde_json::json!("custom_tool_call")
                && v["item"]["output"]["output"] == serde_json::json!("1")),
        "expected synthesized custom tool output item: {result_frames:?}"
    );
}

#[test]
fn responses_stream_proxy_roundtrips_mcp_tool_parts_without_raw_item() {
    let encoder = OpenAiResponsesEventConverter::new();

    let tool_call_bytes = encoder
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "mcp_1",
                "toolName": "mcp.web_search_exa",
                "providerExecuted": true,
                "dynamic": true,
                "input": "{\"query\":\"nyc mayor\"}",
            }),
        })
        .expect("serialize mcp tool-call");
    let call_frames = parse_sse_frames(&tool_call_bytes);
    assert!(call_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.added" && v["item"]["type"] == serde_json::json!("mcp_call")
    }));
    assert!(call_frames.iter().any(|(ev, v)| {
        ev == "response.mcp_call_arguments.done"
            && v["arguments"] == serde_json::json!("{\"query\":\"nyc mayor\"}")
    }));

    let tool_result_bytes = encoder
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "mcp_1",
                "toolName": "mcp.web_search_exa",
                "providerExecuted": true,
                "dynamic": true,
                "result": {
                    "type": "call",
                    "serverLabel": "exa",
                    "name": "web_search_exa",
                    "arguments": "{\"query\":\"nyc mayor\"}",
                    "output": { "hits": 3 }
                }
            }),
        })
        .expect("serialize mcp tool-result");
    let result_frames = parse_sse_frames(&tool_result_bytes);
    assert!(result_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done"
            && v["item"]["type"] == serde_json::json!("mcp_call")
            && v["item"]["output"]["hits"] == serde_json::json!(3)
    }));

    let decoder = OpenAiResponsesEventConverter::new();
    let mut events = Vec::new();
    for (index, (event_name, payload)) in call_frames
        .into_iter()
        .chain(result_frames.into_iter())
        .enumerate()
    {
        let out = futures::executor::block_on(decoder.convert_event(eventsource_stream::Event {
            event: event_name,
            data: serde_json::to_string(&payload).expect("serialize payload"),
            id: index.to_string(),
            retry: None,
        }));
        for item in out {
            events.push(item.expect("decode roundtrip frame"));
        }
    }

    let tool_calls = events
        .iter()
        .filter_map(|event| match event.part_ref() {
            Some(crate::types::ChatStreamPart::ToolCall(call))
                if call.tool_name == "mcp.web_search_exa"
                    && call.provider_executed == Some(true) =>
            {
                Some(call)
            }
            _ => None,
        })
        .count();
    let tool_results = events
        .iter()
        .filter_map(|event| match event.part_ref() {
            Some(crate::types::ChatStreamPart::ToolResult(result))
                if result.tool_name == "mcp.web_search_exa" =>
            {
                Some(result)
            }
            _ => None,
        })
        .count();

    assert_eq!(
        tool_calls, 1,
        "expected one provider-executed mcp tool-call"
    );
    assert_eq!(
        tool_results, 1,
        "expected one provider-executed mcp tool-result"
    );
}

#[test]
fn responses_stream_proxy_synthesizes_hosted_dynamic_tool_items_without_raw_item() {
    let conv = OpenAiResponsesEventConverter::new();

    let local_call_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "call_local",
                "toolName": "shell",
                "providerExecuted": false,
                "dynamic": true,
                "input": {
                    "action": {
                        "type": "exec",
                        "command": ["pwd"],
                        "timeoutMs": 1000,
                        "workingDirectory": "/tmp",
                        "env": { "RUST_LOG": "debug" }
                    }
                },
                "providerMetadata": {
                    "openai": { "itemId": "lsh_1" }
                }
            }),
        })
        .expect("serialize local shell tool-call");
    let local_call_frames = parse_sse_frames(&local_call_bytes);
    assert!(local_call_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done"
            && v["item"]["id"] == serde_json::json!("lsh_1")
            && v["item"]["type"] == serde_json::json!("local_shell_call")
            && v["item"]["action"]["timeout_ms"] == serde_json::json!(1000)
            && v["item"]["action"]["working_directory"] == serde_json::json!("/tmp")
    }));

    let shell_call_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "call_shell",
                "toolName": "shell",
                "providerExecuted": true,
                "dynamic": true,
                "input": {
                    "action": {
                        "commands": ["echo ok"],
                        "timeoutMs": 2000,
                        "maxOutputLength": 4096
                    }
                },
                "providerMetadata": {
                    "openai": { "itemId": "sh_1" }
                }
            }),
        })
        .expect("serialize shell tool-call");
    let shell_call_frames = parse_sse_frames(&shell_call_bytes);
    assert!(shell_call_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done"
            && v["item"]["id"] == serde_json::json!("sh_1")
            && v["item"]["type"] == serde_json::json!("shell_call")
            && v["item"]["action"]["timeout_ms"] == serde_json::json!(2000)
            && v["item"]["action"]["max_output_length"] == serde_json::json!(4096)
    }));

    let apply_patch_call_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "call_apply",
                "toolName": "apply_patch",
                "providerExecuted": false,
                "dynamic": true,
                "input": {
                    "callId": "call_apply",
                    "operation": {
                        "type": "update_file",
                        "path": "src/lib.rs",
                        "diff": "@@"
                    }
                },
                "providerMetadata": {
                    "openai": { "itemId": "apc_1" }
                }
            }),
        })
        .expect("serialize apply_patch tool-call");
    let apply_patch_call_frames = parse_sse_frames(&apply_patch_call_bytes);
    assert!(apply_patch_call_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done"
            && v["item"]["id"] == serde_json::json!("apc_1")
            && v["item"]["type"] == serde_json::json!("apply_patch_call")
            && v["item"]["operation"]["type"] == serde_json::json!("update_file")
    }));

    let local_result_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "call_local",
                "toolName": "shell",
                "dynamic": true,
                "result": { "output": "ok\n" }
            }),
        })
        .expect("serialize local shell tool-result");
    let local_result_frames = parse_sse_frames(&local_result_bytes);
    assert!(local_result_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done"
            && v["item"]["type"] == serde_json::json!("local_shell_call_output")
            && v["item"]["call_id"] == serde_json::json!("call_local")
            && v["item"]["output"] == serde_json::json!("ok\n")
    }));

    let shell_result_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "call_shell",
                "toolName": "shell",
                "dynamic": true,
                "result": {
                    "output": [{
                        "stdout": "ok\n",
                        "stderr": "",
                        "outcome": { "type": "exit", "exitCode": 7 }
                    }]
                }
            }),
        })
        .expect("serialize shell tool-result");
    let shell_result_frames = parse_sse_frames(&shell_result_bytes);
    assert!(shell_result_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done"
            && v["item"]["type"] == serde_json::json!("shell_call_output")
            && v["item"]["output"][0]["outcome"]["exit_code"] == serde_json::json!(7)
    }));

    let apply_patch_result_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "call_apply",
                "toolName": "apply_patch",
                "dynamic": true,
                "result": {
                    "status": "completed",
                    "output": "patched"
                }
            }),
        })
        .expect("serialize apply_patch tool-result");
    let apply_patch_result_frames = parse_sse_frames(&apply_patch_result_bytes);
    assert!(apply_patch_result_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done"
            && v["item"]["type"] == serde_json::json!("apply_patch_call_output")
            && v["item"]["status"] == serde_json::json!("completed")
            && v["item"]["output"] == serde_json::json!("patched")
    }));
}

#[test]
fn responses_stream_decoder_maps_hosted_dynamic_output_items() {
    let conv = OpenAiResponsesEventConverter::new();

    let local_events = futures::executor::block_on(
        conv.convert_event(eventsource_stream::Event {
            event: "response.output_item.done".to_string(),
            data: serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "local_shell_call_output",
                    "call_id": "call_local",
                    "output": "ok\n"
                }
            })
            .to_string(),
            id: "1".to_string(),
            retry: None,
        }),
    );
    let local_result = local_events
        .iter()
        .find_map(|event| match stream_part(event)? {
            crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => Some(result),
            _ => None,
        })
        .expect("local shell output should decode as tool-result");
    assert_eq!(local_result.tool_call_id, "call_local");
    assert_eq!(local_result.tool_name, "shell");
    assert_eq!(local_result.result["output"], serde_json::json!("ok\n"));

    let shell_events = futures::executor::block_on(
        conv.convert_event(eventsource_stream::Event {
            event: "response.output_item.done".to_string(),
            data: serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "sho_1",
                    "type": "shell_call_output",
                    "status": "completed",
                    "call_id": "call_shell",
                    "output": [{
                        "stdout": "ok\n",
                        "stderr": "",
                        "outcome": { "type": "exit", "exit_code": 7 }
                    }]
                }
            })
            .to_string(),
            id: "2".to_string(),
            retry: None,
        }),
    );
    let shell_result = shell_events
        .iter()
        .find_map(|event| match stream_part(event)? {
            crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => Some(result),
            _ => None,
        })
        .expect("shell output should decode as tool-result");
    assert_eq!(shell_result.tool_call_id, "call_shell");
    assert_eq!(
        shell_result.result["output"][0]["outcome"]["exitCode"],
        serde_json::json!(7)
    );

    let apply_patch_events = futures::executor::block_on(
        conv.convert_event(eventsource_stream::Event {
            event: "response.output_item.done".to_string(),
            data: serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 2,
                "item": {
                    "type": "apply_patch_call_output",
                    "call_id": "call_apply",
                    "status": "failed",
                    "output": "conflict"
                }
            })
            .to_string(),
            id: "3".to_string(),
            retry: None,
        }),
    );
    let apply_patch_result = apply_patch_events
        .iter()
        .find_map(|event| match stream_part(event)? {
            crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => Some(result),
            _ => None,
        })
        .expect("apply_patch output should decode as tool-result");
    assert_eq!(apply_patch_result.tool_call_id, "call_apply");
    assert_eq!(apply_patch_result.tool_name, "apply_patch");
    assert_eq!(
        apply_patch_result.result["status"],
        serde_json::json!("failed")
    );
    assert_eq!(
        apply_patch_result.result["output"],
        serde_json::json!("conflict")
    );
}

#[test]
fn responses_stream_bridge_maps_gemini_tool_events_to_openai_output_items() {
    let conv = OpenAiResponsesEventConverter::new();
    let mut bridge = crate::streaming::OpenAiResponsesStreamPartsBridge::new();

    let tool_call = crate::streaming::ChatStreamEvent::Custom {
        event_type: "gemini:tool".to_string(),
        data: serde_json::json!({
            "type": "tool-call",
            "toolCallId": "call_1",
            "toolName": "code_execution",
            "providerExecuted": true,
            "input": { "language": "PYTHON", "code": "print(1)" }
        }),
    };

    let tool_result = crate::streaming::ChatStreamEvent::Custom {
        event_type: "gemini:tool".to_string(),
        data: serde_json::json!({
            "type": "tool-result",
            "toolCallId": "call_1",
            "toolName": "code_execution",
            "providerExecuted": true,
            "result": { "outcome": "OUTCOME_OK", "output": "1" }
        }),
    };

    let mut frames: Vec<(String, serde_json::Value)> = Vec::new();
    for ev in bridge.bridge_event(tool_call) {
        let bytes = conv
            .serialize_event(&ev)
            .expect("serialize bridged tool-call");
        frames.extend(parse_sse_frames(&bytes));
    }
    for ev in bridge.bridge_event(tool_result) {
        let bytes = conv
            .serialize_event(&ev)
            .expect("serialize bridged tool-result");
        frames.extend(parse_sse_frames(&bytes));
    }

    let added = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_item.added");
    let done = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_item.done");
    assert!(
        added.is_some(),
        "tool-call should produce output_item.added"
    );
    assert!(
        done.is_some(),
        "tool-result should produce output_item.done"
    );

    let added_output_index = added
        .and_then(|(_, v)| v.get("output_index").and_then(|v| v.as_u64()))
        .unwrap_or_default();
    let done_output_index = done
        .and_then(|(_, v)| v.get("output_index").and_then(|v| v.as_u64()))
        .unwrap_or_default();

    assert_eq!(added_output_index, done_output_index);
}

#[test]
fn responses_stream_bridge_synthesizes_tool_call_when_only_result_is_available() {
    let conv = OpenAiResponsesEventConverter::new();
    let mut bridge = crate::streaming::OpenAiResponsesStreamPartsBridge::new();

    let tool_result_only = crate::streaming::ChatStreamEvent::Custom {
        event_type: "anthropic:tool-result".to_string(),
        data: serde_json::json!({
            "type": "tool-result",
            "toolCallId": "call_2",
            "toolName": "web_search",
            "providerExecuted": true,
            "isError": false,
            "result": [{ "type": "web_search_result", "url": "https://example.com", "title": "Example" }]
        }),
    };

    let mut frames: Vec<(String, serde_json::Value)> = Vec::new();
    for ev in bridge.bridge_event(tool_result_only) {
        let bytes = conv
            .serialize_event(&ev)
            .expect("serialize bridged tool-result-only");
        frames.extend(parse_sse_frames(&bytes));
    }

    let added = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_item.added");
    let done = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_item.done");
    assert!(
        added.is_some(),
        "bridged tool-result should synthesize output_item.added"
    );
    assert!(
        done.is_some(),
        "bridged tool-result should produce output_item.done"
    );

    let added_output_index = added
        .and_then(|(_, v)| v.get("output_index").and_then(|v| v.as_u64()))
        .unwrap_or_default();
    let done_output_index = done
        .and_then(|(_, v)| v.get("output_index").and_then(|v| v.as_u64()))
        .unwrap_or_default();

    assert_eq!(added_output_index, done_output_index);
}

#[test]
fn responses_stream_proxy_serializes_tool_approval_request_stream_part_raw_item() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:tool-approval-request".to_string(),
            data: serde_json::json!({
                "type": "tool-approval-request",
                "approvalId": "apr_1",
                "toolCallId": "mcp_approval_1",
                "outputIndex": 2,
                "rawItem": {
                    "id": "apr_1",
                    "type": "mcp_approval_request",
                    "status": "completed"
                }
            }),
        })
        .expect("serialize tool approval request");

    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done"
            && v["output_index"] == serde_json::json!(2)
            && v["item"]["type"] == serde_json::json!("mcp_approval_request")
    }));
}

#[test]
fn responses_stream_proxy_serializes_stream_start_and_response_metadata_parts() {
    let conv = OpenAiResponsesEventConverter::new();

    let start_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::StreamStart { warnings: vec![] },
        })
        .expect("serialize stream-start");
    assert!(start_bytes.is_empty());

    let meta_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ResponseMetadata(crate::types::ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: Some(parse_test_timestamp("2025-01-01T00:00:00.000Z")),
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            }),
        })
        .expect("serialize response-metadata");

    let frames = parse_sse_frames(&meta_bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].0, "response.created");
    assert_eq!(
        frames[0].1["response"]["id"],
        serde_json::json!("resp_test")
    );
    assert_eq!(
        frames[0].1["response"]["model"],
        serde_json::json!("gpt-test")
    );
}

#[test]
fn responses_stream_proxy_serializes_text_start_and_end_parts() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ResponseMetadata(crate::types::ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: Some(parse_test_timestamp("2025-01-01T00:00:00.000Z")),
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            }),
        })
        .expect("serialize response-metadata");

    let start_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextStart {
                id: "msg_1".to_string(),
                provider_metadata: Some(openai_provider_metadata(serde_json::json!({
                    "itemId": "msg_1"
                }))),
            },
        })
        .expect("serialize text-start");
    let start_frames = parse_sse_frames(&start_bytes);
    assert!(start_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.added" && v["item"]["type"] == serde_json::json!("message")
    }));
    assert!(
        start_frames
            .iter()
            .any(|(ev, _)| ev == "response.content_part.added")
    );

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextDelta {
                id: "msg_1".to_string(),
                delta: "Hello".to_string(),
                provider_metadata: None,
            },
        })
        .expect("serialize text-delta");

    let end_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextEnd {
                id: "msg_1".to_string(),
                provider_metadata: Some(openai_provider_metadata(serde_json::json!({
                    "itemId": "msg_1"
                }))),
            },
        })
        .expect("serialize text-end");
    let end_frames = parse_sse_frames(&end_bytes);
    assert!(end_frames.iter().any(
        |(ev, v)| ev == "response.output_text.done" && v["text"] == serde_json::json!("Hello")
    ));
    assert!(
        end_frames
            .iter()
            .any(|(ev, _)| ev == "response.output_item.done")
    );
}

#[test]
fn responses_stream_proxy_serializes_reasoning_start_and_end_parts() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ResponseMetadata(crate::types::ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: Some(parse_test_timestamp("2025-01-01T00:00:00.000Z")),
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            }),
        })
        .expect("serialize response-metadata");

    let start_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ReasoningStart {
                id: "rs_1:0".to_string(),
                provider_metadata: Some(openai_provider_metadata(serde_json::json!({
                    "itemId": "rs_1"
                }))),
            },
        })
        .expect("serialize reasoning-start");
    let start_frames = parse_sse_frames(&start_bytes);
    assert!(start_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.added" && v["item"]["type"] == serde_json::json!("reasoning")
    }));

    let end_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ReasoningEnd {
                id: "rs_1:0".to_string(),
                provider_metadata: Some(openai_provider_metadata(serde_json::json!({
                    "itemId": "rs_1"
                }))),
            },
        })
        .expect("serialize reasoning-end");
    let end_frames = parse_sse_frames(&end_bytes);
    assert!(end_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.done" && v["item"]["type"] == serde_json::json!("reasoning")
    }));
}

#[test]
fn responses_stream_proxy_serializes_finish_part_as_response_completed() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ResponseMetadata(crate::types::ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: Some(parse_test_timestamp("2025-01-01T00:00:00.000Z")),
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            }),
        })
        .expect("serialize response-metadata");

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextDelta {
                id: "msg_1".to_string(),
                delta: "Hello".to_string(),
                provider_metadata: None,
            },
        })
        .expect("serialize text-delta");

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::Finish {
                usage: Usage::builder()
                    .with_input_total_tokens(3)
                    .with_input_no_cache_tokens(3)
                    .with_input_cache_read_tokens(0)
                    .with_output_total_tokens(5)
                    .with_output_text_tokens(5)
                    .with_output_reasoning_tokens(0)
                    .build(),
                finish_reason: crate::types::ChatStreamFinishInfo {
                    unified: FinishReason::Stop,
                    raw: None,
                },
                provider_metadata: Some(openai_provider_metadata(serde_json::json!({
                    "responseId": "resp_test"
                }))),
            },
        })
        .expect("serialize finish");
    let frames = parse_sse_frames(&bytes);
    let completed = frames
        .iter()
        .find(|(ev, _)| ev == "response.completed")
        .map(|(_, value)| value)
        .expect("response.completed frame");
    assert_eq!(completed["response"]["id"], serde_json::json!("resp_test"));
    assert_eq!(
        completed["response"]["finish_reason"],
        serde_json::json!("stop")
    );
    assert_eq!(
        completed["response"]["status"],
        serde_json::json!("completed")
    );
    assert_eq!(
        completed["response"]["usage"]["input_tokens"],
        serde_json::json!(3)
    );
    assert_eq!(
        completed["response"]["usage"]["output_tokens"],
        serde_json::json!(5)
    );
    assert_eq!(
        completed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
        serde_json::json!(0)
    );
    assert_eq!(
        completed["response"]["usage"]["output_tokens_details"]["reasoning_tokens"],
        serde_json::json!(0)
    );
}

#[test]
fn responses_stream_proxy_serializes_raw_incomplete_finish_reason() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:finish".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "finishReason": { "raw": "max_output_tokens", "unified": "length" },
                "usage": {
                    "inputTokens": { "total": 1, "cacheRead": 0, "cacheWrite": null, "noCache": 1 },
                    "outputTokens": { "total": 2, "reasoning": 0, "text": 2 },
                    "raw": null
                }
            }),
        })
        .expect("serialize finish");

    let frames = parse_sse_frames(&bytes);
    let incomplete = frames
        .iter()
        .find(|(ev, _)| ev == "response.incomplete")
        .map(|(_, value)| value)
        .expect("response.incomplete frame");
    assert_eq!(incomplete["type"], serde_json::json!("response.incomplete"));
    assert_eq!(
        incomplete["response"]["status"],
        serde_json::json!("incomplete")
    );
    assert_eq!(
        incomplete["response"]["incomplete_details"]["reason"],
        serde_json::json!("max_output_tokens")
    );
    assert_eq!(
        incomplete["response"]["finish_reason"],
        serde_json::json!("length")
    );
}

#[test]
fn responses_stream_proxy_maps_v3_tool_finish_to_tool_calls_finish_reason() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:finish".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "finishReason": { "raw": null, "unified": "tool-calls" },
                "usage": {
                    "inputTokens": { "total": 1, "cacheRead": 0, "cacheWrite": null, "noCache": 1 },
                    "outputTokens": { "total": 2, "reasoning": 0, "text": 2 },
                    "raw": null
                }
            }),
        })
        .expect("serialize finish");

    let completed = parse_sse_frames(&bytes)
        .into_iter()
        .find(|(ev, _)| ev == "response.completed")
        .map(|(_, value)| value)
        .expect("response.completed frame");
    assert_eq!(
        completed["response"]["finish_reason"],
        serde_json::json!("tool_calls")
    );
}

#[test]
fn responses_failed_event_buffers_finish_with_unknown_usage_totals() {
    let conv = OpenAiResponsesEventConverter::new();
    let event = eventsource_stream::Event {
        event: "response.failed".to_string(),
        data: r#"{"type":"response.failed","response":{"id":"resp_failed_1","status":"failed"}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };

    let events = futures::executor::block_on(conv.convert_event(event));
    assert!(
        events.is_empty(),
        "response.failed should only buffer finish"
    );

    let pending = conv.handle_stream_end_events();
    assert_eq!(pending.len(), 1);

    match stream_part(&pending[0]).expect("pending finish part") {
        crate::streaming::LanguageModelV3StreamPart::Finish {
            usage,
            finish_reason,
            ..
        } => {
            assert_eq!(finish_reason.unified, "other");
            assert_eq!(usage.input_tokens.total, None);
            assert_eq!(usage.output_tokens.total, None);
            assert_eq!(usage.raw, None);
        }
        other => panic!("expected finish part, got {other:?}"),
    }
}

#[test]
fn responses_custom_tool_input_stream_events_emit_stable_tool_input_parts() {
    let conv = OpenAiResponsesEventConverter::new();

    let added = eventsource_stream::Event {
        event: "".to_string(),
        data: r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"ct_1","type":"custom_tool_call","name":"x_keyword_search","status":"in_progress"}}"#
            .to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let added_events = futures::executor::block_on(conv.convert_event(added));
    assert_eq!(added_events.len(), 1);
    match stream_part(&added_events[0]).expect("tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart { id, tool_name, .. } => {
            assert_eq!(id, "ct_1");
            assert_eq!(tool_name, "x_keyword_search");
        }
        other => panic!("expected tool-input-start part, got {other:?}"),
    }

    let delta = eventsource_stream::Event {
        event: "response.custom_tool_call_input.delta".to_string(),
        data: r#"{"type":"response.custom_tool_call_input.delta","item_id":"ct_1","delta":"{\"q\":\"rust\""}"#
            .to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let delta_events = futures::executor::block_on(conv.convert_event(delta));
    assert_eq!(delta_events.len(), 1);
    match stream_part(&delta_events[0]).expect("tool-input-delta part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputDelta { id, delta, .. } => {
            assert_eq!(id, "ct_1");
            assert_eq!(delta, "{\"q\":\"rust\"");
        }
        other => panic!("expected tool-input-delta part, got {other:?}"),
    }

    let done = eventsource_stream::Event {
        event: "response.custom_tool_call_input.done".to_string(),
        data: r#"{"type":"response.custom_tool_call_input.done","item_id":"ct_1","input":"{\"q\":\"rust\"}"}"#
            .to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let done_events = futures::executor::block_on(conv.convert_event(done));
    assert_eq!(done_events.len(), 1);
    match stream_part(&done_events[0]).expect("tool-input-end part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputEnd { id, .. } => {
            assert_eq!(id, "ct_1");
        }
        other => panic!("expected tool-input-end part, got {other:?}"),
    }
}

#[test]
fn responses_stream_proxy_serializes_failed_finish_with_null_usage_totals() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:error".to_string(),
            data: serde_json::json!({
                "type": "error",
                "error": {
                    "message": "boom"
                }
            }),
        })
        .expect("serialize error");

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:finish".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "finishReason": { "raw": null, "unified": "other" },
                "usage": {
                    "inputTokens": { "total": null, "cacheRead": null, "cacheWrite": null, "noCache": null },
                    "outputTokens": { "total": null, "reasoning": null, "text": null },
                    "raw": null
                }
            }),
        })
        .expect("serialize failed finish");

    let failed = parse_sse_frames(&bytes)
        .into_iter()
        .find(|(ev, _)| ev == "response.failed")
        .map(|(_, value)| value)
        .expect("response.failed frame");

    assert_eq!(failed["response"]["status"], serde_json::json!("failed"));
    assert_eq!(
        failed["response"]["usage"]["input_tokens"],
        serde_json::Value::Null
    );
    assert_eq!(
        failed["response"]["usage"]["output_tokens"],
        serde_json::Value::Null
    );
    assert_eq!(
        failed["response"]["usage"]["total_tokens"],
        serde_json::Value::Null
    );
}

#[test]
fn responses_stream_proxy_preserves_source_metadata_in_completed_response() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ResponseMetadata(crate::types::ResponseMetadata {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                created: Some(parse_test_timestamp("2025-01-01T00:00:00.000Z")),
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
            }),
        })
        .expect("serialize response-metadata");

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextDelta {
                id: "msg_1".to_string(),
                delta: "See files.".to_string(),
                provider_metadata: None,
            },
        })
        .expect("serialize text-delta");

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::Source {
                id: "ann:doc:file_container_1".to_string(),
                source: crate::types::SourcePart::Document {
                    media_type: "text/plain".to_string(),
                    title: "Bundle".to_string(),
                    filename: Some("bundle.txt".to_string()),
                },
                provider_metadata: Some(openai_provider_metadata(serde_json::json!({
                    "type": "container_file_citation",
                    "fileId": "file_container_1",
                    "containerId": "container_42",
                    "index": 3
                }))),
            },
        })
        .expect("serialize container source");

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::Source {
                id: "ann:doc:file_path_9".to_string(),
                source: crate::types::SourcePart::Document {
                    media_type: "application/octet-stream".to_string(),
                    title: "artifact.bin".to_string(),
                    filename: Some("artifact.bin".to_string()),
                },
                provider_metadata: Some(openai_provider_metadata(serde_json::json!({
                    "type": "file_path",
                    "fileId": "file_path_9",
                    "index": 5
                }))),
            },
        })
        .expect("serialize file path source");

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::Finish {
                usage: Usage::builder()
                    .with_input_total_tokens(3)
                    .with_input_no_cache_tokens(3)
                    .with_input_cache_read_tokens(0)
                    .with_output_total_tokens(5)
                    .with_output_text_tokens(5)
                    .with_output_reasoning_tokens(0)
                    .build(),
                finish_reason: crate::types::ChatStreamFinishInfo {
                    unified: FinishReason::Stop,
                    raw: None,
                },
                provider_metadata: Some(openai_provider_metadata(serde_json::json!({
                    "responseId": "resp_test",
                    "serviceTier": "default"
                }))),
            },
        })
        .expect("serialize finish");

    let completed = parse_sse_frames(&bytes)
        .into_iter()
        .find(|(ev, _)| ev == "response.completed")
        .map(|(_, value)| value)
        .expect("response.completed frame");

    let tx = crate::standards::openai::transformers::OpenAiResponsesResponseTransformer::new();
    let response =
        crate::execution::transformers::response::ResponseTransformer::transform_chat_response(
            &tx, &completed,
        )
        .expect("transform completed response");
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));

    let meta = crate::provider_metadata::openai::OpenAiChatResponseExt::openai_metadata(&response)
        .expect("openai metadata");
    assert_eq!(meta.response_id.as_deref(), Some("resp_test"));
    assert_eq!(meta.service_tier.as_deref(), Some("default"));
    let sources = meta.sources.expect("sources present");

    let container_source = sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("container source present");
    let container_meta =
        crate::provider_metadata::openai::OpenAiSourceExt::openai_metadata(container_source)
            .expect("container source metadata");
    assert_eq!(
        container_meta.metadata_type.as_deref(),
        Some("container_file_citation")
    );
    assert_eq!(container_meta.file_id.as_deref(), Some("file_container_1"));
    assert_eq!(container_meta.container_id.as_deref(), Some("container_42"));
    assert_eq!(container_meta.index, Some(3));

    let file_path_source = sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("file path source present");
    let file_path_meta =
        crate::provider_metadata::openai::OpenAiSourceExt::openai_metadata(file_path_source)
            .expect("file path source metadata");
    assert_eq!(file_path_meta.metadata_type.as_deref(), Some("file_path"));
    assert_eq!(file_path_meta.file_id.as_deref(), Some("file_path_9"));
    assert!(file_path_meta.container_id.is_none());
    assert_eq!(file_path_meta.index, Some(5));
}

#[test]
fn responses_stream_proxy_serializes_error_part_as_response_error() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:error".to_string(),
            data: serde_json::json!({
                "type": "error",
                "error": { "error": { "message": "boom" } }
            }),
        })
        .expect("serialize error");
    let frames = parse_sse_frames(&bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].0, "response.error");
    assert_eq!(frames[0].1["error"]["message"], serde_json::json!("boom"));
}

#[test]
fn responses_stream_proxy_serializes_tool_call_delta() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::ToolCallDelta {
            id: "call_1".to_string(),
            function_name: Some("lookup".to_string()),
            arguments_delta: Some("{\"q\":\"rust\"}".to_string()),
            index: None,
        })
        .expect("serialize tool delta");

    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.output_item.added"
            && v["item"]["type"] == serde_json::json!("function_call")
            && v["item"]["call_id"] == serde_json::json!("call_1")
            && v["item"]["name"] == serde_json::json!("lookup")
    }));
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.function_call_arguments.delta"
            && v["delta"] == serde_json::json!("{\"q\":\"rust\"}")
    }));
}

#[test]
fn responses_stream_proxy_serializes_response_error() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Error {
            error: "Upstream failure".to_string(),
        })
        .expect("serialize error");
    let frames = parse_sse_frames(&bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].0, "response.error");
    assert_eq!(
        frames[0].1["error"]["message"],
        serde_json::json!("Upstream failure")
    );
}

#[test]
fn responses_stream_proxy_serializes_reasoning_delta() {
    let conv = OpenAiResponsesEventConverter::new();

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::ThinkingDelta {
            delta: "think".to_string(),
        })
        .expect("serialize reasoning delta");
    let frames = parse_sse_frames(&bytes);

    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.output_item.added" && v["item"]["type"] == serde_json::json!("reasoning")
    }));
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.reasoning_summary_text.delta" && v["delta"] == serde_json::json!("think")
    }));
}

#[test]
fn responses_stream_proxy_closes_function_call_on_stream_end() {
    let conv = OpenAiResponsesEventConverter::new();

    let _ = conv.serialize_event(&crate::streaming::ChatStreamEvent::StreamStart {
        metadata: ResponseMetadata {
            id: Some("resp_test".to_string()),
            model: Some("gpt-test".to_string()),
            created: None,
            provider: "openai".to_string(),
            request_id: None,
            headers: None,
        },
    });

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::ToolCallDelta {
            id: "call_1".to_string(),
            function_name: Some("lookup".to_string()),
            arguments_delta: Some("{\"q\":\"rust".to_string()),
            index: Some(0),
        })
        .expect("serialize tool delta 1");
    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::ToolCallDelta {
            id: "call_1".to_string(),
            function_name: None,
            arguments_delta: Some("\"}".to_string()),
            index: Some(0),
        })
        .expect("serialize tool delta 2");

    let end_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: Some("resp_test".to_string()),
                model: Some("gpt-test".to_string()),
                content: MessageContent::Text(String::new()),
                usage: None,
                finish_reason: Some(FinishReason::Stop),
                raw_finish_reason: None,
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            },
        })
        .expect("serialize end");

    let frames = parse_sse_frames(&end_bytes);
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.function_call_arguments.done"
            && v["arguments"] == serde_json::json!("{\"q\":\"rust\"}")
    }));
    assert!(frames.iter().any(|(ev, v)| {
        ev == "response.completed"
            && v["response"]["output"]
                .as_array()
                .is_some_and(|arr| arr.iter().any(|it| it["type"] == "function_call"))
    }));
}
