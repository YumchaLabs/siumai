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
        data: r#"{"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}"#.to_string(),
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
    assert_eq!(out_added.len(), 3);
    match out_added[0].as_ref().unwrap() {
        crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "openai:tool-input-start");
            assert_eq!(data["id"], serde_json::json!("ws_1"));
            assert_eq!(data["toolName"], serde_json::json!("web_search"));
            assert_eq!(data["providerExecuted"], serde_json::json!(true));
        }
        other => panic!("expected Custom tool-input-start, got {other:?}"),
    }
    match out_added[1].as_ref().unwrap() {
        crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "openai:tool-input-end");
            assert_eq!(data["id"], serde_json::json!("ws_1"));
        }
        other => panic!("expected Custom tool-input-end, got {other:?}"),
    }
    match out_added[2].as_ref().unwrap() {
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
    match out_added[0].as_ref().unwrap() {
        crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "openai:tool-input-start");
            assert_eq!(data["toolName"], serde_json::json!("web_search_preview"));
        }
        other => panic!("expected Custom tool-input-start, got {other:?}"),
    }
    match out_added[1].as_ref().unwrap() {
        crate::streaming::ChatStreamEvent::Custom { event_type, .. } => {
            assert_eq!(event_type, "openai:tool-input-end");
        }
        other => panic!("expected Custom tool-input-end, got {other:?}"),
    }
    match out_added[2].as_ref().unwrap() {
        crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "openai:tool-call");
            assert_eq!(data["toolName"], serde_json::json!("web_search_preview"));
        }
        other => panic!("expected Custom tool-call, got {other:?}"),
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
    match out_done[0].as_ref().unwrap() {
        crate::streaming::ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "openai:tool-result");
            assert_eq!(data["toolName"], serde_json::json!("web_search_preview"));
        }
        other => panic!("expected Custom tool-result, got {other:?}"),
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
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            },
        })
        .expect("serialize end");
    let end_frames = parse_sse_frames(&end_bytes);
    assert!(end_frames.iter().any(|(ev, v)| {
        ev == "response.completed" && v["type"] == serde_json::json!("response.completed")
    }));
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
                "providerMetadata": { "openai": { "fileId": "file_123" } },
            }),
        })
        .expect("serialize doc source");
    let doc_frames = parse_sse_frames(&doc_bytes);
    assert!(doc_frames.iter().any(|(ev, v)| {
        ev == "response.output_text.annotation.added"
            && v["annotation"]["type"] == serde_json::json!("file_citation")
            && v["annotation"]["file_id"] == serde_json::json!("file_123")
            && v["annotation"]["filename"] == serde_json::json!("notes.txt")
    }));
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
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:stream-start".to_string(),
            data: serde_json::json!({
                "type": "stream-start",
                "warnings": [],
            }),
        })
        .expect("serialize stream-start");
    assert!(start_bytes.is_empty());

    let meta_bytes = conv
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

    let start_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-start".to_string(),
            data: serde_json::json!({
                "type": "text-start",
                "id": "msg_1",
                "providerMetadata": { "openai": { "itemId": "msg_1" } },
            }),
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
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": "msg_1",
                "delta": "Hello",
            }),
        })
        .expect("serialize text-delta");

    let end_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-end".to_string(),
            data: serde_json::json!({
                "type": "text-end",
                "id": "msg_1",
                "providerMetadata": { "openai": { "itemId": "msg_1" } },
            }),
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

    let start_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:reasoning-start".to_string(),
            data: serde_json::json!({
                "type": "reasoning-start",
                "id": "rs_1:0",
                "providerMetadata": { "openai": { "itemId": "rs_1" } },
            }),
        })
        .expect("serialize reasoning-start");
    let start_frames = parse_sse_frames(&start_bytes);
    assert!(start_frames.iter().any(|(ev, v)| {
        ev == "response.output_item.added" && v["item"]["type"] == serde_json::json!("reasoning")
    }));

    let end_bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:reasoning-end".to_string(),
            data: serde_json::json!({
                "type": "reasoning-end",
                "id": "rs_1:0",
                "providerMetadata": { "openai": { "itemId": "rs_1" } },
            }),
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

    let _ = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": "msg_1",
                "delta": "Hello",
            }),
        })
        .expect("serialize text-delta");

    let bytes = conv
        .serialize_event(&crate::streaming::ChatStreamEvent::Custom {
            event_type: "openai:finish".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "finishReason": { "raw": null, "unified": "stop" },
                "providerMetadata": { "openai": { "responseId": "resp_test" } },
                "usage": {
                    "inputTokens": { "total": 3, "cacheRead": 0, "cacheWrite": null, "noCache": 3 },
                    "outputTokens": { "total": 5, "reasoning": 0, "text": 5 },
                    "raw": null
                }
            }),
        })
        .expect("serialize finish");
    let frames = parse_sse_frames(&bytes);
    assert!(frames.iter().any(|(ev, v)| ev == "response.completed"
        && v["response"]["id"] == serde_json::json!("resp_test")));
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
