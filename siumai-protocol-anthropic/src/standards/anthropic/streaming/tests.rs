use super::super::params::AnthropicParams;
use super::*;
use eventsource_stream::Event;

fn create_test_config() -> AnthropicParams {
    AnthropicParams::default()
}

#[tokio::test]
async fn test_anthropic_streaming_conversion() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    // Test content delta conversion
    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(!result.is_empty());

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = result.first() {
        assert_eq!(delta, "Hello");
    } else {
        panic!("Expected ContentDelta event");
    }
}

#[tokio::test]
async fn test_anthropic_streaming_error_event_is_exposed() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let event = Event {
        event: "error".to_string(),
        data: r#"{"type":"error","error":{"type":"overloaded_error","message":"rate limited"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    let err = result
        .iter()
        .find(|e| matches!(e, Ok(ChatStreamEvent::Error { .. })));
    match err {
        Some(Ok(ChatStreamEvent::Error { error })) => {
            assert!(error.contains("rate limited"));
        }
        other => panic!("Expected Error event, got: {other:?}"),
    }
}

#[tokio::test]
async fn test_anthropic_streaming_error_event_without_type_is_exposed() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let event = Event {
        event: "error".to_string(),
        data: r#"{"error":{"message":"bad request"}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    let err = result
        .iter()
        .find(|e| matches!(e, Ok(ChatStreamEvent::Error { .. })));
    match err {
        Some(Ok(ChatStreamEvent::Error { error })) => {
            assert!(error.contains("bad request"));
        }
        other => panic!("Expected Error event, got: {other:?}"),
    }
}

#[tokio::test]
async fn test_anthropic_streaming_ping_event_is_ignored() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let event = Event {
        event: "ping".to_string(),
        data: r#"{"type":"ping"}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(result.is_empty());
}

#[tokio::test]
async fn test_anthropic_stream_end_is_error_after_error_event() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let event = Event {
        event: "error".to_string(),
        data: r#"{"type":"error","error":{"type":"api_error","message":"nope"}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let _ = converter.convert_event(event).await;

    let end = converter
        .handle_stream_end()
        .expect("expected stream end after error")
        .expect("expected Ok stream end");

    match end {
        ChatStreamEvent::StreamEnd { response } => {
            assert!(matches!(response.finish_reason, Some(FinishReason::Error)));
        }
        other => panic!("Expected StreamEnd event, got: {other:?}"),
    }
}

#[tokio::test]
async fn test_anthropic_stream_finish_includes_context_management_provider_metadata() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let start = Event {
        event: "".to_string(),
        data: r#"{"type":"message_start","message":{"id":"msg_test","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let _ = converter.convert_event(start).await;

    let delta = Event {
        event: "".to_string(),
        data: r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"context_management":{"applied_edits":[{"type":"clear_tool_uses_20250919","cleared_tool_uses":5,"cleared_input_tokens":10000}]}},"usage":{"input_tokens":1,"output_tokens":1}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out = converter.convert_event(delta).await;

    let finish = out.iter().find_map(|r| match r.as_ref().ok() {
        Some(ChatStreamEvent::Custom { data, .. })
            if data.get("type") == Some(&serde_json::json!("finish")) =>
        {
            Some(data.clone())
        }
        _ => None,
    });
    let finish = finish.expect("expected finish event");

    let cm = finish["providerMetadata"]["anthropic"]["contextManagement"].clone();
    assert_eq!(
        cm["appliedEdits"][0]["type"],
        serde_json::json!("clear_tool_uses_20250919")
    );
    assert_eq!(
        cm["appliedEdits"][0]["clearedToolUses"],
        serde_json::json!(5)
    );
    assert_eq!(
        cm["appliedEdits"][0]["clearedInputTokens"],
        serde_json::json!(10000)
    );

    let end = out.iter().find_map(|r| match r.as_ref().ok() {
        Some(ChatStreamEvent::StreamEnd { response }) => Some(response.clone()),
        _ => None,
    });
    let end = end.expect("expected StreamEnd");

    let cm = end
        .provider_metadata
        .as_ref()
        .and_then(|m| m.get("anthropic"))
        .and_then(|m| m.get("contextManagement"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    assert_eq!(
        cm["appliedEdits"][0]["clearedToolUses"],
        serde_json::json!(5)
    );
}

// Removed legacy merge-provider-params test; behavior now covered by transformers

#[tokio::test]
async fn test_anthropic_stream_end() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let result = converter.handle_stream_end();
    assert!(result.is_some());

    if let Some(Ok(ChatStreamEvent::StreamEnd { .. })) = result {
        // Success
    } else {
        panic!("Expected StreamEnd event");
    }
}

#[tokio::test]
async fn emits_custom_events_for_server_tool_use_and_results() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let tool_call_event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"rust"}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(tool_call_event).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:tool-call");
            assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_1"));
            assert_eq!(data["toolName"], serde_json::json!("web_search"));
            assert_eq!(data["providerExecuted"], serde_json::json!(true));
        }
        other => panic!("Expected Custom event, got {:?}", other),
    }

    let tool_result_event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","title":"Rust","url":"https://www.rust-lang.org","encrypted_content":"..."}]}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(tool_result_event).await;
    assert_eq!(evs.len(), 2);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:tool-result");
            assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_1"));
            assert_eq!(data["toolName"], serde_json::json!("web_search"));
            assert_eq!(data["providerExecuted"], serde_json::json!(true));
            assert!(data["result"].is_array());
        }
        other => panic!("Expected Custom event, got {:?}", other),
    }

    match evs.get(1).unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:source");
            assert_eq!(data["sourceType"], serde_json::json!("url"));
            assert_eq!(data["url"], serde_json::json!("https://www.rust-lang.org"));
        }
        other => panic!("Expected source Custom event, got {:?}", other),
    }

    let web_fetch_result_event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":2,"content_block":{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_2","content":{"type":"web_fetch_result","url":"https://example.com","retrieved_at":"2025-01-01T00:00:00Z","content":{"type":"document","title":"Example","citations":{"enabled":true},"source":{"type":"text","media_type":"text/plain","data":"hello"}}}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(web_fetch_result_event).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:tool-result");
            assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_2"));
            assert_eq!(data["toolName"], serde_json::json!("web_fetch"));
            assert_eq!(data["isError"], serde_json::json!(false));
            assert_eq!(
                data["result"]["type"],
                serde_json::json!("web_fetch_result")
            );
            assert_eq!(
                data["result"]["retrievedAt"],
                serde_json::json!("2025-01-01T00:00:00Z")
            );
            assert_eq!(
                data["result"]["content"]["source"]["mediaType"],
                serde_json::json!("text/plain")
            );
        }
        other => panic!("Expected tool-result Custom event, got {:?}", other),
    }

    let tool_search_call_event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":3,"content_block":{"type":"server_tool_use","id":"srvtoolu_3","name":"tool_search_tool_regex","input":{"pattern":"weather","limit":2}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(tool_search_call_event).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:tool-call");
            assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_3"));
            assert_eq!(data["toolName"], serde_json::json!("tool_search"));
            assert_eq!(data["input"]["pattern"], serde_json::json!("weather"));
        }
        other => panic!("Expected tool-call Custom event, got {:?}", other),
    }

    let tool_search_result_event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":4,"content_block":{"type":"tool_search_tool_result","tool_use_id":"srvtoolu_3","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"get_weather"}]}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(tool_search_result_event).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:tool-result");
            assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_3"));
            assert_eq!(data["toolName"], serde_json::json!("tool_search"));
            assert_eq!(data["isError"], serde_json::json!(false));
            assert!(data["result"].is_array());
            assert_eq!(
                data["result"][0]["toolName"],
                serde_json::json!("get_weather")
            );
        }
        other => panic!("Expected tool-result Custom event, got {:?}", other),
    }

    let code_exec_call_event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":5,"content_block":{"type":"server_tool_use","id":"srvtoolu_4","name":"code_execution","input":{"code":"print(1+1)"}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(code_exec_call_event).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:tool-call");
            assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_4"));
            assert_eq!(data["toolName"], serde_json::json!("code_execution"));
        }
        other => panic!("Expected tool-call Custom event, got {:?}", other),
    }

    let code_exec_result_event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":6,"content_block":{"type":"code_execution_tool_result","tool_use_id":"srvtoolu_4","content":{"type":"code_execution_result","stdout":"2\n","stderr":"","return_code":0}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(code_exec_result_event).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:tool-result");
            assert_eq!(data["toolCallId"], serde_json::json!("srvtoolu_4"));
            assert_eq!(data["toolName"], serde_json::json!("code_execution"));
            assert_eq!(data["isError"], serde_json::json!(false));
            assert_eq!(
                data["result"]["type"],
                serde_json::json!("code_execution_result")
            );
            assert_eq!(data["result"]["return_code"], serde_json::json!(0));
        }
        other => panic!("Expected tool-result Custom event, got {:?}", other),
    }
}

#[tokio::test]
async fn emits_tool_call_delta_for_local_tool_use_input_json_delta() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let start = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"location":"tokyo"}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(start).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            index,
        } => {
            assert_eq!(id, "toolu_1");
            assert_eq!(function_name.as_deref(), Some("get_weather"));
            assert!(arguments_delta.is_none());
            assert_eq!(*index, Some(0));
        }
        other => panic!("Expected ToolCallDelta, got {:?}", other),
    }

    let delta = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"unit\":\"c\"}"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(delta).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            index,
        } => {
            assert_eq!(id, "toolu_1");
            assert!(function_name.is_none());
            assert_eq!(arguments_delta.as_deref(), Some("{\"unit\":\"c\"}"));
            assert_eq!(*index, Some(0));
        }
        other => panic!("Expected ToolCallDelta, got {:?}", other),
    }
}

#[tokio::test]
async fn captures_thinking_signature_delta_and_exposes_in_stream_end() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let thinking_start = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let _ = converter.convert_event(thinking_start).await;

    let sig_delta = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let evs = converter.convert_event(sig_delta).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:thinking-signature-delta");
            assert_eq!(data["signatureDelta"], serde_json::json!("sig-1"));
        }
        other => panic!("Expected signature delta Custom event, got {:?}", other),
    }

    let stop = Event {
        event: "".to_string(),
        data: r#"{"type":"message_stop"}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let evs = converter.convert_event(stop).await;
    let end = evs
        .into_iter()
        .find(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
        .expect("stream end");
    match end.unwrap() {
        ChatStreamEvent::StreamEnd { response } => {
            let meta = response.provider_metadata.expect("provider_metadata");
            assert_eq!(
                meta.get("anthropic")
                    .unwrap()
                    .get("thinking_signature")
                    .unwrap(),
                &serde_json::json!("sig-1")
            );
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn captures_redacted_thinking_data_in_stream_end() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let redacted_start = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"abc123"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let evs = converter.convert_event(redacted_start).await;
    assert_eq!(evs.len(), 1);
    match evs.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:reasoning-start");
            assert_eq!(data["redactedData"], serde_json::json!("abc123"));
        }
        other => panic!("Expected reasoning-start Custom event, got {:?}", other),
    }

    let stop = Event {
        event: "".to_string(),
        data: r#"{"type":"message_stop"}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let evs = converter.convert_event(stop).await;
    let end = evs
        .into_iter()
        .find(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
        .expect("stream end");
    match end.unwrap() {
        ChatStreamEvent::StreamEnd { response } => {
            let meta = response.provider_metadata.expect("provider_metadata");
            assert_eq!(
                meta.get("anthropic")
                    .unwrap()
                    .get("redacted_thinking_data")
                    .unwrap(),
                &serde_json::json!("abc123")
            );
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn emits_source_event_for_citations_delta_with_document_location() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config).with_citation_documents(vec![
        AnthropicCitationDocument {
            title: "Doc A".to_string(),
            filename: Some("a.pdf".to_string()),
            media_type: "application/pdf".to_string(),
        },
    ]);

    let ev = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"page_location","cited_text":"hello","document_index":0,"document_title":null,"start_page_number":1,"end_page_number":1}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let out = converter.convert_event(ev).await;
    assert_eq!(out.len(), 1);
    match out.first().unwrap().as_ref().unwrap() {
        ChatStreamEvent::Custom { event_type, data } => {
            assert_eq!(event_type, "anthropic:source");
            assert_eq!(data["sourceType"], serde_json::json!("document"));
            assert_eq!(data["mediaType"], serde_json::json!("application/pdf"));
            assert_eq!(data["title"], serde_json::json!("Doc A"));
            assert_eq!(data["filename"], serde_json::json!("a.pdf"));
            assert_eq!(
                data["providerMetadata"]["anthropic"]["startPageNumber"],
                serde_json::json!(1)
            );
        }
        other => panic!("Expected source Custom event, got {:?}", other),
    }
}

#[tokio::test]
async fn accumulates_sources_into_stream_end_provider_metadata() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config).with_citation_documents(vec![
        AnthropicCitationDocument {
            title: "Doc A".to_string(),
            filename: Some("a.pdf".to_string()),
            media_type: "application/pdf".to_string(),
        },
    ]);

    let ev = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"page_location","cited_text":"hello","document_index":0,"document_title":null,"start_page_number":1,"end_page_number":1}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let _ = converter.convert_event(ev).await;

    let stop = Event {
        event: "".to_string(),
        data: r#"{"type":"message_stop"}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out = converter.convert_event(stop).await;
    let end = out
        .into_iter()
        .find(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
        .expect("stream end");
    match end.unwrap() {
        ChatStreamEvent::StreamEnd { response } => {
            let meta = response.provider_metadata.expect("provider_metadata");
            let anthropic = meta.get("anthropic").expect("anthropic");
            let sources = anthropic
                .get("sources")
                .and_then(|v| v.as_array())
                .expect("sources array");
            assert_eq!(sources.len(), 1);
            assert_eq!(sources[0]["source_type"], serde_json::json!("document"));
            assert_eq!(
                sources[0]["media_type"],
                serde_json::json!("application/pdf")
            );
            assert_eq!(sources[0]["filename"], serde_json::json!("a.pdf"));
        }
        _ => unreachable!(),
    }
}

fn parse_sse_json_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|chunk| {
            let line = chunk
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .map(str::trim)?;
            if line.is_empty() {
                return None;
            }
            serde_json::from_str::<serde_json::Value>(line).ok()
        })
        .collect()
}

#[derive(Debug)]
struct SseFrame {
    event: Option<String>,
    data: serde_json::Value,
}

fn parse_sse_frames(bytes: &[u8]) -> Vec<SseFrame> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|chunk| {
            let mut event: Option<String> = None;
            let mut data_line: Option<&str> = None;

            for line in chunk.lines() {
                if let Some(v) = line.strip_prefix("event: ") {
                    event = Some(v.trim().to_string());
                    continue;
                }
                if let Some(v) = line.strip_prefix("data: ") {
                    data_line = Some(v.trim());
                    continue;
                }
            }

            let data_str = data_line?;
            if data_str.is_empty() {
                return None;
            }
            let data = serde_json::from_str::<serde_json::Value>(data_str).ok()?;
            Some(SseFrame { event, data })
        })
        .collect()
}

#[test]
fn serializes_text_stream_events_to_anthropic_sse() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let start = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
            },
        })
        .expect("serialize start");
    let start_frames = parse_sse_json_frames(&start);
    assert_eq!(start_frames.len(), 1);
    assert_eq!(start_frames[0]["type"], serde_json::json!("message_start"));

    let delta = converter
        .serialize_event(&ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        })
        .expect("serialize delta");
    let delta_frames = parse_sse_json_frames(&delta);
    assert!(
        delta_frames
            .iter()
            .any(|v| v["type"] == "content_block_start")
    );
    assert!(
        delta_frames
            .iter()
            .any(|v| v["type"] == "content_block_delta")
    );

    let end = converter
        .serialize_event(&ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                content: MessageContent::Text(String::new()),
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
    let end_frames = parse_sse_json_frames(&end);
    assert!(end_frames.iter().any(|v| v["type"] == "content_block_stop"));
    assert!(end_frames.iter().any(|v| v["type"] == "message_delta"));
    assert!(end_frames.iter().any(|v| v["type"] == "message_stop"));
}

#[test]
fn serializes_error_event_with_event_prefix() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Error {
            error: "Overloaded".to_string(),
        })
        .expect("serialize error");

    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.starts_with("event: error\n"),
        "expected `event: error` prefix, got: {text:?}"
    );

    let frames = parse_sse_json_frames(&bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0]["type"], serde_json::json!("error"));
    assert_eq!(frames[0]["error"]["type"], serde_json::json!("api_error"));
}

#[test]
fn serializes_blocks_in_order_and_closes_before_message_stop() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(
        &converter
            .serialize_event(&ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("msg_test".to_string()),
                    model: Some("claude-test".to_string()),
                    created: None,
                    provider: "anthropic".to_string(),
                    request_id: None,
                },
            })
            .expect("serialize start"),
    );
    bytes.extend_from_slice(
        &converter
            .serialize_event(&ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            })
            .expect("serialize text delta"),
    );
    bytes.extend_from_slice(
        &converter
            .serialize_event(&ChatStreamEvent::ThinkingDelta {
                delta: "Thinking".to_string(),
            })
            .expect("serialize thinking delta"),
    );
    bytes.extend_from_slice(
        &converter
            .serialize_event(&ChatStreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                function_name: Some("get_weather".to_string()),
                arguments_delta: Some("{\"city\":".to_string()),
                index: None,
            })
            .expect("serialize tool call delta"),
    );
    bytes.extend_from_slice(
        &converter
            .serialize_event(&ChatStreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                function_name: None,
                arguments_delta: Some("\"Tokyo\"}".to_string()),
                index: None,
            })
            .expect("serialize tool call delta 2"),
    );
    bytes.extend_from_slice(
        &converter
            .serialize_event(&ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("msg_test".to_string()),
                    model: Some("claude-test".to_string()),
                    content: MessageContent::Text(String::new()),
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
            .expect("serialize end"),
    );

    let frames = parse_sse_frames(&bytes);
    assert!(!frames.is_empty(), "expected frames");

    let types: Vec<String> = frames
        .iter()
        .filter_map(|f| {
            f.data
                .get("type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect();

    assert_eq!(
        types.first().map(String::as_str),
        Some("message_start"),
        "expected message_start first, got: {types:?}"
    );
    assert_eq!(
        types.last().map(String::as_str),
        Some("message_stop"),
        "expected message_stop last, got: {types:?}"
    );

    let message_delta_pos = types
        .iter()
        .position(|t| t == "message_delta")
        .expect("message_delta present");
    let message_stop_pos = types
        .iter()
        .position(|t| t == "message_stop")
        .expect("message_stop present");
    assert_eq!(
        message_stop_pos,
        types.len() - 1,
        "message_stop must be the last frame: {types:?}"
    );

    let mut starts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut stops: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut deltas: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();

    for (pos, f) in frames.iter().enumerate() {
        let Some(t) = f.data.get("type").and_then(|v| v.as_str()) else {
            continue;
        };
        match t {
            "content_block_start" => {
                let idx = f
                    .data
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .expect("content_block_start index") as usize;
                starts.insert(idx, pos);
            }
            "content_block_delta" => {
                let idx = f
                    .data
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .expect("content_block_delta index") as usize;
                deltas.entry(idx).or_default().push(pos);
            }
            "content_block_stop" => {
                let idx = f
                    .data
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .expect("content_block_stop index") as usize;
                stops.insert(idx, pos);
            }
            _ => {}
        }
    }

    assert!(
        !starts.is_empty(),
        "expected at least one content_block_start"
    );

    for (idx, start_pos) in &starts {
        let stop_pos = stops
            .get(idx)
            .copied()
            .expect("content_block_stop for started block");
        assert!(
            stop_pos < message_delta_pos,
            "content_block_stop must appear before message_delta (idx={idx}): {types:?}"
        );
        assert!(
            start_pos < &stop_pos,
            "content_block_start must appear before stop (idx={idx}): {types:?}"
        );

        let ds = deltas.get(idx).cloned().unwrap_or_default();
        assert!(
            !ds.is_empty(),
            "expected at least one delta for started block idx={idx}"
        );
        for dpos in ds {
            assert!(
                dpos > *start_pos && dpos < stop_pos,
                "delta must be between start and stop (idx={idx}): {types:?}"
            );
        }
    }

    let tool_start = frames.iter().find(|f| {
        f.data.get("type").and_then(|v| v.as_str()) == Some("content_block_start")
            && f.data
                .get("content_block")
                .and_then(|v| v.get("type"))
                .and_then(|v| v.as_str())
                == Some("tool_use")
    });
    let tool_start = tool_start.expect("tool_use content_block_start");
    assert_eq!(
        tool_start
            .data
            .get("content_block")
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str()),
        Some("call_1")
    );
    assert_eq!(
        tool_start
            .data
            .get("content_block")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str()),
        Some("get_weather")
    );

    assert!(
        frames
            .iter()
            .filter(|f| f.data.get("type").and_then(|v| v.as_str()) != Some("error"))
            .all(|f| f.event.is_none()),
        "only error frames should have an SSE event name"
    );
}

#[test]
fn serializes_v3_custom_parts_to_anthropic_sse() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
            },
        })
        .expect("serialize start");

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": "0",
                "delta": "Hello",
            }),
        })
        .expect("serialize custom text-delta");
    let frames = parse_sse_json_frames(&bytes);
    assert!(
        frames.iter().any(|v| v["type"] == "content_block_delta"),
        "expected content_block_delta from custom text-delta: {frames:?}"
    );

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "call_1",
                "toolName": "get_weather",
                "input": r#"{"city":"Tokyo"}"#,
            }),
        })
        .expect("serialize custom tool-call");
    let frames = parse_sse_json_frames(&bytes);
    assert!(
        frames.iter().any(|v| v["type"] == "content_block_start"
            && v["content_block"]["type"] == "tool_use"
            && v["content_block"]["id"] == "call_1"
            && v["content_block"]["name"] == "get_weather"),
        "expected tool_use content_block_start from custom tool-call: {frames:?}"
    );
    assert!(
        frames
            .iter()
            .any(|v| v["type"] == "content_block_delta"
                && v["delta"]["type"] == "input_json_delta"),
        "expected input_json_delta from custom tool-call: {frames:?}"
    );
}

#[test]
fn serializes_v3_finish_part_as_message_stop_sequence() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "gemini:reasoning".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "usage": {
                    "inputTokens": { "total": 3 },
                    "outputTokens": { "total": 5 }
                },
                "finishReason": { "unified": "stop" }
            }),
        })
        .expect("serialize v3 finish");

    let frames = parse_sse_json_frames(&bytes);
    assert!(
        frames.iter().any(|v| v["type"] == "message_start"),
        "expected message_start: {frames:?}"
    );
    assert!(
        frames.iter().any(|v| v["type"] == "message_delta"
            && v["delta"]["stop_reason"] == serde_json::json!("end_turn")),
        "expected message_delta stop_reason end_turn: {frames:?}"
    );
    assert!(
        frames.iter().any(|v| v["type"] == "message_stop"),
        "expected message_stop: {frames:?}"
    );
}

#[test]
fn serializes_v3_tool_result_as_text_when_configured() {
    let converter = AnthropicEventConverter::new(create_test_config())
        .with_v3_unsupported_part_behavior(crate::streaming::V3UnsupportedPartBehavior::AsText);

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
            },
        })
        .expect("serialize start");

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "call_1",
                "toolName": "web_search",
                "result": [{ "type": "web_search_result", "url": "https://example.com" }]
            }),
        })
        .expect("serialize v3 tool-result");

    let frames = parse_sse_json_frames(&bytes);
    assert!(
        frames.iter().any(|v| v["type"] == "content_block_delta"
            && v["delta"]["type"] == "text_delta"
            && v["delta"]["text"]
                .as_str()
                .is_some_and(|s| s.contains("[tool-result]"))),
        "expected text_delta containing [tool-result]: {frames:?}"
    );
}
