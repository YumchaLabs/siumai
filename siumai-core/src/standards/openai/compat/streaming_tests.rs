use super::adapter::{OpenAiStandardAdapter, ProviderAdapter};
use super::openai_config::OpenAiCompatibleConfig;
use super::provider_registry::{ConfigurableAdapter, ProviderConfig};
use super::streaming::OpenAiCompatibleEventConverter;
use super::transformers::CompatResponseTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::streaming::StreamProcessor;
use crate::streaming::{ChatStreamEvent, SseEventConverter, SseStreamExt};
use eventsource_stream::Event;
use futures_util::StreamExt;
use std::sync::Arc;

fn make_converter() -> OpenAiCompatibleEventConverter {
    let base = "https://api.openai.com/v1".to_string();
    let adapter = Arc::new(OpenAiStandardAdapter {
        base_url: base.clone(),
    });
    let cfg = OpenAiCompatibleConfig::new("openai", "sk-test", &base, adapter.clone())
        .with_model("gpt-4o-mini");
    OpenAiCompatibleEventConverter::new(cfg, adapter)
}

#[tokio::test]
async fn streaming_tool_calls_match_non_streaming_tool_calls() {
    let base = "https://api.openai.com/v1".to_string();
    let adapter = Arc::new(OpenAiStandardAdapter {
        base_url: base.clone(),
    });
    let cfg = OpenAiCompatibleConfig::new("openai", "sk-test", &base, adapter.clone())
        .with_model("gpt-4o-mini");

    let conv = OpenAiCompatibleEventConverter::new(cfg.clone(), adapter.clone());
    let tx = CompatResponseTransformer {
        config: cfg,
        adapter,
    };

    let raw = serde_json::json!({
        "id": "chatcmpl_1",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"q\":\"rust\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    });

    let non_stream = tx
        .transform_chat_response(&raw)
        .expect("non-stream transform");
    assert_eq!(
        non_stream.finish_reason,
        Some(crate::types::FinishReason::ToolCalls)
    );
    assert_eq!(non_stream.tool_calls().len(), 1);

    let mut sp = StreamProcessor::new();
    let mut finish_reason = None;

    let stream_events = vec![
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":""}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\": \""}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"rust\"}"}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    for e in stream_events {
        for ev in conv.convert_event(e).await.into_iter().flatten() {
            match ev {
                ChatStreamEvent::StreamEnd { response } => {
                    finish_reason = response.finish_reason;
                }
                other => {
                    let _ = sp.process_event(other);
                }
            }
        }
    }

    let streaming = sp.build_final_response_with_finish_reason(finish_reason);
    assert_eq!(
        streaming.finish_reason,
        Some(crate::types::FinishReason::ToolCalls)
    );
    assert_eq!(streaming.tool_calls().len(), 1);

    let a = non_stream.tool_calls()[0]
        .as_tool_call()
        .expect("tool call info");
    let b = streaming.tool_calls()[0]
        .as_tool_call()
        .expect("tool call info");
    assert_eq!(b.tool_call_id, a.tool_call_id);
    assert_eq!(b.tool_name, a.tool_name);
    assert_eq!(b.arguments, a.arguments);
}

#[tokio::test]
async fn xai_runtime_provider_streaming_tool_calls_match_non_streaming_tool_calls() {
    let base = "https://api.x.ai/v1".to_string();
    let adapter: Arc<dyn ProviderAdapter> = Arc::new(ConfigurableAdapter::new(ProviderConfig {
        id: "xai".to_string(),
        name: "xAI".to_string(),
        base_url: base.clone(),
        field_mappings: Default::default(),
        capabilities: vec!["tools".to_string()],
        default_model: None,
        supports_reasoning: true,
        api_key_env: None,
        api_key_env_aliases: vec![],
    }));
    let cfg = OpenAiCompatibleConfig::new("xai", "sk-test", &base, adapter.clone())
        .with_model("grok-3-mini");

    let conv = OpenAiCompatibleEventConverter::new(cfg.clone(), adapter.clone());
    let tx = CompatResponseTransformer {
        config: cfg,
        adapter,
    };

    let raw = serde_json::json!({
        "id": "chatcmpl_1",
        "model": "grok-3-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Tokyo\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    });

    let non_stream = tx
        .transform_chat_response(&raw)
        .expect("non-stream transform");
    assert_eq!(
        non_stream.finish_reason,
        Some(crate::types::FinishReason::ToolCalls)
    );
    assert_eq!(non_stream.tool_calls().len(), 1);

    let mut sp = StreamProcessor::new();
    let mut finish_reason = None;

    let stream_events = vec![
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\""}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Tokyo\"}"}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    for e in stream_events {
        for ev in conv.convert_event(e).await.into_iter().flatten() {
            match ev {
                ChatStreamEvent::StreamEnd { response } => {
                    finish_reason = response.finish_reason;
                }
                other => {
                    let _ = sp.process_event(other);
                }
            }
        }
    }

    let streaming = sp.build_final_response_with_finish_reason(finish_reason);
    assert_eq!(
        streaming.finish_reason,
        Some(crate::types::FinishReason::ToolCalls)
    );
    assert_eq!(streaming.tool_calls().len(), 1);

    let a = non_stream.tool_calls()[0]
        .as_tool_call()
        .expect("tool call info");
    let b = streaming.tool_calls()[0]
        .as_tool_call()
        .expect("tool call info");
    assert_eq!(b.tool_call_id, a.tool_call_id);
    assert_eq!(b.tool_name, a.tool_name);
    assert_eq!(b.arguments, a.arguments);
}

#[tokio::test]
async fn deepseek_runtime_provider_streaming_tool_calls_match_non_streaming_tool_calls() {
    let base = "https://api.deepseek.com/v1".to_string();
    let adapter: Arc<dyn ProviderAdapter> = Arc::new(ConfigurableAdapter::new(ProviderConfig {
        id: "deepseek".to_string(),
        name: "DeepSeek".to_string(),
        base_url: base.clone(),
        field_mappings: Default::default(),
        capabilities: vec!["tools".to_string()],
        default_model: None,
        supports_reasoning: true,
        api_key_env: None,
        api_key_env_aliases: vec![],
    }));
    let cfg = OpenAiCompatibleConfig::new("deepseek", "sk-test", &base, adapter.clone())
        .with_model("deepseek-chat");

    let conv = OpenAiCompatibleEventConverter::new(cfg.clone(), adapter.clone());
    let tx = CompatResponseTransformer {
        config: cfg,
        adapter,
    };

    let raw = serde_json::json!({
        "id": "chatcmpl_1",
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Tokyo\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    });

    let non_stream = tx
        .transform_chat_response(&raw)
        .expect("non-stream transform");
    assert_eq!(
        non_stream.finish_reason,
        Some(crate::types::FinishReason::ToolCalls)
    );
    assert_eq!(non_stream.tool_calls().len(), 1);

    let mut sp = StreamProcessor::new();
    let mut finish_reason = None;

    let stream_events = vec![
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\""}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Tokyo\"}"}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    for e in stream_events {
        for ev in conv.convert_event(e).await.into_iter().flatten() {
            match ev {
                ChatStreamEvent::StreamEnd { response } => {
                    finish_reason = response.finish_reason;
                }
                other => {
                    let _ = sp.process_event(other);
                }
            }
        }
    }

    let streaming = sp.build_final_response_with_finish_reason(finish_reason);
    assert_eq!(
        streaming.finish_reason,
        Some(crate::types::FinishReason::ToolCalls)
    );
    assert_eq!(streaming.tool_calls().len(), 1);

    let a = non_stream.tool_calls()[0]
        .as_tool_call()
        .expect("tool call info");
    let b = streaming.tool_calls()[0]
        .as_tool_call()
        .expect("tool call info");
    assert_eq!(b.tool_call_id, a.tool_call_id);
    assert_eq!(b.tool_name, a.tool_name);
    assert_eq!(b.arguments, a.arguments);
}

#[tokio::test]
async fn responses_shape_delta_plain_string_yields_content() {
    let conv = make_converter();

    let event = Event {
        event: "message".to_string(),
        data: r#"{"delta":"Hello"}"#.to_string(),
        id: "1".to_string(),
        retry: None,
    };
    let out = conv.convert_event(event).await;
    let content = out
        .into_iter()
        .filter_map(|e| e.ok())
        .find_map(|ev| match ev {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta),
            _ => None,
        })
        .expect("expected ContentDelta");
    assert_eq!(content, "Hello");
}

#[tokio::test]
async fn responses_shape_delta_text_yields_content() {
    let conv = make_converter();

    let event = Event {
        event: "message".to_string(),
        data: r#"{"delta":{"text":"World"}}"#.to_string(),
        id: "2".to_string(),
        retry: None,
    };
    let out = conv.convert_event(event).await;
    let content = out
        .into_iter()
        .filter_map(|e| e.ok())
        .find_map(|ev| match ev {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta),
            _ => None,
        })
        .expect("expected ContentDelta");
    assert_eq!(content, "World");
}

#[tokio::test]
async fn responses_shape_json_string_event_yields_content() {
    let conv = make_converter();

    let event = Event {
        event: "message".to_string(),
        data: r#""Hi""#.to_string(),
        id: "3".to_string(),
        retry: None,
    };
    let out = conv.convert_event(event).await;
    let content = out
        .into_iter()
        .filter_map(|e| e.ok())
        .find_map(|ev| match ev {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta),
            _ => None,
        })
        .expect("expected ContentDelta");
    assert_eq!(content, "Hi");
}

#[tokio::test]
async fn responses_shape_finish_reason_emits_stream_end() {
    let conv = make_converter();

    let event = Event {
        event: "message".to_string(),
        data: r#"{"choices":[{"index":0,"finish_reason":"stop"}]}"#.to_string(),
        id: "4".to_string(),
        retry: None,
    };
    let out = conv.convert_event(event).await;
    assert!(
        out.into_iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. }))),
        "expected StreamEnd emitted on finish_reason"
    );
}

#[tokio::test]
async fn tool_call_deltas_without_id_are_mapped_by_tool_call_index() {
    let conv = make_converter();

    let event1 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":""}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out1 = conv.convert_event(event1).await;
    assert!(
        out1.iter().any(|e| matches!(
            e,
            Ok(ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, .. })
                if id == "call_1" && function_name.as_deref() == Some("lookup") && arguments_delta.as_deref() == Some("")
        )),
        "first chunk should include id + function name"
    );

    let event2 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\": \""}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out2 = conv.convert_event(event2).await;
    assert!(
        out2.iter().any(|e| matches!(
            e,
            Ok(ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, .. })
                if id == "call_1" && function_name.is_none() && arguments_delta.as_deref() == Some("{\"q\": \"")
        )),
        "follow-up chunk should reuse id by tool_call_index"
    );

    let event3 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"rust\"}"}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out3 = conv.convert_event(event3).await;
    assert!(
        out3.iter().any(|e| matches!(
            e,
            Ok(ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, .. })
                if id == "call_1" && function_name.is_none() && arguments_delta.as_deref() == Some("rust\"}")
        )),
        "follow-up chunk should keep stable id"
    );
}

#[tokio::test]
async fn multi_tool_calls_are_mapped_by_index() {
    let conv = make_converter();

    let event1 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"a","arguments":""}},{"index":1,"id":"call_b","function":{"name":"b","arguments":""}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out1 = conv.convert_event(event1).await;
    assert!(out1.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::ToolCallDelta { id, function_name, .. })
            if id == "call_a" && function_name.as_deref() == Some("a")
    )));
    assert!(out1.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::ToolCallDelta { id, function_name, .. })
            if id == "call_b" && function_name.as_deref() == Some("b")
    )));

    let event2 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"x\":1}"}},{"index":0,"function":{"arguments":"{\"y\":2}"}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out2 = conv.convert_event(event2).await;
    assert!(out2.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::ToolCallDelta { id, arguments_delta, .. })
            if id == "call_b" && arguments_delta.as_deref() == Some("{\"x\":1}")
    )));
    assert!(out2.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::ToolCallDelta { id, arguments_delta, .. })
            if id == "call_a" && arguments_delta.as_deref() == Some("{\"y\":2}")
    )));
}

#[tokio::test]
async fn finish_reason_tool_calls_without_tool_calls_array_emits_stream_end() {
    let conv = make_converter();

    let event = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out = conv.convert_event(event).await;
    assert!(
        out.into_iter().any(|e| {
            matches!(e, Ok(ChatStreamEvent::StreamEnd { response }) if matches!(response.finish_reason, Some(crate::types::FinishReason::ToolCalls)))
        }),
        "expected StreamEnd with finish_reason ToolCalls"
    );
}

#[tokio::test]
async fn multi_event_sequence() {
    let converter = make_converter();

    // 1) First chunk with content + metadata -> StreamStart + ContentDelta
    let event1 = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","created": 1731234567,
                  "choices":[{"index":0,"delta":{"content":"Hello"}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r1 = converter.convert_event(event1).await;
    assert!(
        r1.iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::StreamStart { .. })))
    );
    assert!(
        r1.iter().any(
            |e| matches!(e, Ok(ChatStreamEvent::ContentDelta{ delta, .. }) if delta == "Hello")
        )
    );

    // 2) Thinking delta
    let event2 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"thinking":"Reasoning..."}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r2 = converter.convert_event(event2).await;
    assert!(r2.iter().any(
        |e| matches!(e, Ok(ChatStreamEvent::ThinkingDelta{ delta }) if delta == "Reasoning...")
    ));

    // 3) Tool call delta (function)
    let event3 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1",
                      "function":{"name":"lookup","arguments":"{\"q\":\"rust\"}"}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r3 = converter.convert_event(event3).await;
    assert!(r3.iter().any(|e| matches!(e,
        Ok(ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, .. })
        if id == "call_1" && function_name.as_deref() == Some("lookup") && arguments_delta.as_deref() == Some("{\"q\":\"rust\"}")
    )));

    // 4) Usage update
    let event4 = Event {
        event: "".to_string(),
        data: r#"{"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r4 = converter.convert_event(event4).await;
    assert!(r4.iter().any(|e| matches!(e,
        Ok(ChatStreamEvent::UsageUpdate { usage }) if usage.prompt_tokens == 5 && usage.completion_tokens == 7 && usage.total_tokens == 12
    )));

    // 5) End of stream ([DONE]) -> StreamEnd
    let end = converter.handle_stream_end().expect("end event");
    assert!(matches!(end, Ok(ChatStreamEvent::StreamEnd { .. })));
}

#[tokio::test]
async fn finish_reason_without_done_emits_stream_end() {
    let converter = make_converter();

    // Simulate standard OpenAI chat.completions stream without [DONE]
    let sse_chunks = vec![
        // First delta: role only (common in OpenAI streams)
        format!(
            "data: {}\n\n",
            r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","created":1731234567,"choices":[{"index":0,"delta":{"role":"assistant"}}]}"#
        ),
        // Content delta
        format!(
            "data: {}\n\n",
            r#"{"choices":[{"index":0,"delta":{"content":"1\n2\n"}}]}"#
        ),
        // Final chunk with finish_reason but no [DONE]
        format!(
            "data: {}\n\n",
            r#"{"choices":[{"index":0,"finish_reason":"stop"}]}"#
        ),
    ];

    let bytes: Vec<Result<Vec<u8>, std::io::Error>> =
        sse_chunks.into_iter().map(|s| Ok(s.into_bytes())).collect();
    let byte_stream = futures_util::stream::iter(bytes);
    let mut sse_stream = byte_stream.into_sse_stream();

    let mut saw_content = false;
    let mut saw_end = false;
    while let Some(item) = sse_stream.next().await {
        let event: Event = item.expect("valid event");
        let converted = converter.convert_event(event).await;
        for e in converted {
            match e.expect("ok") {
                ChatStreamEvent::ContentDelta { delta, .. } => {
                    assert_eq!(delta, "1\n2\n");
                    saw_content = true;
                }
                ChatStreamEvent::StreamEnd { .. } => saw_end = true,
                _ => {}
            }
        }
    }

    assert!(saw_content, "should see content delta");
    assert!(saw_end, "should emit StreamEnd on finish_reason");
}

#[tokio::test]
async fn end_to_end_sse_multi_event_flow() {
    let converter = make_converter();

    // Build SSE byte stream: multiple data: lines
    let sse_chunks = vec![
        format!(
            "data: {}\n\n",
            r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","created":1731234567,"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#
        ),
        format!(
            "data: {}\n\n",
            r#"{"choices":[{"index":0,"delta":{"thinking":"Reasoning..."}}]}"#
        ),
        format!(
            "data: {}\n\n",
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","function":{"name":"lookup","arguments":"{\"q\":\"rust\"}"}}]}}]}"#
        ),
        format!(
            "data: {}\n\n",
            r#"{"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}"#
        ),
        "data: [DONE]\n\n".to_string(),
    ];

    // Convert into a stream of bytes
    let bytes: Vec<Result<Vec<u8>, std::io::Error>> =
        sse_chunks.into_iter().map(|s| Ok(s.into_bytes())).collect();
    let byte_stream = futures_util::stream::iter(bytes);
    let mut sse_stream = byte_stream.into_sse_stream();

    // Collect ChatStreamEvents in order
    let mut events: Vec<ChatStreamEvent> = Vec::new();
    while let Some(item) = sse_stream.next().await {
        let event: Event = item.expect("valid event");
        if event.data.trim() == "[DONE]" {
            if let Some(end) = converter.handle_stream_end() {
                events.push(end.expect("stream end ok"));
            }
            break;
        }
        let converted = converter.convert_event(event).await;
        for e in converted {
            events.push(e.expect("ok"));
        }
    }

    // Validate sequence has key events
    assert!(
        matches!(events.first(), Some(ChatStreamEvent::StreamStart { .. })),
        "first should be StreamStart"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ChatStreamEvent::ContentDelta { delta, .. } if delta == "Hello")),
        "should contain content delta"
    );
    assert!(
        events.iter().any(
            |e| matches!(e, ChatStreamEvent::ThinkingDelta { delta } if delta == "Reasoning...")
        ),
        "should contain thinking delta"
    );
    assert!(
        events.iter().any(|e| matches!(e, ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, .. }
            if id == "call_1" && function_name.as_deref() == Some("lookup") && arguments_delta.as_deref() == Some("{\"q\":\"rust\"}")
        )),
        "should contain tool call delta"
    );
    assert!(
        events.iter().any(
            |e| matches!(e, ChatStreamEvent::UsageUpdate { usage } if usage.total_tokens == 12)
        ),
        "should contain usage update"
    );
    assert!(
        matches!(events.last(), Some(ChatStreamEvent::StreamEnd { .. })),
        "last should be StreamEnd"
    );
}
