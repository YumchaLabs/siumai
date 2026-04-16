use super::adapter::{OpenAiStandardAdapter, ProviderAdapter};
use super::openai_config::OpenAiCompatibleConfig;
use super::provider_registry::{ConfigurableAdapter, ProviderConfig};
use super::streaming::OpenAiCompatibleEventConverter;
use super::transformers::CompatResponseTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::streaming::StreamProcessor;
use crate::streaming::{ChatStreamEvent, SseEventConverter, SseStreamExt};
use crate::types::ChatStreamPart;
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
        provider_metadata_key: None,
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
        provider_metadata_key: None,
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
        provider_metadata_key: None,
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
async fn parser_emits_stable_tool_parts_before_legacy_shadow_deltas() {
    let conv = make_converter();

    let start_chunk = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":""}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let start_events: Vec<ChatStreamEvent> = conv
        .convert_event(start_chunk)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();
    let stable_start_pos = start_events
        .iter()
        .position(|event| {
            matches!(
                event,
                ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::ToolInputStart { id, tool_name, .. }
                } if id == "call_1" && tool_name == "lookup"
            )
        })
        .expect("stable tool-input-start");
    let legacy_start_pos = start_events
        .iter()
        .position(|event| {
            matches!(
                event,
                ChatStreamEvent::ToolCallDelta {
                    id,
                    function_name,
                    arguments_delta,
                    ..
                } if id == "call_1"
                    && function_name.as_deref() == Some("lookup")
                    && arguments_delta.as_deref() == Some("")
            )
        })
        .expect("legacy shadow tool-call delta");
    assert!(
        stable_start_pos < legacy_start_pos,
        "stable part should be emitted before legacy shadow delta"
    );

    let delta_chunk = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"rust\"}"}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let delta_events: Vec<ChatStreamEvent> = conv
        .convert_event(delta_chunk)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();
    assert!(delta_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolInputDelta { id, delta, .. }
        } if id == "call_1" && delta == "{\"q\":\"rust\"}"
    )));
    assert!(delta_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolInputEnd { id, .. }
        } if id == "call_1"
    )));
    assert!(delta_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolCall(call)
        } if call.tool_call_id == "call_1"
            && call.tool_name == "lookup"
            && call.input == "{\"q\":\"rust\"}"
    )));
    assert!(delta_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::ToolCallDelta {
            id,
            arguments_delta,
            ..
        } if id == "call_1" && arguments_delta.as_deref() == Some("{\"q\":\"rust\"}")
    )));
}

#[tokio::test]
async fn parser_emits_annotations_as_stable_source_parts() {
    let conv = make_converter();

    let event = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl_1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com/rust","title":"Rust"}},{"type":"url_citation","url_citation":{"url":"https://example.com/book"}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let converted: Vec<ChatStreamEvent> = conv
        .convert_event(event)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();

    let source_parts: Vec<(String, String, Option<String>)> = converted
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Part {
                part:
                    crate::types::ChatStreamPart::Source {
                        id,
                        source: crate::types::SourcePart::Url { url, title },
                        ..
                    },
            } => Some((id.clone(), url.clone(), title.clone())),
            _ => None,
        })
        .collect();

    assert_eq!(source_parts.len(), 2);
    assert_eq!(
        source_parts[0],
        (
            "source_chatcmpl_1_0".to_string(),
            "https://example.com/rust".to_string(),
            Some("Rust".to_string())
        )
    );
    assert_eq!(
        source_parts[1],
        (
            "source_chatcmpl_1_1".to_string(),
            "https://example.com/book".to_string(),
            None
        )
    );
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
        Ok(ChatStreamEvent::UsageUpdate { usage }) if usage.prompt_tokens() == Some(5) && usage.completion_tokens() == Some(7) && usage.total_tokens() == Some(12)
    )));

    // 5) End of stream ([DONE]) -> StreamEnd
    let end = converter.handle_stream_end().expect("end event");
    assert!(matches!(end, Ok(ChatStreamEvent::StreamEnd { .. })));
}

#[tokio::test]
async fn parser_emits_stream_start_and_response_metadata_parts_on_first_chunk() {
    let converter = make_converter();

    let event = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","created":1731234567,"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let converted: Vec<ChatStreamEvent> = converter
        .convert_event(event)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();

    assert!(matches!(
        converted.first(),
        Some(ChatStreamEvent::StreamStart { .. })
    ));
    assert!(matches!(
        converted.get(1),
        Some(ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::StreamStart { warnings }
        }) if warnings.is_empty()
    ));
    assert!(matches!(
        converted.get(2),
        Some(ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ResponseMetadata(metadata)
        }) if metadata.id.as_deref() == Some("chatcmpl-1")
            && metadata.model.as_deref() == Some("gpt-4o-mini")
            && metadata.created.is_some()
    ));
    assert!(converted.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextStart { .. }
        }
    )));
    assert!(converted.iter().any(|event| matches!(
        event,
        ChatStreamEvent::ContentDelta { delta, .. } if delta == "Hello"
    )));
}

#[tokio::test]
async fn parser_defers_response_metadata_until_model_router_chunk_has_real_metadata() {
    let converter = make_converter();

    let prompt_filter_chunk = Event {
        event: "".to_string(),
        data: r#"{"choices":[],"created":0,"id":"","model":"","object":"","prompt_filter_results":[{"prompt_index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"}}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let first_events: Vec<ChatStreamEvent> = converter
        .convert_event(prompt_filter_chunk)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();

    assert!(matches!(
        first_events.first(),
        Some(ChatStreamEvent::StreamStart { metadata })
            if metadata.id.is_none()
                && metadata.model.as_deref() == Some("gpt-4o-mini")
                && metadata.created.is_none()
    ));
    assert!(
        !first_events.iter().any(|event| matches!(
            event,
            ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ResponseMetadata(_)
            }
        )),
        "prompt-filter prelude must not emit response metadata"
    );

    let actual_chunk = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"content_filter_results":{},"delta":{"content":"","refusal":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1762317021,"id":"chatcmpl-CYPS1lijGoK8gd9lYzY3r9Sx50nbt","model":"gpt-5-nano-2025-08-07","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let second_events: Vec<ChatStreamEvent> = converter
        .convert_event(actual_chunk)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();

    assert!(matches!(
        second_events
            .iter()
            .find(|event| matches!(
                event,
                ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::ResponseMetadata(_)
                }
            )),
        Some(ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ResponseMetadata(metadata)
        }) if metadata.id.as_deref() == Some("chatcmpl-CYPS1lijGoK8gd9lYzY3r9Sx50nbt")
            && metadata.model.as_deref() == Some("gpt-5-nano-2025-08-07")
            && metadata
                .created
                .as_ref()
                .map(|created| created.timestamp())
                == Some(1_762_317_021)
    ));

    let later_chunk = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"content_filter_results":{"hate":{"filtered":false,"severity":"safe"}},"delta":{"content":"Capital"},"finish_reason":null,"index":0,"logprobs":null}],"created":1762317021,"id":"chatcmpl-CYPS1lijGoK8gd9lYzY3r9Sx50nbt","model":"gpt-5-nano-2025-08-07","object":"chat.completion.chunk","system_fingerprint":null,"usage":null}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let third_events: Vec<ChatStreamEvent> = converter
        .convert_event(later_chunk)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();

    assert!(
        !third_events.iter().any(|event| matches!(
            event,
            ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ResponseMetadata(_)
            }
        )),
        "response metadata should only be emitted once after real metadata appears"
    );
}

#[tokio::test]
async fn parser_emits_text_reasoning_lifecycle_parts_without_duplicate_deltas() {
    let converter = make_converter();

    let first = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let first_events: Vec<ChatStreamEvent> = converter
        .convert_event(first)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();
    let text_part_id = first_events
        .iter()
        .find_map(|event| match event {
            ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::TextStart { id, .. },
            } => Some(id.clone()),
            _ => None,
        })
        .expect("text start id");
    assert_eq!(
        first_events
            .iter()
            .filter(|event| matches!(event, ChatStreamEvent::ContentDelta { .. }))
            .count(),
        1,
        "first chunk should keep a single legacy content delta"
    );

    let second = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"thinking":"Reasoning..."}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let second_events: Vec<ChatStreamEvent> = converter
        .convert_event(second)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();
    assert!(second_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextEnd { id, .. }
        } if id == &text_part_id
    )));
    let reasoning_part_id = second_events
        .iter()
        .find_map(|event| match event {
            ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ReasoningStart { id, .. },
            } => Some(id.clone()),
            _ => None,
        })
        .expect("reasoning start id");
    assert_eq!(
        second_events
            .iter()
            .filter(|event| matches!(event, ChatStreamEvent::ThinkingDelta { .. }))
            .count(),
        1,
        "reasoning chunk should keep a single legacy thinking delta"
    );

    let third = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"content":" world"}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let third_events: Vec<ChatStreamEvent> = converter
        .convert_event(third)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();
    assert!(third_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ReasoningEnd { id, .. }
        } if id == &reasoning_part_id
    )));
    let resumed_text_part_id = third_events
        .iter()
        .find_map(|event| match event {
            ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::TextStart { id, .. },
            } => Some(id.clone()),
            _ => None,
        })
        .expect("resumed text start id");
    assert_ne!(resumed_text_part_id, text_part_id);

    let final_chunk = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let final_events: Vec<ChatStreamEvent> = converter
        .convert_event(final_chunk)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();
    assert!(final_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::TextEnd { id, .. }
        } if id == &resumed_text_part_id
    )));
    assert!(final_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::Finish {
                usage,
                finish_reason,
                ..
            }
        } if usage.total_tokens() == Some(12)
            && finish_reason.unified == crate::types::FinishReason::Stop
            && finish_reason.raw.as_deref() == Some("stop")
    )));
    assert!(final_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::StreamEnd { response }
            if matches!(response.finish_reason, Some(crate::types::FinishReason::Stop))
    )));
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
            |e| matches!(e, ChatStreamEvent::UsageUpdate { usage } if usage.total_tokens() == Some(12))
        ),
        "should contain usage update"
    );
    assert!(
        matches!(events.last(), Some(ChatStreamEvent::StreamEnd { .. })),
        "last should be StreamEnd"
    );
}

#[tokio::test]
async fn compat_stream_finish_keeps_requested_provider_metadata_key_even_without_extra_fields() {
    let base = "https://api.example.com/v1".to_string();
    let adapter: Arc<dyn ProviderAdapter> = Arc::new(ConfigurableAdapter::new(ProviderConfig {
        id: "test-provider".to_string(),
        name: "Test Provider".to_string(),
        base_url: base.clone(),
        field_mappings: Default::default(),
        capabilities: vec!["chat".to_string(), "streaming".to_string()],
        default_model: None,
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: vec![],
    }));
    let cfg = OpenAiCompatibleConfig::new("test-provider", "sk-test", &base, adapter.clone())
        .with_model("test-model");
    let conv = OpenAiCompatibleEventConverter::new(cfg, adapter)
        .with_provider_metadata_key("testProvider");

    let finish_event = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl_1","model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let events = conv.convert_event(finish_event).await;
    let finish = events
        .into_iter()
        .flatten()
        .find_map(|event| match event {
            ChatStreamEvent::Part {
                part:
                    ChatStreamPart::Finish {
                        provider_metadata, ..
                    },
            } => provider_metadata,
            _ => None,
        })
        .expect("finish provider metadata");

    assert!(finish.get("testProvider").is_some());
    assert!(finish.get("test-provider").is_none());
}

#[tokio::test]
async fn compat_stream_tool_call_carries_thought_signature_under_requested_metadata_key() {
    let base = "https://api.example.com/v1".to_string();
    let adapter: Arc<dyn ProviderAdapter> = Arc::new(ConfigurableAdapter::new(ProviderConfig {
        id: "test-provider".to_string(),
        name: "Test Provider".to_string(),
        base_url: base.clone(),
        field_mappings: Default::default(),
        capabilities: vec![
            "chat".to_string(),
            "streaming".to_string(),
            "tools".to_string(),
        ],
        default_model: None,
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: vec![],
    }));
    let cfg = OpenAiCompatibleConfig::new("test-provider", "sk-test", &base, adapter.clone())
        .with_model("test-model");
    let conv = OpenAiCompatibleEventConverter::new(cfg, adapter)
        .with_provider_metadata_key("testProvider");

    let start_event = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl_1","model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"rust\"}"},"extra_content":{"google":{"thought_signature":"<Sig>"}}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let events = conv.convert_event(start_event).await;
    let tool_call_metadata = events
        .into_iter()
        .flatten()
        .find_map(|event| match event {
            ChatStreamEvent::Part {
                part: ChatStreamPart::ToolCall(tool_call),
            } => tool_call.provider_metadata,
            _ => None,
        })
        .expect("tool call provider metadata");

    assert_eq!(
        tool_call_metadata
            .get("testProvider")
            .and_then(|value| value.get("thoughtSignature")),
        Some(&serde_json::json!("<Sig>"))
    );
    assert!(tool_call_metadata.get("test-provider").is_none());
}
