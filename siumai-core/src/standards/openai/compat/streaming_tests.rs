use super::adapter::{MetadataExtractingAdapter, OpenAiStandardAdapter, ProviderAdapter};
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

async fn convert_ok(
    converter: &OpenAiCompatibleEventConverter,
    event: Event,
) -> Vec<ChatStreamEvent> {
    converter
        .convert_event(event)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect()
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
        .find_map(|ev| ev.text_delta().map(ToString::to_string))
        .expect("expected typed text delta");
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
        .find_map(|ev| ev.text_delta().map(ToString::to_string))
        .expect("expected typed text delta");
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
        .find_map(|ev| ev.text_delta().map(ToString::to_string))
        .expect("expected typed text delta");
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
            Ok(ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ToolInputStart { id, tool_name, .. }
            }) if id == "call_1" && tool_name == "lookup"
        )),
        "first chunk should include stable tool-input-start"
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
            Ok(ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ToolInputDelta { id, delta, .. }
            }) if id == "call_1" && delta == "{\"q\": \""
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
            Ok(ChatStreamEvent::Part {
                part: crate::types::ChatStreamPart::ToolInputEnd { id, .. }
            }) if id == "call_1"
        )),
        "follow-up chunk should close the typed tool input"
    );
    assert!(out3.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolCall(call)
        }) if call.tool_call_id == "call_1"
            && call.tool_name == "lookup"
            && call.input == "{\"q\": \"rust\"}"
    )));
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
        Ok(ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolInputStart { id, tool_name, .. }
        }) if id == "call_a" && tool_name == "a"
    )));
    assert!(out1.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolInputStart { id, tool_name, .. }
        }) if id == "call_b" && tool_name == "b"
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
        Ok(ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolInputDelta { id, delta, .. }
        }) if id == "call_b" && delta == "{\"x\":1}"
    )));
    assert!(out2.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolInputDelta { id, delta, .. }
        }) if id == "call_a" && delta == "{\"y\":2}"
    )));
}

#[tokio::test]
async fn parser_emits_stable_tool_parts() {
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
    assert!(start_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolInputStart { id, tool_name, .. }
        } if id == "call_1" && tool_name == "lookup"
    )));

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

    // 1) First chunk with content + metadata -> StreamStart + typed text delta
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
        r1.iter()
            .any(|e| matches!(e, Ok(event) if event.text_delta() == Some("Hello")))
    );

    // 2) Thinking delta
    let event2 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"thinking":"Reasoning..."}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r2 = converter.convert_event(event2).await;
    assert!(
        r2.iter()
            .any(|e| matches!(e, Ok(event) if event.reasoning_delta() == Some("Reasoning...")))
    );

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
        Ok(ChatStreamEvent::Part {
            part: crate::types::ChatStreamPart::ToolCall(call)
        }) if call.tool_call_id == "call_1"
            && call.tool_name == "lookup"
            && call.input == "{\"q\":\"rust\"}"
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
    assert!(
        r4.is_empty(),
        "usage-only chunks should update terminal state without emitting a legacy usage event"
    );

    // 5) End of stream ([DONE]) -> StreamEnd
    let end = converter.handle_stream_end().expect("end event");
    assert!(matches!(
        end,
        Ok(ChatStreamEvent::StreamEnd { response })
            if response
                .usage
                .as_ref()
                .and_then(|usage| usage.total_tokens())
                == Some(12)
    ));
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
    assert!(
        converted
            .iter()
            .any(|event| event.text_delta() == Some("Hello"))
    );
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
            .filter(|event| event.text_delta().is_some())
            .count(),
        1,
        "first chunk should keep a single typed text delta"
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
            .filter(|event| event.reasoning_delta().is_some())
            .count(),
        1,
        "reasoning chunk should keep a single typed reasoning delta"
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
async fn compat_stream_same_chunk_reasoning_precedes_text_parts() {
    let converter = make_converter();

    let events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"reasoning_content":"Think first","content":"Answer second"}}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;

    let reasoning_start_pos = events
        .iter()
        .position(|event| {
            matches!(
                event,
                ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::ReasoningStart { .. }
                }
            )
        })
        .expect("reasoning start");
    let text_start_pos = events
        .iter()
        .position(|event| {
            matches!(
                event,
                ChatStreamEvent::Part {
                    part: crate::types::ChatStreamPart::TextStart { .. }
                }
            )
        })
        .expect("text start");
    let reasoning_delta_pos = events
        .iter()
        .position(|event| event.reasoning_delta() == Some("Think first"))
        .expect("reasoning delta");
    let text_delta_pos = events
        .iter()
        .position(|event| event.text_delta() == Some("Answer second"))
        .expect("text delta");

    assert!(
        reasoning_start_pos < text_start_pos,
        "reasoning lane should open before text when both arrive in one chunk"
    );
    assert!(
        reasoning_delta_pos < text_delta_pos,
        "reasoning delta should be emitted before text delta"
    );
}

#[tokio::test]
async fn compat_stream_reasoning_field_is_used_when_reasoning_content_is_missing() {
    let converter = make_converter();

    let events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"reasoning":"Fallback reasoning"}}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;

    assert!(
        events
            .iter()
            .any(|event| event.reasoning_delta() == Some("Fallback reasoning"))
    );
}

#[tokio::test]
async fn compat_stream_reasoning_content_takes_priority_over_reasoning_field() {
    let converter = make_converter();

    let events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"reasoning_content":"Preferred reasoning","reasoning":"Ignored reasoning"}}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;

    assert!(
        events
            .iter()
            .any(|event| event.reasoning_delta() == Some("Preferred reasoning"))
    );
    assert!(
        !events
            .iter()
            .any(|event| event.reasoning_delta() == Some("Ignored reasoning"))
    );
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
            let event = e.expect("ok");
            if event.text_delta() == Some("1\n2\n") {
                saw_content = true;
            }
            if matches!(event, ChatStreamEvent::StreamEnd { .. }) {
                saw_end = true;
            }
        }
    }

    assert!(saw_content, "should see typed text delta");
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
        events.iter().any(|e| e.text_delta() == Some("Hello")),
        "should contain typed text delta"
    );
    assert!(
        events
            .iter()
            .any(|e| e.reasoning_delta() == Some("Reasoning...")),
        "should contain typed reasoning delta"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            ChatStreamEvent::Part {
                part: ChatStreamPart::ToolCall(call)
            } if call.tool_call_id == "call_1"
                && call.tool_name == "lookup"
                && call.input == "{\"q\":\"rust\"}"
        )),
        "should contain typed tool call"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            ChatStreamEvent::StreamEnd { response }
                if response
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.total_tokens())
                    == Some(12)
        )),
        "should carry usage on StreamEnd"
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

    assert!(finish.contains_key("testProvider"));
    assert!(!finish.contains_key("test-provider"));
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
    assert!(!tool_call_metadata.contains_key("test-provider"));
}

#[tokio::test]
async fn compat_stream_tool_call_sent_in_one_chunk_emits_single_complete_lifecycle() {
    let converter = make_converter();

    let start_events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"rust\"}"}}]}}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;

    assert_eq!(
        start_events
            .iter()
            .filter(|event| matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolInputStart { id, .. }
                } if id == "call_1"
            ))
            .count(),
        1
    );
    assert_eq!(
        start_events
            .iter()
            .filter(|event| matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolInputEnd { id, .. }
                } if id == "call_1"
            ))
            .count(),
        1
    );
    assert_eq!(
        start_events
            .iter()
            .filter(|event| matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolCall(call)
                } if call.tool_call_id == "call_1"
                    && call.tool_name == "lookup"
                    && call.input == "{\"q\":\"rust\"}"
            ))
            .count(),
        1
    );

    let finish_events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;

    assert!(!finish_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: ChatStreamPart::ToolCall(call)
        } if call.tool_call_id == "call_1"
    )));
    assert!(finish_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: ChatStreamPart::Finish { finish_reason, .. }
        } if finish_reason.unified == crate::types::FinishReason::ToolCalls
    )));
}

#[tokio::test]
async fn compat_stream_finish_reason_tool_calls_finalizes_empty_tool_call_once() {
    let converter = make_converter();

    let start_events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_empty","type":"function","function":{"name":"lookup","arguments":""}}]}}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;
    assert!(start_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputStart { id, tool_name, .. }
        } if id == "call_empty" && tool_name == "lookup"
    )));

    let finish_events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;

    assert_eq!(
        finish_events
            .iter()
            .filter(|event| matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolInputEnd { id, .. }
                } if id == "call_empty"
            ))
            .count(),
        1,
        "finish_reason tool_calls should close the pending tool input exactly once"
    );
    assert_eq!(
        finish_events
            .iter()
            .filter(|event| matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolCall(call)
                } if call.tool_call_id == "call_empty"
                    && call.tool_name == "lookup"
                    && call.input.is_empty()
            ))
            .count(),
        1,
        "finish_reason tool_calls should emit the finalized empty tool call"
    );
    assert!(finish_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: ChatStreamPart::Finish { finish_reason, .. }
        } if finish_reason.unified == crate::types::FinishReason::ToolCalls
    )));
}

#[tokio::test]
async fn compat_stream_completed_tool_call_is_not_duplicated_by_later_empty_chunks() {
    let converter = make_converter();

    let _ = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":""}}]}}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;
    let completed_events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"rust\"}"}}]}}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;
    let finish_events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]},"finish_reason":"tool_calls"}]}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;
    let trailing_empty_events = convert_ok(
        &converter,
        Event {
            event: "".to_string(),
            data: r#"{"choices":[]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    )
    .await;

    assert_eq!(
        completed_events
            .iter()
            .filter(|event| matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::ToolCall(call)
                } if call.tool_call_id == "call_1"
            ))
            .count(),
        1
    );
    assert!(!finish_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: ChatStreamPart::ToolCall(call)
        } if call.tool_call_id == "call_1"
    )));
    assert!(!trailing_empty_events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: ChatStreamPart::ToolCall(call)
        } if call.tool_call_id == "call_1"
    )));
}

#[tokio::test]
async fn compat_stream_explicit_error_payload_emits_stable_error_and_error_finish() {
    let conv = make_converter();

    let event = Event {
        event: "".to_string(),
        data: r#"{"error":{"message":"Incorrect API key provided: as***T7. You can obtain an API key from https://console.api.com."}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let events: Vec<ChatStreamEvent> = conv
        .convert_event(event)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();

    assert!(matches!(
        events.first(),
        Some(ChatStreamEvent::StreamStart { .. })
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: ChatStreamPart::Error { error }
        } if error == &serde_json::Value::String(
            "Incorrect API key provided: as***T7. You can obtain an API key from https://console.api.com.".to_string()
        )
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: ChatStreamPart::Finish { finish_reason, .. }
        } if finish_reason.unified == crate::types::FinishReason::Error
            && finish_reason.raw.is_none()
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::StreamEnd { response }
            if response.finish_reason == Some(crate::types::FinishReason::Error)
                && response.raw_finish_reason.is_none()
    )));
}

#[tokio::test]
async fn compat_stream_unparsable_chunk_emits_raw_error_and_error_finish() {
    let conv = make_converter().with_include_raw_chunks(true);

    let event = Event {
        event: "".to_string(),
        data: "{unparsable}".to_string(),
        id: "".to_string(),
        retry: None,
    };

    let events: Vec<ChatStreamEvent> = conv
        .convert_event(event)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .collect();

    let raw_pos = events
        .iter()
        .position(|event| {
            matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::Raw { raw_value }
                } if raw_value == &serde_json::Value::String("{unparsable}".to_string())
            )
        })
        .expect("raw part");
    let error_pos = events
        .iter()
        .position(|event| {
            matches!(
                event,
                ChatStreamEvent::Part {
                    part: ChatStreamPart::Error { error }
                } if error
                    .as_str()
                    .is_some_and(|message| message.contains("Failed to parse OpenAI-compatible event"))
            )
        })
        .expect("error part");

    assert!(
        raw_pos < error_pos,
        "raw part should be emitted before error part"
    );
    assert!(events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::Part {
            part: ChatStreamPart::Finish { finish_reason, .. }
        } if finish_reason.unified == crate::types::FinishReason::Error
            && finish_reason.raw.is_none()
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ChatStreamEvent::StreamEnd { response }
            if response.finish_reason == Some(crate::types::FinishReason::Error)
    )));
}

#[tokio::test]
async fn compat_stream_finish_surfaces_prediction_token_provider_metadata() {
    let conv = make_converter();

    let event = Event {
        event: "".to_string(),
        data: r#"{"id":"chat-id","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":30,"prompt_tokens_details":{"cached_tokens":5},"completion_tokens_details":{"reasoning_tokens":10,"accepted_prediction_tokens":15,"rejected_prediction_tokens":5}}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let finish = conv
        .convert_event(event)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
        .find_map(|event| match event {
            ChatStreamEvent::Part {
                part:
                    ChatStreamPart::Finish {
                        usage,
                        finish_reason,
                        provider_metadata,
                    },
            } => Some((usage, finish_reason, provider_metadata)),
            _ => None,
        })
        .expect("finish part");

    assert_eq!(finish.0.prompt_tokens(), Some(20));
    assert_eq!(finish.0.completion_tokens(), Some(30));
    assert_eq!(finish.1.unified, crate::types::FinishReason::Stop);
    let provider_metadata = finish.2.expect("provider metadata");
    let openai = provider_metadata.get("openai").expect("openai namespace");
    assert_eq!(
        openai.get("acceptedPredictionTokens"),
        Some(&serde_json::json!(15))
    );
    assert_eq!(
        openai.get("rejectedPredictionTokens"),
        Some(&serde_json::json!(5))
    );
}

#[tokio::test]
async fn compat_stream_metadata_extracting_adapter_merges_finish_metadata() {
    let base = "https://api.example.com/v1".to_string();
    let inner = ConfigurableAdapter::new(ProviderConfig {
        id: "test-provider".to_string(),
        name: "Test Provider".to_string(),
        base_url: base.clone(),
        field_mappings: Default::default(),
        capabilities: vec!["chat".to_string(), "streaming".to_string()],
        default_model: None,
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: vec![],
    });
    let adapter: Arc<dyn ProviderAdapter> = Arc::new(MetadataExtractingAdapter::new(
        Box::new(inner),
        Arc::new(|raw: &serde_json::Value| {
            raw.get("test_field").map(|value| {
                std::collections::HashMap::from([(
                    "test-provider".to_string(),
                    serde_json::Value::Object(serde_json::Map::from_iter([(
                        "value".to_string(),
                        value.clone(),
                    )])),
                )])
            })
        }),
    ));
    let cfg = OpenAiCompatibleConfig::new("test-provider", "sk-test", &base, adapter.clone())
        .with_model("test-model");
    let conv = OpenAiCompatibleEventConverter::new(cfg, adapter);

    let event = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl_1","model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"test_field":"test_value"}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let provider_metadata = conv
        .convert_event(event)
        .await
        .into_iter()
        .map(|event| event.expect("event ok"))
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

    assert_eq!(
        provider_metadata
            .get("test-provider")
            .and_then(|value| value.get("value")),
        Some(&serde_json::json!("test_value"))
    );
}
