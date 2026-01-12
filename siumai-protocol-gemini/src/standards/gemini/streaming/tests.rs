use super::*;
use crate::standards::gemini::types::GeminiConfig;
use crate::streaming::SseEventConverter;
use crate::types::{ChatResponse, FinishReason, MessageContent, Usage};

fn create_test_config() -> GeminiConfig {
    use secrecy::SecretString;
    GeminiConfig {
        api_key: SecretString::from("test-key".to_string()),
        base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        ..Default::default()
    }
}

#[tokio::test]
async fn test_gemini_streaming_conversion() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    // Test content delta conversion
    let json_data = r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}"#;
    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: json_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(!result.is_empty());

    // In the new architecture, we might get StreamStart + ContentDelta
    let content_event = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::ContentDelta { .. })));

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = content_event {
        assert_eq!(delta, "Hello");
    } else {
        panic!("Expected ContentDelta event in results: {:?}", result);
    }
}

#[tokio::test]
async fn test_gemini_streaming_emits_source_events_and_dedups() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let json_data = r#"{"candidates":[{"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://www.rust-lang.org/","title":"Rust"}}]}}]}"#;
    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: json_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let r1 = converter.convert_event(event).await;
    assert!(
        r1.iter().any(|e| matches!(e, Ok(ChatStreamEvent::Custom { event_type, .. }) if event_type == "gemini:source")),
        "expected gemini:source in first chunk: {r1:?}"
    );

    // Same payload again should not emit a duplicate source event.
    let event2 = eventsource_stream::Event {
        event: "".to_string(),
        data: json_data.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r2 = converter.convert_event(event2).await;
    assert!(
        !r2.iter().any(|e| matches!(e, Ok(ChatStreamEvent::Custom { event_type, .. }) if event_type == "gemini:source")),
        "expected no duplicate gemini:source in second chunk: {r2:?}"
    );
}

#[tokio::test]
async fn test_gemini_finish_reason() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    // Test finish reason conversion
    let json_data = r#"{"candidates":[{"finishReason":"STOP"}]}"#;
    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: json_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(!result.is_empty());

    // In the new architecture, first event might be StreamStart, look for StreamEnd
    let stream_end_event = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamEnd { .. })));

    if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = stream_end_event {
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    } else {
        panic!("Expected StreamEnd event in results: {:?}", result);
    }
}

#[tokio::test]
async fn test_gemini_finish_reason_tool_calls_when_stop_with_function_call() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let json_data = r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"test-tool","args":{"a":1}}}]},"finishReason":"STOP"}]}"#;
    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: json_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    let stream_end_event = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamEnd { .. })));

    if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = stream_end_event {
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    } else {
        panic!("Expected StreamEnd event in results: {:?}", result);
    }
}

#[tokio::test]
async fn test_gemini_finish_reason_length() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let json_data = r#"{"candidates":[{"finishReason":"MAX_TOKENS"}]}"#;
    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: json_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    let stream_end_event = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamEnd { .. })));

    if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = stream_end_event {
        assert_eq!(response.finish_reason, Some(FinishReason::Length));
    } else {
        panic!("Expected StreamEnd event in results: {:?}", result);
    }
}

#[tokio::test]
async fn test_gemini_finish_reason_content_filter() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let json_data = r#"{"candidates":[{"finishReason":"PROHIBITED_CONTENT"}]}"#;
    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: json_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    let stream_end_event = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamEnd { .. })));

    if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = stream_end_event {
        assert_eq!(response.finish_reason, Some(FinishReason::ContentFilter));
    } else {
        panic!("Expected StreamEnd event in results: {:?}", result);
    }
}

#[tokio::test]
async fn test_gemini_finish_reason_error() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let json_data = r#"{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL"}]}"#;
    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: json_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    let stream_end_event = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamEnd { .. })));

    if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = stream_end_event {
        assert_eq!(response.finish_reason, Some(FinishReason::Error));
    } else {
        panic!("Expected StreamEnd event in results: {:?}", result);
    }
}

#[tokio::test]
async fn test_gemini_finish_reason_other() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let json_data = r#"{"candidates":[{"finishReason":"OTHER"}]}"#;
    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: json_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    let stream_end_event = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamEnd { .. })));

    if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = stream_end_event {
        assert_eq!(
            response.finish_reason,
            Some(FinishReason::Other("OTHER".to_string()))
        );
    } else {
        panic!("Expected StreamEnd event in results: {:?}", result);
    }
}

#[tokio::test]
async fn test_empty_event_is_ignored() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);
    let event = eventsource_stream::Event {
        event: "".into(),
        data: "".into(),
        id: "".into(),
        retry: None,
    };
    let result = converter.convert_event(event).await;
    assert!(result.is_empty(), "Empty SSE event should be ignored");
}

// In strict mode (no json-repair), invalid JSON should produce a parse error
#[cfg(not(feature = "json-repair"))]
#[tokio::test]
async fn test_invalid_json_emits_error() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);
    let event = eventsource_stream::Event {
        event: "".into(),
        data: "{ not json".into(),
        id: "".into(),
        retry: None,
    };
    let result = converter.convert_event(event).await;
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], Err(LlmError::ParseError(_))));
}

// In tolerant mode (json-repair enabled), invalid JSON should not error with ParseError
#[cfg(feature = "json-repair")]
#[tokio::test]
async fn test_invalid_json_is_tolerated_with_repair() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);
    let event = eventsource_stream::Event {
        event: "".into(),
        data: "{ not json".into(),
        id: "".into(),
        retry: None,
    };
    let result = converter.convert_event(event).await;
    assert_eq!(result.len(), 1);
    assert!(!matches!(result[0], Err(LlmError::ParseError(_))));
}

#[tokio::test]
async fn test_stream_start_emitted_once_across_events() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);
    let mk_event = |text: &str| eventsource_stream::Event {
        event: "".into(),
        data: format!(
            "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"{}\"}}]}}}}]}}",
            text
        ),
        id: "".into(),
        retry: None,
    };

    let r1 = converter.convert_event(mk_event("first")).await;
    let r2 = converter.convert_event(mk_event("second")).await;

    // First batch should contain a StreamStart
    assert!(
        r1.iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::StreamStart { .. })))
    );
    // Second batch should not contain StreamStart
    assert!(
        !r2.iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::StreamStart { .. })))
    );
}

#[tokio::test]
async fn test_multi_parts_emit_multiple_deltas_in_order() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);
    let json = r#"{"candidates":[{"content":{"parts":[{"text":"A"},{"text":"B"}]}}]}"#;
    let event = eventsource_stream::Event {
        event: "".into(),
        data: json.into(),
        id: "".into(),
        retry: None,
    };
    let result = converter.convert_event(event).await;
    let deltas: Vec<_> = result
        .into_iter()
        .filter_map(|e| match e {
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) => Some(delta),
            _ => None,
        })
        .collect();
    assert!(deltas.contains(&"A".to_string()));
    assert!(deltas.contains(&"B".to_string()));
    // Order is preserved within a single event
    let a_pos = deltas.iter().position(|d| d == "A").unwrap();
    let b_pos = deltas.iter().position(|d| d == "B").unwrap();
    assert!(a_pos < b_pos);
}

#[tokio::test]
async fn test_thinking_delta_extraction() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);
    let json = r#"{"candidates":[{"content":{"parts":[{"text":"thinking..","thought":true}]}}]}"#;
    let event = eventsource_stream::Event {
        event: "".into(),
        data: json.into(),
        id: "".into(),
        retry: None,
    };
    let result = converter.convert_event(event).await;
    assert!(
        result
            .iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::ThinkingDelta { .. })))
    );
}

fn parse_sse_json_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|chunk| {
            let line = chunk
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .map(str::trim)?;
            if line.is_empty() || line == "[DONE]" {
                return None;
            }
            serde_json::from_str::<serde_json::Value>(line).ok()
        })
        .collect()
}

#[tokio::test]
async fn gemini_stream_proxy_serializes_content_delta() {
    let converter = GeminiEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        })
        .expect("serialize ok");
    let frames = parse_sse_json_frames(&bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["text"],
        serde_json::json!("Hello")
    );

    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::to_string(&frames[0]).expect("json"),
        id: "".to_string(),
        retry: None,
    };
    let out = converter.convert_event(event).await;
    assert!(out.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::ContentDelta { delta, .. }) if delta == "Hello"
    )));
}

#[tokio::test]
async fn gemini_stream_proxy_serializes_thinking_delta() {
    let converter = GeminiEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::ThinkingDelta {
            delta: "think".to_string(),
        })
        .expect("serialize ok");
    let frames = parse_sse_json_frames(&bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["thought"],
        serde_json::json!(true)
    );

    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::to_string(&frames[0]).expect("json"),
        id: "".to_string(),
        retry: None,
    };
    let out = converter.convert_event(event).await;
    assert!(out.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::ThinkingDelta { delta }) if delta == "think"
    )));
}

#[tokio::test]
async fn gemini_stream_proxy_serializes_usage_update() {
    let converter = GeminiEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::UsageUpdate {
            usage: Usage::builder()
                .prompt_tokens(3)
                .completion_tokens(5)
                .total_tokens(8)
                .build(),
        })
        .expect("serialize ok");
    let frames = parse_sse_json_frames(&bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0]["usageMetadata"]["promptTokenCount"],
        serde_json::json!(3)
    );

    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::to_string(&frames[0]).expect("json"),
        id: "".to_string(),
        retry: None,
    };
    let out = converter.convert_event(event).await;
    assert!(out.iter().any(|e| matches!(
        e,
        Ok(ChatStreamEvent::UsageUpdate { usage }) if usage.prompt_tokens == 3
    )));
}

#[tokio::test]
async fn gemini_stream_proxy_serializes_stream_end_finish_reason() {
    let converter = GeminiEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: None,
                model: None,
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
        .expect("serialize ok");
    let frames = parse_sse_json_frames(&bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0]["candidates"][0]["finishReason"],
        serde_json::json!("STOP")
    );

    let event = eventsource_stream::Event {
        event: "".to_string(),
        data: serde_json::to_string(&frames[0]).expect("json"),
        id: "".to_string(),
        retry: None,
    };
    let out = converter.convert_event(event).await;
    assert!(
        out.iter()
            .any(|e| matches!(
                e,
                Ok(ChatStreamEvent::StreamEnd { response }) if response.finish_reason == Some(FinishReason::Stop)
            ))
    );
}

#[test]
fn gemini_serializes_tool_call_delta_as_function_call_part() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let bytes = converter
        .serialize_event(&ChatStreamEvent::ToolCallDelta {
            id: "call_1".to_string(),
            function_name: Some("get_weather".to_string()),
            arguments_delta: Some(r#"{"city":"Tokyo"}"#.to_string()),
            index: None,
        })
        .expect("serialize tool call delta");

    let text = String::from_utf8(bytes).expect("utf8");
    let json_line = text
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("data line");

    let v: serde_json::Value = serde_json::from_str(json_line).expect("json");
    assert_eq!(
        v["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
        serde_json::json!("get_weather")
    );
    assert_eq!(
        v["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["city"],
        serde_json::json!("Tokyo")
    );
}

#[test]
fn gemini_serializes_v3_custom_parts_best_effort() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

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
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["text"],
        serde_json::json!("Hello")
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
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
        serde_json::json!("get_weather")
    );
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["city"],
        serde_json::json!("Tokyo")
    );
}

#[test]
fn gemini_serializes_v3_source_part_as_grounding_chunk() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "anthropic:source".to_string(),
            data: serde_json::json!({
                "type": "source",
                "sourceType": "url",
                "id": "src_1",
                "url": "https://www.rust-lang.org/",
                "title": "Rust",
            }),
        })
        .expect("serialize v3 source");

    let frames = parse_sse_json_frames(&bytes);
    assert_eq!(
        frames[0]["candidates"][0]["groundingMetadata"]["groundingChunks"][0]["web"]["uri"],
        serde_json::json!("https://www.rust-lang.org/")
    );
}

#[test]
fn gemini_serializes_v3_finish_part_as_finish_reason_chunk() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:finish".to_string(),
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
    assert_eq!(
        frames[0]["candidates"][0]["finishReason"],
        serde_json::json!("STOP")
    );
    assert_eq!(
        frames[0]["usageMetadata"]["totalTokenCount"],
        serde_json::json!(8)
    );
}

#[test]
fn gemini_serializes_v3_code_execution_tool_result_as_code_execution_result_part() {
    let config = create_test_config();
    let converter = GeminiEventConverter::new(config);

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "call_1",
                "toolName": "code_execution",
                "result": { "outcome": "OUTCOME_OK", "output": "1" }
            }),
        })
        .expect("serialize v3 tool-result");

    let frames = parse_sse_json_frames(&bytes);
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["codeExecutionResult"]["outcome"],
        serde_json::json!("OUTCOME_OK")
    );
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["codeExecutionResult"]["output"],
        serde_json::json!("1")
    );
}
