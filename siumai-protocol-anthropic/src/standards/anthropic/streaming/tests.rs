use super::super::params::{AnthropicParams, StructuredOutputMode};
use super::*;
use crate::execution::transformers::response::ResponseTransformer;
use crate::provider_metadata::anthropic::AnthropicChatResponseExt;
use crate::streaming::StreamProcessor;
use crate::types::ChatStreamPart;
use eventsource_stream::Event;

fn create_test_config() -> AnthropicParams {
    AnthropicParams::default()
}

fn stream_part(
    event: &Result<ChatStreamEvent, crate::error::LlmError>,
) -> Option<crate::streaming::LanguageModelV3StreamPart> {
    crate::streaming::LanguageModelV3StreamPart::try_from_chat_event(event.as_ref().ok()?)
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
async fn indexed_text_delta_emits_runtime_text_part() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let result = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    assert_eq!(result.len(), 1);
    match stream_part(result.first().unwrap()).expect("text part") {
        crate::streaming::LanguageModelV3StreamPart::TextDelta {
            id,
            delta,
            provider_metadata,
        } => {
            assert_eq!(id, "0");
            assert_eq!(delta, "Hello");
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("contentBlockIndex")),
                Some(&serde_json::json!(0))
            );
        }
        other => panic!("Expected TextDelta part, got {:?}", other),
    }
}

#[tokio::test]
async fn message_start_raw_chunk_follows_stream_start_before_response_metadata() {
    let converter =
        AnthropicEventConverter::new(create_test_config()).with_include_raw_chunks(true);

    let result = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_test","model":"claude-test","role":"assistant","usage":{"input_tokens":1,"output_tokens":0}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let parts: Vec<_> = result
        .into_iter()
        .map(|event| event.expect("stream event"))
        .filter_map(|event| match event {
            ChatStreamEvent::Part { part } => Some(part),
            _ => None,
        })
        .collect();

    assert!(matches!(
        parts.first(),
        Some(ChatStreamPart::StreamStart { .. })
    ));
    match parts.get(1).expect("raw part") {
        ChatStreamPart::Raw { raw_value } => {
            assert_eq!(raw_value["type"], serde_json::json!("message_start"));
        }
        other => panic!("expected raw part, got {other:?}"),
    }
    match parts.get(2).expect("response metadata part") {
        ChatStreamPart::ResponseMetadata(metadata) => {
            assert_eq!(metadata.id.as_deref(), Some("msg_test"));
            assert_eq!(metadata.model.as_deref(), Some("claude-test"));
        }
        other => panic!("expected response metadata, got {other:?}"),
    }
}

#[tokio::test]
async fn invalid_json_raw_chunk_still_follows_stream_start_before_parse_error() {
    let converter =
        AnthropicEventConverter::new(create_test_config()).with_include_raw_chunks(true);

    let result = converter
        .convert_event(Event {
            event: "".to_string(),
            data: "{ not json".to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    assert_eq!(result.len(), 4);
    match result.first().expect("stream-start event") {
        Ok(ChatStreamEvent::StreamStart { metadata }) => {
            assert_eq!(metadata.id, None);
            assert_eq!(metadata.model, None);
            assert_eq!(metadata.provider, "anthropic");
            assert!(metadata.created.is_some());
        }
        other => panic!("expected stream-start event, got {other:?}"),
    }
    match result.get(1).expect("stream-start part") {
        Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::StreamStart { warnings },
        }) => assert!(warnings.is_empty()),
        other => panic!("expected stream-start part, got {other:?}"),
    }
    match result.get(2).expect("raw part") {
        Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Raw { raw_value },
        }) => assert_eq!(
            raw_value,
            &serde_json::Value::String("{ not json".to_string())
        ),
        other => panic!("expected raw part, got {other:?}"),
    }
    match result.get(3).expect("parse error") {
        Err(crate::error::LlmError::ParseError(message)) => {
            assert!(message.contains("Failed to parse Anthropic event"));
        }
        other => panic!("expected parse error, got {other:?}"),
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

    let finish = out
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::Finish {
                finish_reason,
                provider_metadata,
                ..
            }) => Some((finish_reason, provider_metadata)),
            _ => None,
        })
        .expect("expected finish part");
    assert_eq!(finish.0.unified, "stop");

    let cm = finish
        .1
        .as_ref()
        .and_then(|meta| meta.get("anthropic"))
        .and_then(|meta| meta.get("contextManagement"))
        .cloned()
        .expect("context management metadata");
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
        .anthropic_metadata()
        .and_then(|meta| meta.context_management)
        .expect("typed context management metadata");
    assert_eq!(cm.applied_edits.len(), 1);
    match &cm.applied_edits[0] {
        crate::provider_metadata::anthropic::AnthropicContextManagementEdit::ClearToolUses20250919 {
            cleared_tool_uses,
            cleared_input_tokens,
        } => {
            assert_eq!(*cleared_tool_uses, 5);
            assert_eq!(*cleared_input_tokens, 10000);
        }
        other => panic!("unexpected context-management edit: {other:?}"),
    }
}

#[tokio::test]
async fn test_anthropic_stream_finish_replays_container_from_non_terminal_message_delta() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let start = Event {
        event: "".to_string(),
        data: r#"{"type":"message_start","message":{"id":"msg_container","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let _ = converter.convert_event(start).await;

    let delta = Event {
        event: "".to_string(),
        data: r#"{"type":"message_delta","delta":{"stop_reason":null,"stop_sequence":null,"container":{"id":"container_123","expires_at":"2025-10-14T10:28:40.590791Z","skills":[{"type":"custom","skill_id":"skill_alpha","version":"1"}]}},"usage":{"input_tokens":1,"output_tokens":0}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let _ = converter.convert_event(delta).await;

    let stop = Event {
        event: "".to_string(),
        data: r#"{"type":"message_stop"}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out = converter.convert_event(stop).await;

    let finish = out
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::Finish {
                provider_metadata, ..
            }) => Some(provider_metadata),
            _ => None,
        })
        .expect("expected finish part");
    let finish_container = finish
        .as_ref()
        .and_then(|meta| meta.get("anthropic"))
        .and_then(|meta| meta.get("container"))
        .cloned()
        .expect("finish container metadata");
    assert_eq!(finish_container["id"], serde_json::json!("container_123"));
    assert_eq!(
        finish_container["expiresAt"],
        serde_json::json!("2025-10-14T10:28:40.590791Z")
    );
    assert_eq!(
        finish_container["skills"][0]["skillId"],
        serde_json::json!("skill_alpha")
    );

    let end = out.iter().find_map(|r| match r.as_ref().ok() {
        Some(ChatStreamEvent::StreamEnd { response }) => Some(response.clone()),
        _ => None,
    });
    let end = end.expect("expected StreamEnd");
    let container = end
        .anthropic_metadata()
        .and_then(|meta| meta.container)
        .expect("container metadata");
    assert_eq!(container.id.as_deref(), Some("container_123"));
    assert_eq!(
        container.expires_at.as_deref(),
        Some("2025-10-14T10:28:40.590791Z")
    );
    let first_skill = container
        .skills
        .as_ref()
        .and_then(|skills| skills.first())
        .expect("first container skill");
    assert_eq!(first_skill.r#type.as_deref(), Some("custom"));
    assert_eq!(first_skill.skill_id.as_deref(), Some("skill_alpha"));
    assert_eq!(first_skill.version.as_deref(), Some("1"));
}

#[tokio::test]
async fn test_anthropic_message_delta_without_container_clears_message_start_container() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let start = Event {
        event: "".to_string(),
        data: r#"{"type":"message_start","message":{"id":"msg_container_clear","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0},"container":{"id":"container_123","expires_at":"2025-10-14T10:28:40.590791Z","skills":[{"type":"custom","skill_id":"skill_alpha","version":"1"}]}}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let _ = converter.convert_event(start).await;

    let delta = Event {
        event: "".to_string(),
        data: r#"{"type":"message_delta","delta":{"stop_reason":null,"stop_sequence":null},"usage":{"input_tokens":1,"output_tokens":0}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let _ = converter.convert_event(delta).await;

    let stop = Event {
        event: "".to_string(),
        data: r#"{"type":"message_stop"}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let out = converter.convert_event(stop).await;

    let finish = out
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::Finish {
                provider_metadata, ..
            }) => Some(provider_metadata),
            _ => None,
        })
        .expect("expected finish part");
    assert!(
        finish
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.get("container"))
            .is_some_and(serde_json::Value::is_null),
        "finish container should be cleared after message_delta without container"
    );

    let end = out.iter().find_map(|r| match r.as_ref().ok() {
        Some(ChatStreamEvent::StreamEnd { response }) => Some(response.clone()),
        _ => None,
    });
    let end = end.expect("expected StreamEnd");
    assert!(
        end.anthropic_metadata()
            .and_then(|meta| meta.container)
            .is_none(),
        "stream end container metadata should be cleared"
    );
    assert!(
        end.provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("anthropic"))
            .and_then(|metadata| metadata.get("container"))
            .is_some_and(serde_json::Value::is_null),
        "stream end raw provider metadata should keep container: null"
    );
    assert!(
        end.provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("anthropic"))
            .and_then(|metadata| metadata.get("contextManagement"))
            .is_some_and(serde_json::Value::is_null),
        "stream end raw provider metadata should keep contextManagement: null"
    );
}

#[tokio::test]
async fn test_anthropic_stream_finish_replays_stop_sequence_from_non_terminal_message_delta() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_stop_seq","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_delta","delta":{"stop_reason":null,"stop_sequence":"STOP"},"usage":{"input_tokens":1,"output_tokens":0}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let out = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let finish = out
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::Finish {
                provider_metadata, ..
            }) => Some(provider_metadata),
            _ => None,
        })
        .expect("expected finish part");
    assert_eq!(
        finish
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.get("stopSequence")),
        Some(&serde_json::json!("STOP"))
    );

    let end = out.iter().find_map(|r| match r.as_ref().ok() {
        Some(ChatStreamEvent::StreamEnd { response }) => Some(response.clone()),
        _ => None,
    });
    let end = end.expect("expected StreamEnd");
    assert_eq!(
        end.anthropic_metadata()
            .and_then(|meta| meta.stop_sequence)
            .as_deref(),
        Some("STOP")
    );
}

#[tokio::test]
async fn custom_provider_key_duplicates_finish_and_stream_end_metadata() {
    let converter = AnthropicEventConverter::new(create_test_config())
        .with_provider_metadata_key("my-custom-anthropic.messages");

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_custom_stop_seq","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_delta","delta":{"stop_reason":null,"stop_sequence":"STOP"},"usage":{"input_tokens":1,"output_tokens":0}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let out = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let finish = out
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::Finish {
                provider_metadata, ..
            }) => Some(provider_metadata),
            _ => None,
        })
        .expect("expected finish part");
    assert!(
        finish
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .is_some()
    );
    assert_eq!(
        finish
            .as_ref()
            .and_then(|meta| meta.get("my-custom-anthropic"))
            .and_then(|meta| meta.get("stopSequence")),
        Some(&serde_json::json!("STOP"))
    );

    let end = out.iter().find_map(|r| match r.as_ref().ok() {
        Some(ChatStreamEvent::StreamEnd { response }) => Some(response.clone()),
        _ => None,
    });
    let end = end.expect("expected StreamEnd");
    assert_eq!(
        end.anthropic_metadata_with_key("my-custom-anthropic")
            .and_then(|meta| meta.stop_sequence)
            .as_deref(),
        Some("STOP")
    );
}

#[tokio::test]
async fn test_anthropic_message_delta_without_stop_sequence_clears_older_value() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_stop_seq_clear","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":"START","usage":{"input_tokens":0,"output_tokens":0}}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_delta","delta":{"stop_reason":null,"stop_sequence":"STOP"},"usage":{"input_tokens":1,"output_tokens":0}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_delta","delta":{"stop_reason":null},"usage":{"input_tokens":2,"output_tokens":0}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let out = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let finish = out
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::Finish {
                provider_metadata, ..
            }) => Some(provider_metadata),
            _ => None,
        })
        .expect("expected finish part");
    assert!(
        finish
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.get("stopSequence"))
            .is_some_and(serde_json::Value::is_null),
        "finish stopSequence should be cleared after message_delta without stop_sequence"
    );

    let end = out.iter().find_map(|r| match r.as_ref().ok() {
        Some(ChatStreamEvent::StreamEnd { response }) => Some(response.clone()),
        _ => None,
    });
    let end = end.expect("expected StreamEnd");
    assert!(
        end.anthropic_metadata()
            .and_then(|meta| meta.stop_sequence)
            .is_none(),
        "stream end stop_sequence should be cleared"
    );
}

#[tokio::test]
async fn test_anthropic_stream_finish_preserves_extended_usage_fields() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_usage","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":45000,"output_tokens":0,"cache_creation_input_tokens":10,"cache_read_input_tokens":5,"cache_creation":{"ephemeral_5m_input_tokens":0},"inference_geo":"not_available","service_tier":"standard","server_tool_use":{"web_search_requests":2},"iterations":[{"type":"compaction","input_tokens":180000,"output_tokens":3500},{"type":"message","input_tokens":23000,"output_tokens":0}]}}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let out = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":45000,"output_tokens":1234,"cache_creation_input_tokens":10,"cache_read_input_tokens":5,"cache_creation":{"ephemeral_5m_input_tokens":0},"inference_geo":"not_available","service_tier":"standard","server_tool_use":{"web_search_requests":2},"iterations":[{"type":"compaction","input_tokens":180000,"output_tokens":3500},{"type":"message","input_tokens":23000,"output_tokens":1000}]}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let (finish_usage, finish_metadata) = out
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::Finish {
                usage,
                provider_metadata,
                ..
            }) => Some((usage, provider_metadata)),
            _ => None,
        })
        .expect("expected finish part");
    assert_eq!(finish_usage.input_tokens.no_cache, Some(203000));
    assert_eq!(finish_usage.input_tokens.total, Some(203015));
    assert_eq!(finish_usage.output_tokens.total, Some(4500));
    assert_eq!(
        finish_metadata
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.pointer("/iterations/0/inputTokens")),
        Some(&serde_json::json!(180000))
    );
    assert_eq!(
        finish_metadata
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.pointer("/iterations/1/outputTokens")),
        Some(&serde_json::json!(1000))
    );

    let end = out.iter().find_map(|r| match r.as_ref().ok() {
        Some(ChatStreamEvent::StreamEnd { response }) => Some(response.clone()),
        _ => None,
    });
    let end = end.expect("expected StreamEnd");

    let usage = end.usage.clone().expect("usage");
    assert_eq!(usage.prompt_tokens(), Some(203000));
    assert_eq!(usage.completion_tokens(), Some(4500));
    assert_eq!(usage.normalized_input_tokens().total, Some(203015));
    assert_eq!(usage.normalized_input_tokens().no_cache, Some(203000));
    assert_eq!(
        usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens),
        Some(5)
    );
    assert_eq!(
        usage.raw_usage_value(),
        Some(serde_json::json!({
            "input_tokens": 45000,
            "output_tokens": 1234,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 5,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0
            },
            "inference_geo": "not_available",
            "service_tier": "standard",
            "server_tool_use": {
                "web_search_requests": 2
            },
            "iterations": [
                {
                    "type": "compaction",
                    "input_tokens": 180000,
                    "output_tokens": 3500
                },
                {
                    "type": "message",
                    "input_tokens": 23000,
                    "output_tokens": 1000
                }
            ]
        }))
    );
    assert_eq!(end.service_tier.as_deref(), Some("standard"));

    let meta = end.anthropic_metadata().expect("anthropic metadata");
    assert_eq!(meta.cache_creation_input_tokens, Some(10));
    assert_eq!(meta.cache_read_input_tokens, Some(5));
    assert_eq!(meta.service_tier.as_deref(), Some("standard"));
    assert_eq!(
        meta.server_tool_use
            .as_ref()
            .and_then(|usage| usage.web_search_requests),
        Some(2)
    );
    let iterations = meta.iterations.as_ref().expect("iterations");
    assert_eq!(iterations.len(), 2);
    assert_eq!(iterations[0].r#type, "compaction");
    assert_eq!(iterations[0].input_tokens, 180000);
    assert_eq!(iterations[0].output_tokens, 3500);
    assert_eq!(iterations[1].r#type, "message");
    assert_eq!(iterations[1].input_tokens, 23000);
    assert_eq!(iterations[1].output_tokens, 1000);
    assert_eq!(
        meta.usage
            .as_ref()
            .and_then(|usage| usage.get("server_tool_use"))
            .and_then(|value| value.get("web_search_requests"))
            .and_then(|value| value.as_u64()),
        Some(2)
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
async fn message_start_emits_runtime_stream_start_and_response_metadata_parts() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let out = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_test","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

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
                if metadata.id.as_deref() == Some("msg_test")
                    && metadata.model_id.as_deref() == Some("claude-test")
        )
    }));
}

#[tokio::test]
async fn stream_end_includes_accumulated_text_and_reasoning_content() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let events = [
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_test","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning..."}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":1}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    let mut end = None;
    for event in events {
        let out = converter.convert_event(event).await;
        if let Some(response) = out.into_iter().flatten().find_map(|event| match event {
            ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        }) {
            end = Some(response);
            break;
        }
    }

    let response = end.expect("expected stream end");
    assert_eq!(response.id.as_deref(), Some("msg_test"));
    assert_eq!(response.model.as_deref(), Some("claude-test"));
    assert_eq!(response.content_text(), Some("hello"));
    assert_eq!(response.reasoning(), vec!["reasoning...".to_string()]);
}

#[tokio::test]
async fn thinking_blocks_emit_runtime_reasoning_parts() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let start = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    assert!(matches!(
        stream_part(start.first().expect("start event")),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningStart { ref id, .. })
            if id == "0"
    ));

    let delta = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning..."}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    assert!(matches!(
        stream_part(delta.first().expect("delta event")),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningDelta { ref id, ref delta, .. })
            if id == "0" && delta == "reasoning..."
    ));

    let end = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    assert!(matches!(
        stream_part(end.first().expect("end event")),
        Some(crate::streaming::LanguageModelV3StreamPart::ReasoningEnd { ref id, .. })
            if id == "0"
    ));
}

#[tokio::test]
async fn compaction_blocks_emit_runtime_text_parts_with_provider_metadata() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let start = converter
        .convert_event(Event {
            event: "".to_string(),
            data:
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"compaction"}}"#
                    .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    match stream_part(start.first().expect("start event")).expect("text-start part") {
        crate::streaming::LanguageModelV3StreamPart::TextStart {
            id,
            provider_metadata,
        } => {
            assert_eq!(id, "0");
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("type")),
                Some(&serde_json::json!("compaction"))
            );
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("contentBlockIndex")),
                Some(&serde_json::json!(0))
            );
        }
        other => panic!("expected TextStart part, got {other:?}"),
    }

    let delta = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"Condensed"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    match stream_part(delta.first().expect("delta event")).expect("text-delta part") {
        crate::streaming::LanguageModelV3StreamPart::TextDelta {
            id,
            delta,
            provider_metadata,
        } => {
            assert_eq!(id, "0");
            assert_eq!(delta, "Condensed");
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("type")),
                Some(&serde_json::json!("compaction"))
            );
        }
        other => panic!("expected TextDelta part, got {other:?}"),
    }

    let end = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    match stream_part(end.first().expect("end event")).expect("text-end part") {
        crate::streaming::LanguageModelV3StreamPart::TextEnd {
            id,
            provider_metadata,
        } => {
            assert_eq!(id, "0");
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("type")),
                Some(&serde_json::json!("compaction"))
            );
        }
        other => panic!("expected TextEnd part, got {other:?}"),
    }
}

#[tokio::test]
async fn compaction_text_is_accumulated_into_stream_end_content() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let events = [
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_compaction","model":"claude-test","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"compaction"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"Condensed"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    let mut stream_end = None;
    for event in events {
        let out = converter.convert_event(event).await;
        if let Some(response) = out.into_iter().flatten().find_map(|event| match event {
            ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        }) {
            stream_end = Some(response);
            break;
        }
    }

    let response = stream_end.expect("expected stream end");
    assert_eq!(response.id.as_deref(), Some("msg_compaction"));
    assert_eq!(response.content_text(), Some("Condensed"));
}

#[tokio::test]
async fn message_start_preloaded_tool_use_emits_runtime_tool_parts_and_caller_metadata() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let out = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_preloaded","model":"claude-test","type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"code_execution","input":{"code":"print(1)"},"caller":{"type":"server_tool_use","tool_id":"srvtoolu_1"}}],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    assert!(out.iter().any(|event| {
        matches!(
            stream_part(event),
            Some(crate::streaming::LanguageModelV3StreamPart::ToolInputStart {
                ref id,
                ref tool_name,
                ..
            }) if id == "toolu_1" && tool_name == "code_execution"
        )
    }));

    let input_delta = out
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::ToolInputDelta {
                id,
                delta,
                provider_metadata,
            }) if id == "toolu_1" => Some((delta, provider_metadata)),
            _ => None,
        })
        .expect("tool-input-delta");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&input_delta.0).expect("input delta json"),
        serde_json::json!({ "code": "print(1)" })
    );
    assert_eq!(
        input_delta
            .1
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.get("contentBlockIndex")),
        Some(&serde_json::json!(0))
    );

    let tool_call = out
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::ToolCall(call))
                if call.tool_call_id == "toolu_1" =>
            {
                Some(call)
            }
            _ => None,
        })
        .expect("tool-call part");
    assert_eq!(tool_call.tool_name, "code_execution");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&tool_call.input).expect("tool input json"),
        serde_json::json!({ "code": "print(1)" })
    );
    assert_eq!(
        tool_call
            .provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.get("caller"))
            .and_then(|caller| caller.get("type")),
        Some(&serde_json::json!("server_tool_use"))
    );
    assert_eq!(
        tool_call
            .provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.get("caller"))
            .and_then(|caller| caller.get("toolId")),
        Some(&serde_json::json!("srvtoolu_1"))
    );
}

#[tokio::test]
async fn tool_use_blocks_preserve_caller_metadata_on_tool_call() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"code_execution","input":{"code":"print(1)"},"caller":{"type":"server_tool_use","tool_id":"srvtoolu_1"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let end = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let tool_call = end
        .iter()
        .find_map(|event| match stream_part(event) {
            Some(crate::streaming::LanguageModelV3StreamPart::ToolCall(call))
                if call.tool_call_id == "toolu_1" =>
            {
                Some(call)
            }
            _ => None,
        })
        .expect("tool-call part");
    assert_eq!(tool_call.tool_name, "code_execution");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&tool_call.input).expect("tool input json"),
        serde_json::json!({ "code": "print(1)" })
    );
    assert_eq!(
        tool_call
            .provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.get("caller"))
            .and_then(|caller| caller.get("type")),
        Some(&serde_json::json!("server_tool_use"))
    );
    assert_eq!(
        tool_call
            .provider_metadata
            .as_ref()
            .and_then(|meta| meta.get("anthropic"))
            .and_then(|meta| meta.get("caller"))
            .and_then(|caller| caller.get("toolId")),
        Some(&serde_json::json!("srvtoolu_1"))
    );
}

#[tokio::test]
async fn emits_runtime_parts_for_provider_hosted_server_tool_use_and_results() {
    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);

    let web_search_start = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"rust"}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(web_search_start).await;
    assert_eq!(evs.len(), 2);
    match stream_part(evs.first().unwrap()).expect("web_search tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart {
            id,
            tool_name,
            provider_executed,
            dynamic,
            provider_metadata,
            ..
        } => {
            assert_eq!(id, "srvtoolu_1");
            assert_eq!(tool_name, "web_search");
            assert_eq!(provider_executed, Some(true));
            assert_eq!(dynamic, None);
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("contentBlockIndex")),
                Some(&serde_json::json!(0))
            );
        }
        other => panic!("Expected ToolInputStart part, got {:?}", other),
    }
    match stream_part(evs.get(1).unwrap()).expect("web_search initial tool-input-delta part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputDelta { id, delta, .. } => {
            assert_eq!(id, "srvtoolu_1");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&delta).expect("web_search input json"),
                serde_json::json!({ "query": "rust" })
            );
        }
        other => panic!("Expected ToolInputDelta part, got {:?}", other),
    }

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    assert_eq!(evs.len(), 2);
    assert!(matches!(
        stream_part(evs.first().unwrap()),
        Some(crate::streaming::LanguageModelV3StreamPart::ToolInputEnd { ref id, .. })
            if id == "srvtoolu_1"
    ));
    match stream_part(evs.get(1).unwrap()).expect("web_search tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "srvtoolu_1");
            assert_eq!(call.tool_name, "web_search");
            assert_eq!(call.provider_executed, Some(true));
            assert_eq!(call.dynamic, None);
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&call.input)
                    .expect("web_search call input json"),
                serde_json::json!({ "query": "rust" })
            );
        }
        other => panic!("Expected ToolCall part, got {:?}", other),
    }

    let web_search_result = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","title":"Rust","url":"https://www.rust-lang.org","encrypted_content":"..."}]}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(web_search_result).await;
    assert_eq!(evs.len(), 2);
    match stream_part(evs.first().unwrap()).expect("web_search tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "srvtoolu_1");
            assert_eq!(result.tool_name, "web_search");
            assert_eq!(result.is_error, Some(false));
            assert_eq!(result.dynamic, None);
            assert!(result.result.is_array());
            assert_eq!(result.result[0]["pageAge"], serde_json::Value::Null);
            assert_eq!(
                result.result[0]["encryptedContent"],
                serde_json::json!("...")
            );
            assert!(result.result[0].get("page_age").is_none());
            assert!(result.result[0].get("encrypted_content").is_none());
            assert_eq!(
                result
                    .provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("contentBlockIndex")),
                Some(&serde_json::json!(1))
            );
        }
        other => panic!("Expected ToolResult part, got {:?}", other),
    }

    match evs.get(1).unwrap().as_ref().unwrap() {
        ChatStreamEvent::Part { part } => match part {
            ChatStreamPart::Source {
                id,
                source: crate::types::SourcePart::Url { url, title: _ },
                provider_metadata,
            } => {
                assert_eq!(id, "id-0");
                assert_eq!(url, "https://www.rust-lang.org");
                assert_eq!(
                    provider_metadata
                        .as_ref()
                        .and_then(|meta| meta.get("anthropic"))
                        .and_then(|meta| meta.get("pageAge")),
                    Some(&serde_json::Value::Null)
                );
            }
            other => panic!("Expected Source part, got {:?}", other),
        },
        other => panic!("Expected source Part event, got {:?}", other),
    }

    let web_fetch_result = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":2,"content_block":{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_2","content":{"type":"web_fetch_result","url":"https://example.com","retrieved_at":"2025-01-01T00:00:00Z","content":{"type":"document","title":"Example","citations":{"enabled":true},"source":{"type":"text","media_type":"text/plain","data":"hello"}}}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(web_fetch_result).await;
    assert_eq!(evs.len(), 1);
    match stream_part(evs.first().unwrap()).expect("web_fetch tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "srvtoolu_2");
            assert_eq!(result.tool_name, "web_fetch");
            assert_eq!(result.is_error, Some(false));
            assert_eq!(result.result["type"], serde_json::json!("web_fetch_result"));
            assert_eq!(
                result.result["retrievedAt"],
                serde_json::json!("2025-01-01T00:00:00Z")
            );
            assert_eq!(
                result.result["content"]["source"]["mediaType"],
                serde_json::json!("text/plain")
            );
        }
        other => panic!("Expected ToolResult part, got {:?}", other),
    }

    let tool_search_start = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":3,"content_block":{"type":"server_tool_use","id":"srvtoolu_3","name":"tool_search_tool_regex","input":{"pattern":"weather","limit":2}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(tool_search_start).await;
    assert_eq!(evs.len(), 2);
    match stream_part(evs.first().unwrap()).expect("tool_search tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart {
            id,
            tool_name,
            provider_executed,
            provider_metadata,
            ..
        } => {
            assert_eq!(id, "srvtoolu_3");
            assert_eq!(tool_name, "tool_search");
            assert_eq!(provider_executed, Some(true));
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("serverToolName")),
                Some(&serde_json::json!("tool_search_tool_regex"))
            );
        }
        other => panic!("Expected ToolInputStart part, got {:?}", other),
    }
    match stream_part(evs.get(1).unwrap()).expect("tool_search initial tool-input-delta part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputDelta { id, delta, .. } => {
            assert_eq!(id, "srvtoolu_3");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&delta).expect("tool_search input json"),
                serde_json::json!({ "pattern": "weather", "limit": 2 })
            );
        }
        other => panic!("Expected ToolInputDelta part, got {:?}", other),
    }

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":3}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    assert_eq!(evs.len(), 2);
    match stream_part(evs.get(1).unwrap()).expect("tool_search tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "srvtoolu_3");
            assert_eq!(call.tool_name, "tool_search");
            assert_eq!(call.provider_executed, Some(true));
            assert_eq!(
                call.provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("serverToolName")),
                Some(&serde_json::json!("tool_search_tool_regex"))
            );
        }
        other => panic!("Expected ToolCall part, got {:?}", other),
    }

    let tool_search_result = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":4,"content_block":{"type":"tool_search_tool_result","tool_use_id":"srvtoolu_3","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"get_weather"}]}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(tool_search_result).await;
    assert_eq!(evs.len(), 1);
    match stream_part(evs.first().unwrap()).expect("tool_search tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "srvtoolu_3");
            assert_eq!(result.tool_name, "tool_search");
            assert_eq!(result.is_error, Some(false));
            assert!(result.result.is_array());
            assert_eq!(
                result.result[0]["toolName"],
                serde_json::json!("get_weather")
            );
            assert_eq!(
                result
                    .provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("serverToolName")),
                Some(&serde_json::json!("tool_search_tool_regex"))
            );
        }
        other => panic!("Expected ToolResult part, got {:?}", other),
    }

    let code_exec_start = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":5,"content_block":{"type":"server_tool_use","id":"srvtoolu_4","name":"code_execution","input":{"code":"print(1+1)"}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(code_exec_start).await;
    assert_eq!(evs.len(), 2);
    match stream_part(evs.first().unwrap()).expect("code_execution tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart {
            id,
            tool_name,
            provider_executed,
            ..
        } => {
            assert_eq!(id, "srvtoolu_4");
            assert_eq!(tool_name, "code_execution");
            assert_eq!(provider_executed, Some(true));
        }
        other => panic!("Expected ToolInputStart part, got {:?}", other),
    }
    match stream_part(evs.get(1).unwrap()).expect("code_execution initial tool-input-delta part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputDelta { delta, .. } => {
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&delta)
                    .expect("code_execution input json"),
                serde_json::json!({
                    "type": "programmatic-tool-call",
                    "code": "print(1+1)"
                })
            );
        }
        other => panic!("Expected ToolInputDelta part, got {:?}", other),
    }

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":5}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    assert_eq!(evs.len(), 2);
    match stream_part(evs.get(1).unwrap()).expect("code_execution tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "srvtoolu_4");
            assert_eq!(call.tool_name, "code_execution");
            assert_eq!(call.provider_executed, Some(true));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&call.input)
                    .expect("code_execution call input json"),
                serde_json::json!({
                    "type": "programmatic-tool-call",
                    "code": "print(1+1)"
                })
            );
        }
        other => panic!("Expected ToolCall part, got {:?}", other),
    }

    let code_exec_result = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_start","index":6,"content_block":{"type":"code_execution_tool_result","tool_use_id":"srvtoolu_4","content":{"type":"code_execution_result","stdout":"2\n","stderr":"","return_code":0}}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let evs = converter.convert_event(code_exec_result).await;
    assert_eq!(evs.len(), 1);
    match stream_part(evs.first().unwrap()).expect("code_execution tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "srvtoolu_4");
            assert_eq!(result.tool_name, "code_execution");
            assert_eq!(result.is_error, Some(false));
            assert_eq!(
                result.result["type"],
                serde_json::json!("code_execution_result")
            );
            assert_eq!(result.result["return_code"], serde_json::json!(0));
        }
        other => panic!("Expected ToolResult part, got {:?}", other),
    }
}

#[tokio::test]
async fn marks_code_execution_dynamic_for_2026_web_tool_injection() {
    let converter = AnthropicEventConverter::new(
        create_test_config().with_tools(&[crate::tools::anthropic::web_search_20260209()]),
    );

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_dyn","name":"code_execution","input":{"code":"print(1)"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    match stream_part(evs.first().unwrap()).expect("code_execution tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart { dynamic, .. } => {
            assert_eq!(dynamic, Some(true));
        }
        other => panic!("Expected ToolInputStart part, got {:?}", other),
    }

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    match stream_part(evs.get(1).unwrap()).expect("code_execution tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "srvtoolu_dyn");
            assert_eq!(call.dynamic, Some(true));
        }
        other => panic!("Expected ToolCall part, got {:?}", other),
    }
}

#[tokio::test]
async fn explicit_code_execution_tool_disables_dynamic_marking_for_2026_web_tool_injection() {
    let converter = AnthropicEventConverter::new(create_test_config().with_tools(&[
        crate::tools::anthropic::web_fetch_20260209(),
        crate::tools::anthropic::code_execution_20260120(),
    ]));

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_static","name":"code_execution","input":{"code":"print(1)"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    match stream_part(evs.first().unwrap()).expect("code_execution tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart { dynamic, .. } => {
            assert_eq!(dynamic, None);
        }
        other => panic!("Expected ToolInputStart part, got {:?}", other),
    }

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    match stream_part(evs.get(1).unwrap()).expect("code_execution tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "srvtoolu_static");
            assert_eq!(call.dynamic, None);
        }
        other => panic!("Expected ToolCall part, got {:?}", other),
    }
}

#[tokio::test]
async fn emits_runtime_parts_for_mcp_tool_use_and_result() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"mcp_tool_use","id":"mcptoolu_1","name":"echo","server_name":"echo-prod","input":{"message":"hello"}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    assert_eq!(evs.len(), 1);
    match stream_part(evs.first().unwrap()).expect("mcp tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "mcptoolu_1");
            assert_eq!(call.tool_name, "echo");
            assert_eq!(call.provider_executed, Some(true));
            assert_eq!(call.dynamic, Some(true));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&call.input).expect("mcp input json"),
                serde_json::json!({ "message": "hello" })
            );
            assert_eq!(
                call.provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("type")),
                Some(&serde_json::json!("mcp-tool-use"))
            );
            assert_eq!(
                call.provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("serverName")),
                Some(&serde_json::json!("echo-prod"))
            );
        }
        other => panic!("Expected MCP ToolCall part, got {:?}", other),
    }

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":1,"content_block":{"type":"mcp_tool_result","tool_use_id":"mcptoolu_1","is_error":false,"content":[{"type":"text","text":"Tool echo: hello"}]}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    assert_eq!(evs.len(), 1);
    match stream_part(evs.first().unwrap()).expect("mcp tool-result part") {
        crate::streaming::LanguageModelV3StreamPart::ToolResult(result) => {
            assert_eq!(result.tool_call_id, "mcptoolu_1");
            assert_eq!(result.tool_name, "echo");
            assert_eq!(result.is_error, Some(false));
            assert_eq!(result.dynamic, Some(true));
            assert_eq!(
                result
                    .provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("type")),
                Some(&serde_json::json!("mcp-tool-use"))
            );
            assert_eq!(
                result
                    .provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("serverName")),
                Some(&serde_json::json!("echo-prod"))
            );
            assert_eq!(
                result.result,
                serde_json::json!([{ "type": "text", "text": "Tool echo: hello" }])
            );
        }
        other => panic!("Expected MCP ToolResult part, got {:?}", other),
    }
}

#[tokio::test]
async fn emits_runtime_tool_input_parts_for_local_tool_use_input_json_delta() {
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
    assert_eq!(evs.len(), 2);
    match stream_part(evs.first().unwrap()).expect("tool-input-start part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputStart {
            id,
            tool_name,
            provider_metadata,
            ..
        } => {
            assert_eq!(id, "toolu_1");
            assert_eq!(tool_name, "get_weather");
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("contentBlockIndex")),
                Some(&serde_json::json!(0))
            );
        }
        other => panic!("Expected ToolInputStart, got {:?}", other),
    }
    match stream_part(evs.get(1).unwrap()).expect("tool-input-delta part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputDelta {
            id,
            delta,
            provider_metadata,
        } => {
            assert_eq!(id, "toolu_1");
            assert_eq!(delta, r#"{"location":"tokyo"}"#);
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("contentBlockIndex")),
                Some(&serde_json::json!(0))
            );
        }
        other => panic!("Expected initial ToolInputDelta, got {:?}", other),
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
    match stream_part(evs.first().unwrap()).expect("tool-input-delta part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputDelta {
            id,
            delta,
            provider_metadata,
        } => {
            assert_eq!(id, "toolu_1");
            assert_eq!(delta, "{\"unit\":\"c\"}");
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("contentBlockIndex")),
                Some(&serde_json::json!(0))
            );
        }
        other => panic!("Expected ToolInputDelta, got {:?}", other),
    }
}

#[tokio::test]
async fn tool_use_stop_emits_runtime_tool_input_end_and_tool_call_part() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;
    let _ = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Tokyo\"}"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    let evs = converter
        .convert_event(Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        })
        .await;

    assert_eq!(evs.len(), 2);
    match stream_part(evs.first().unwrap()).expect("tool-input-end part") {
        crate::streaming::LanguageModelV3StreamPart::ToolInputEnd {
            id,
            provider_metadata,
        } => {
            assert_eq!(id, "toolu_1");
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("contentBlockIndex")),
                Some(&serde_json::json!(0))
            );
        }
        other => panic!("Expected ToolInputEnd, got {:?}", other),
    }
    match stream_part(evs.get(1).unwrap()).expect("tool-call part") {
        crate::streaming::LanguageModelV3StreamPart::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "toolu_1");
            assert_eq!(call.tool_name, "get_weather");
            assert_eq!(call.input, r#"{"city":"Tokyo"}"#);
        }
        other => panic!("Expected ToolCall part, got {:?}", other),
    }
}

#[tokio::test]
async fn streaming_tool_calls_match_non_streaming_tool_calls() {
    let non_stream_raw = serde_json::json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-7-sonnet-latest",
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": { "input_tokens": 1, "output_tokens": 2 },
        "content": [
            { "type": "tool_use", "id": "toolu_1", "name": "weather", "input": { "city": "Tokyo" } }
        ]
    });

    let tx = crate::standards::anthropic::transformers::AnthropicResponseTransformer::default();
    let non_stream = tx
        .transform_chat_response(&non_stream_raw)
        .expect("non-stream transform");
    assert_eq!(non_stream.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(non_stream.tool_calls().len(), 1);

    let config = create_test_config();
    let converter = AnthropicEventConverter::new(config);
    let mut sp = StreamProcessor::new();
    let mut finish_reason = None;

    let stream_events = vec![
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"weather","input":{}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Tokyo\"}"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":1,"output_tokens":1}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    for e in stream_events {
        for ev in converter.convert_event(e).await.into_iter().flatten() {
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
    assert_eq!(streaming.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(streaming.tool_calls().len(), 1);

    let a = non_stream.tool_calls()[0]
        .as_tool_call()
        .expect("tool call info");
    let b = streaming.tool_calls()[0]
        .as_tool_call()
        .expect("tool call info");
    assert_eq!(b.tool_name, a.tool_name);
    assert_eq!(b.arguments, a.arguments);
    // Anthropic provides stable tool_use ids; keep it invariant across streaming/non-streaming.
    assert_eq!(b.tool_call_id, a.tool_call_id);
}

#[tokio::test]
async fn streaming_reserved_json_tool_matches_non_streaming_structured_output() {
    let non_stream_raw = serde_json::json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-7-sonnet-latest",
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": { "input_tokens": 1, "output_tokens": 2 },
        "content": [
            { "type": "tool_use", "id": "toolu_1", "name": "json", "input": { "value": "ok" } }
        ]
    });

    let tx = crate::standards::anthropic::transformers::AnthropicResponseTransformer::default();
    let non_stream = tx
        .transform_chat_response(&non_stream_raw)
        .expect("non-stream transform");
    assert_eq!(non_stream.finish_reason, Some(FinishReason::Stop));
    assert_eq!(non_stream.content_text(), Some(r#"{"value":"ok"}"#));
    assert!(non_stream.tool_calls().is_empty());

    let config = create_test_config().with_structured_output_mode(StructuredOutputMode::JsonTool);
    let converter = AnthropicEventConverter::new(config);
    let mut sp = StreamProcessor::new();
    let mut finish_reason = None;

    let stream_events = vec![
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"json","input":{}}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"value\":\"ok\"}"}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":1,"output_tokens":1}}"#
                .to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    for e in stream_events {
        for ev in converter.convert_event(e).await.into_iter().flatten() {
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
    assert_eq!(streaming.finish_reason, Some(FinishReason::Stop));
    assert_eq!(streaming.content_text(), Some(r#"{"value":"ok"}"#));
    assert!(streaming.tool_calls().is_empty());
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
    match stream_part(evs.first().unwrap()).expect("reasoning-delta part") {
        crate::streaming::LanguageModelV3StreamPart::ReasoningDelta {
            id,
            delta,
            provider_metadata,
        } => {
            assert_eq!(id, "0");
            assert_eq!(delta, "");
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("signature")),
                Some(&serde_json::json!("sig-1"))
            );
        }
        other => panic!("Expected signature delta reasoning part, got {:?}", other),
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
            let meta = response.anthropic_metadata().expect("anthropic metadata");
            assert_eq!(meta.thinking_signature.as_deref(), Some("sig-1"));
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
    match stream_part(evs.first().unwrap()).expect("reasoning-start part") {
        crate::streaming::LanguageModelV3StreamPart::ReasoningStart {
            id,
            provider_metadata,
        } => {
            assert_eq!(id, "0");
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("redactedData")),
                Some(&serde_json::json!("abc123"))
            );
        }
        other => panic!("Expected reasoning-start part, got {:?}", other),
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
            let meta = response.anthropic_metadata().expect("anthropic metadata");
            assert_eq!(meta.redacted_thinking_data.as_deref(), Some("abc123"));
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
    match stream_part(out.first().unwrap()).expect("source part") {
        crate::streaming::LanguageModelV3StreamPart::Source(
            crate::streaming::LanguageModelV3Source::Document {
                id,
                media_type,
                title,
                filename,
                provider_metadata,
            },
        ) => {
            assert_eq!(id, "id-0");
            assert_eq!(media_type, "application/pdf");
            assert_eq!(title, "Doc A");
            assert_eq!(filename.as_deref(), Some("a.pdf"));
            assert_eq!(
                provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("anthropic"))
                    .and_then(|meta| meta.get("startPageNumber")),
                Some(&serde_json::json!(1))
            );
        }
        other => panic!("Expected source part, got {:?}", other),
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
            let sources = response
                .anthropic_metadata()
                .and_then(|meta| meta.sources)
                .expect("sources array");
            assert_eq!(sources.len(), 1);
            assert_eq!(sources[0].id, "id-0");
            assert_eq!(sources[0].source_type, "document");
            assert_eq!(sources[0].media_type.as_deref(), Some("application/pdf"));
            assert_eq!(sources[0].filename.as_deref(), Some("a.pdf"));
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
                headers: None,
            },
        })
        .expect("serialize start");
    let start_frames = parse_sse_frames(&start);
    assert_eq!(start_frames.len(), 1);
    assert_eq!(start_frames[0].event.as_deref(), Some("message_start"));
    assert_eq!(
        start_frames[0].data["type"],
        serde_json::json!("message_start")
    );

    let delta = converter
        .serialize_event(&ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        })
        .expect("serialize delta");
    let delta_frames = parse_sse_frames(&delta);
    assert!(
        delta_frames
            .iter()
            .any(|v| v.event.as_deref() == Some("content_block_start")
                && v.data["type"] == "content_block_start")
    );
    assert!(
        delta_frames
            .iter()
            .any(|v| v.event.as_deref() == Some("content_block_delta")
                && v.data["type"] == "content_block_delta")
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
                raw_finish_reason: None,
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            },
        })
        .expect("serialize end");
    let end_frames = parse_sse_frames(&end);
    assert!(end_frames.iter().any(|v| {
        v.event.as_deref() == Some("content_block_stop") && v.data["type"] == "content_block_stop"
    }));
    assert!(end_frames.iter().any(|v| {
        v.event.as_deref() == Some("message_delta") && v.data["type"] == "message_delta"
    }));
    assert!(end_frames.iter().any(|v| {
        v.event.as_deref() == Some("message_stop") && v.data["type"] == "message_stop"
    }));
}

#[test]
fn serializes_stream_end_with_raw_anthropic_stop_reason() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: Some("msg_json".to_string()),
                model: Some("claude-test".to_string()),
                content: MessageContent::Text(String::new()),
                usage: None,
                finish_reason: Some(FinishReason::Stop),
                raw_finish_reason: Some("tool_use".to_string()),
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            },
        })
        .expect("serialize end");

    let frames = parse_sse_frames(&bytes);
    let message_delta = frames
        .iter()
        .find(|frame| frame.event.as_deref() == Some("message_delta"))
        .expect("message_delta frame");
    assert_eq!(
        message_delta.data["delta"]["stop_reason"],
        serde_json::json!("tool_use")
    );
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
                    headers: None,
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
                    raw_finish_reason: None,
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
            .all(|f| { f.event.as_deref() == f.data.get("type").and_then(|v| v.as_str()) }),
        "expected every frame event name to match payload type"
    );
}

#[test]
fn serializes_interleaved_blocks_as_separate_monotonic_content_blocks() {
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
                    headers: None,
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
            .expect("serialize text delta 1"),
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
            .serialize_event(&ChatStreamEvent::ContentDelta {
                delta: " world".to_string(),
                index: None,
            })
            .expect("serialize text delta 2"),
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
                    raw_finish_reason: None,
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
    let typed: Vec<serde_json::Value> = frames.iter().map(|frame| frame.data.clone()).collect();
    let types: Vec<&str> = typed
        .iter()
        .filter_map(|value| value.get("type").and_then(|v| v.as_str()))
        .collect();

    assert_eq!(
        types,
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ],
        "expected monotonic block ordering, got: {typed:?}"
    );

    let block_starts: Vec<(u64, &str)> = typed
        .iter()
        .filter_map(|value| {
            (value.get("type").and_then(|v| v.as_str()) == Some("content_block_start")).then(|| {
                (
                    value["index"].as_u64().expect("start index"),
                    value["content_block"]["type"].as_str().expect("block type"),
                )
            })
        })
        .collect();
    assert_eq!(
        block_starts,
        vec![(0, "text"), (1, "thinking"), (2, "text")]
    );

    let block_deltas: Vec<u64> = typed
        .iter()
        .filter_map(|value| {
            (value.get("type").and_then(|v| v.as_str()) == Some("content_block_delta"))
                .then(|| value["index"].as_u64().expect("delta index"))
        })
        .collect();
    assert_eq!(block_deltas, vec![0, 1, 2]);

    let block_stops: Vec<u64> = typed
        .iter()
        .filter_map(|value| {
            (value.get("type").and_then(|v| v.as_str()) == Some("content_block_stop"))
                .then(|| value["index"].as_u64().expect("stop index"))
        })
        .collect();
    assert_eq!(block_stops, vec![0, 1, 2]);
}

#[test]
fn serializes_repeated_thinking_deltas_with_single_start_and_single_stop() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(
        &converter
            .serialize_event(&ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("msg_thinking".to_string()),
                    model: Some("claude-test".to_string()),
                    created: None,
                    provider: "anthropic".to_string(),
                    request_id: None,
                    headers: None,
                },
            })
            .expect("serialize start"),
    );

    for delta in ["I ", "am ", "thinking"] {
        bytes.extend_from_slice(
            &converter
                .serialize_event(&ChatStreamEvent::ThinkingDelta {
                    delta: delta.to_string(),
                })
                .expect("serialize thinking delta"),
        );
    }

    bytes.extend_from_slice(
        &converter
            .serialize_event(&ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("msg_thinking".to_string()),
                    model: Some("claude-test".to_string()),
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
            .expect("serialize end"),
    );

    let frames = parse_sse_frames(&bytes);
    let thinking_starts: Vec<&SseFrame> = frames
        .iter()
        .filter(|frame| {
            frame.event.as_deref() == Some("content_block_start")
                && frame.data["content_block"]["type"] == "thinking"
        })
        .collect();
    assert_eq!(thinking_starts.len(), 1, "expected a single thinking start");

    let thinking_index = thinking_starts[0].data["index"]
        .as_u64()
        .expect("thinking index");

    let thinking_deltas: Vec<&SseFrame> = frames
        .iter()
        .filter(|frame| {
            frame.event.as_deref() == Some("content_block_delta")
                && frame.data["index"].as_u64() == Some(thinking_index)
                && frame.data["delta"]["type"] == "thinking_delta"
        })
        .collect();
    assert_eq!(thinking_deltas.len(), 3, "expected all thinking deltas");

    let thinking_stops: Vec<&SseFrame> = frames
        .iter()
        .filter(|frame| {
            frame.event.as_deref() == Some("content_block_stop")
                && frame.data["index"].as_u64() == Some(thinking_index)
        })
        .collect();
    assert_eq!(thinking_stops.len(), 1, "expected a single thinking stop");
}

#[test]
fn stream_end_clears_open_block_state_before_next_end() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_first".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize first start");
    let _ = converter
        .serialize_event(&ChatStreamEvent::ContentDelta {
            delta: "hello".to_string(),
            index: None,
        })
        .expect("serialize first delta");
    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: Some("msg_first".to_string()),
                model: Some("claude-test".to_string()),
                content: MessageContent::Text("hello".to_string()),
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
        .expect("serialize first end");

    let second_end = converter
        .serialize_event(&ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: Some("msg_second".to_string()),
                model: Some("claude-test".to_string()),
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
        .expect("serialize second end");

    let frames = parse_sse_frames(&second_end);
    assert!(
        frames
            .iter()
            .all(|frame| frame.event.as_deref() != Some("content_block_stop")),
        "expected no stale content_block_stop frames after stream reset: {frames:?}"
    );
    assert!(
        frames
            .iter()
            .any(|frame| frame.event.as_deref() == Some("message_delta")),
        "expected the second end to remain serializable"
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
                headers: None,
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
fn serializes_runtime_part_to_anthropic_sse_without_custom_wrapper() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta {
                id: "0".to_string(),
                delta: "Hello".to_string(),
                provider_metadata: None,
            },
        })
        .expect("serialize runtime text-delta part");
    let frames = parse_sse_json_frames(&bytes);
    assert!(
        frames.iter().any(|v| v["type"] == "content_block_delta"),
        "expected content_block_delta from runtime part: {frames:?}"
    );
}

#[test]
fn serializes_runtime_compaction_text_parts_to_anthropic_sse() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let provider_metadata = Some(std::collections::HashMap::from([(
        "anthropic".to_string(),
        serde_json::json!({
            "contentBlockIndex": 0,
            "type": "compaction",
        }),
    )]));

    let start_bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::TextStart {
                id: "0".to_string(),
                provider_metadata: provider_metadata.clone(),
            },
        })
        .expect("serialize compaction text-start part");
    let delta_bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta {
                id: "0".to_string(),
                delta: "Condensed".to_string(),
                provider_metadata: provider_metadata.clone(),
            },
        })
        .expect("serialize compaction text-delta part");
    let end_bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::TextEnd {
                id: "0".to_string(),
                provider_metadata,
            },
        })
        .expect("serialize compaction text-end part");

    let start_frames = parse_sse_json_frames(&start_bytes);
    assert!(start_frames.iter().any(|v| {
        v["type"] == "content_block_start"
            && v["index"] == serde_json::json!(0)
            && v["content_block"]["type"] == serde_json::json!("compaction")
    }));

    let delta_frames = parse_sse_json_frames(&delta_bytes);
    assert!(delta_frames.iter().any(|v| {
        v["type"] == "content_block_delta"
            && v["index"] == serde_json::json!(0)
            && v["delta"]["type"] == serde_json::json!("compaction_delta")
            && v["delta"]["content"] == serde_json::json!("Condensed")
    }));

    let end_frames = parse_sse_json_frames(&end_bytes);
    assert!(
        end_frames
            .iter()
            .any(|v| { v["type"] == "content_block_stop" && v["index"] == serde_json::json!(0) })
    );
}

#[test]
fn serializes_runtime_tool_input_parts_to_anthropic_sse() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let start_bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputStart {
                id: "call_1".to_string(),
                tool_name: "get_weather".to_string(),
                provider_metadata: None,
                provider_executed: None,
                dynamic: None,
                title: None,
            },
        })
        .expect("serialize runtime tool-input-start part");
    let delta_bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputDelta {
                id: "call_1".to_string(),
                delta: r#"{"city":"Tokyo"}"#.to_string(),
                provider_metadata: None,
            },
        })
        .expect("serialize runtime tool-input-delta part");
    let end_bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputEnd {
                id: "call_1".to_string(),
                provider_metadata: None,
            },
        })
        .expect("serialize runtime tool-input-end part");

    let start_frames = parse_sse_json_frames(&start_bytes);
    assert!(start_frames.iter().any(|v| {
        v["type"] == "content_block_start"
            && v["content_block"]["type"] == "tool_use"
            && v["content_block"]["id"] == "call_1"
            && v["content_block"]["name"] == "get_weather"
    }));

    let delta_frames = parse_sse_json_frames(&delta_bytes);
    assert!(delta_frames.iter().any(|v| {
        v["type"] == "content_block_delta"
            && v["delta"]["type"] == "input_json_delta"
            && v["delta"]["partial_json"] == serde_json::json!(r#"{"city":"Tokyo"}"#)
    }));

    let end_frames = parse_sse_json_frames(&end_bytes);
    assert!(
        end_frames
            .iter()
            .any(|v| v["type"] == serde_json::json!("content_block_stop"))
    );
}

#[test]
fn serializes_runtime_tool_call_part_with_caller_metadata_to_anthropic_sse() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                tool_call_id: "toolu_1".to_string(),
                tool_name: "code_execution".to_string(),
                input: r#"{"code":"print(1)"}"#.to_string(),
                provider_executed: None,
                dynamic: None,
                provider_metadata: Some(std::collections::HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "contentBlockIndex": 0,
                        "caller": {
                            "type": "server_tool_use",
                            "toolId": "srvtoolu_1"
                        }
                    }),
                )])),
            }),
        })
        .expect("serialize runtime tool-call part");

    let frames = parse_sse_json_frames(&bytes);
    assert!(frames.iter().any(|v| {
        v["type"] == "content_block_start"
            && v["index"] == serde_json::json!(0)
            && v["content_block"]["type"] == serde_json::json!("tool_use")
            && v["content_block"]["id"] == serde_json::json!("toolu_1")
            && v["content_block"]["name"] == serde_json::json!("code_execution")
            && v["content_block"]["caller"]["type"] == serde_json::json!("server_tool_use")
            && v["content_block"]["caller"]["tool_id"] == serde_json::json!("srvtoolu_1")
    }));
    assert!(
        frames
            .iter()
            .any(|v| { v["type"] == "content_block_stop" && v["index"] == serde_json::json!(0) })
    );
}

#[test]
fn serializes_runtime_reasoning_delta_part_to_anthropic_sse() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ReasoningDelta {
                id: "0".to_string(),
                delta: "think".to_string(),
                provider_metadata: None,
            },
        })
        .expect("serialize runtime reasoning-delta part");
    let frames = parse_sse_json_frames(&bytes);
    assert!(frames.iter().any(|v| {
        v["type"] == "content_block_start"
            && v["content_block"]["type"] == serde_json::json!("thinking")
    }));
    assert!(frames.iter().any(|v| {
        v["type"] == "content_block_delta"
            && v["delta"]["type"] == serde_json::json!("thinking_delta")
            && v["delta"]["thinking"] == serde_json::json!("think")
    }));
}

#[test]
fn serializes_runtime_reasoning_start_with_redacted_data_to_anthropic_sse() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let start_bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ReasoningStart {
                id: "0".to_string(),
                provider_metadata: Some(std::collections::HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "contentBlockIndex": 0,
                        "redactedData": "abc123",
                    }),
                )])),
            },
        })
        .expect("serialize runtime reasoning-start part");
    let end_bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ReasoningEnd {
                id: "0".to_string(),
                provider_metadata: Some(std::collections::HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "contentBlockIndex": 0,
                    }),
                )])),
            },
        })
        .expect("serialize runtime reasoning-end part");

    let start_frames = parse_sse_json_frames(&start_bytes);
    assert!(start_frames.iter().any(|v| {
        v["type"] == "content_block_start"
            && v["index"] == serde_json::json!(0)
            && v["content_block"]["type"] == serde_json::json!("redacted_thinking")
            && v["content_block"]["data"] == serde_json::json!("abc123")
    }));

    let end_frames = parse_sse_json_frames(&end_bytes);
    assert!(
        end_frames
            .iter()
            .any(|v| { v["type"] == "content_block_stop" && v["index"] == serde_json::json!(0) })
    );
}

#[test]
fn serializes_runtime_reasoning_signature_part_to_anthropic_sse() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let _ = converter
        .serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
            },
        })
        .expect("serialize start");

    let _ = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ReasoningStart {
                id: "0".to_string(),
                provider_metadata: Some(std::collections::HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "contentBlockIndex": 0,
                    }),
                )])),
            },
        })
        .expect("serialize reasoning start");

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ReasoningDelta {
                id: "0".to_string(),
                delta: String::new(),
                provider_metadata: Some(std::collections::HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "contentBlockIndex": 0,
                        "signature": "sig-1",
                    }),
                )])),
            },
        })
        .expect("serialize runtime reasoning signature part");
    let frames = parse_sse_json_frames(&bytes);
    assert!(frames.iter().any(|v| {
        v["type"] == "content_block_delta"
            && v["index"] == serde_json::json!(0)
            && v["delta"]["type"] == serde_json::json!("signature_delta")
            && v["delta"]["signature"] == serde_json::json!("sig-1")
    }));
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
fn serializes_stream_end_replays_extended_usage_fields_from_provider_metadata() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                content: MessageContent::Text(String::new()),
                usage: Some(
                    Usage::builder()
                        .prompt_tokens(17)
                        .completion_tokens(1)
                        .total_tokens(18)
                        .with_cached_tokens(5)
                        .build(),
                ),
                finish_reason: Some(FinishReason::Stop),
                raw_finish_reason: None,
                audio: None,
                system_fingerprint: None,
                service_tier: Some("standard".to_string()),
                warnings: None,
                provider_metadata: Some(std::collections::HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "usage": {
                            "input_tokens": 17,
                            "output_tokens": 1,
                            "cache_creation_input_tokens": 10,
                            "cache_read_input_tokens": 5,
                            "service_tier": "standard",
                            "server_tool_use": {
                                "web_search_requests": 2
                            }
                        },
                        "cacheCreationInputTokens": 10
                    }),
                )])),
            },
        })
        .expect("serialize stream end");

    let frames = parse_sse_json_frames(&bytes);
    let message_delta = frames
        .iter()
        .find(|frame| frame["type"] == "message_delta")
        .expect("message_delta frame");

    assert_eq!(
        message_delta["usage"]["cache_creation_input_tokens"],
        serde_json::json!(10)
    );
    assert_eq!(
        message_delta["usage"]["cache_read_input_tokens"],
        serde_json::json!(5)
    );
    assert_eq!(
        message_delta["usage"]["service_tier"],
        serde_json::json!("standard")
    );
    assert_eq!(
        message_delta["usage"]["server_tool_use"]["web_search_requests"],
        serde_json::json!(2)
    );
}

#[test]
fn serializes_stream_end_does_not_synthesize_zero_usage_totals_when_unknown() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: Some("msg_test".to_string()),
                model: Some("claude-test".to_string()),
                content: MessageContent::Text(String::new()),
                usage: Some(Usage::unknown()),
                finish_reason: Some(FinishReason::Stop),
                raw_finish_reason: None,
                audio: None,
                system_fingerprint: None,
                service_tier: Some("standard".to_string()),
                warnings: None,
                provider_metadata: None,
            },
        })
        .expect("serialize stream end");

    let frames = parse_sse_json_frames(&bytes);
    let message_delta = frames
        .iter()
        .find(|frame| frame["type"] == "message_delta")
        .expect("message_delta frame");
    let usage = message_delta["usage"].as_object().expect("usage object");

    assert!(!usage.contains_key("input_tokens"));
    assert!(!usage.contains_key("output_tokens"));
    assert_eq!(
        message_delta["usage"]["service_tier"],
        serde_json::json!("standard")
    );
}

#[test]
fn serializes_v3_finish_replays_raw_usage_and_context_management() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "anthropic:finish".to_string(),
            data: serde_json::json!({
                "type": "finish",
                "finishReason": { "unified": "stop" },
                "providerMetadata": {
                    "anthropic": {
                        "usage": {
                            "input_tokens": 17,
                            "output_tokens": 1,
                            "cache_creation_input_tokens": 10,
                            "cache_read_input_tokens": 5,
                            "service_tier": "standard",
                            "server_tool_use": {
                                "web_search_requests": 2
                            }
                        },
                        "stopSequence": "done",
                        "contextManagement": {
                            "appliedEdits": [
                                {
                                    "type": "clear_tool_uses_20250919",
                                    "clearedToolUses": 3,
                                    "clearedInputTokens": 1000
                                }
                            ]
                        }
                    }
                },
                "usage": {
                    "inputTokens": {
                        "total": 32,
                        "noCache": 17,
                        "cacheRead": 5,
                        "cacheWrite": 10
                    },
                    "outputTokens": {
                        "total": 1
                    }
                }
            }),
        })
        .expect("serialize v3 finish");

    let frames = parse_sse_json_frames(&bytes);
    let message_delta = frames
        .iter()
        .find(|frame| frame["type"] == "message_delta")
        .expect("message_delta frame");

    assert_eq!(
        message_delta["delta"]["stop_sequence"],
        serde_json::json!("done")
    );
    assert_eq!(
        message_delta["usage"]["cache_creation_input_tokens"],
        serde_json::json!(10)
    );
    assert_eq!(
        message_delta["usage"]["cache_read_input_tokens"],
        serde_json::json!(5)
    );
    assert_eq!(
        message_delta["usage"]["server_tool_use"]["web_search_requests"],
        serde_json::json!(2)
    );
    assert_eq!(
        message_delta["context_management"]["applied_edits"][0]["cleared_tool_uses"],
        serde_json::json!(3)
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
                headers: None,
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

#[test]
fn serializes_v3_tool_approval_request_part_with_replay_as_text_when_configured() {
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
                headers: None,
            },
        })
        .expect("serialize start");

    let bytes = converter
        .serialize_event(&ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolApprovalRequest(
                crate::types::ChatStreamToolApprovalRequest {
                    approval_id: "mcpr_1".to_string(),
                    tool_call_id: "call_1".to_string(),
                    provider_metadata: None,
                },
            ),
            replay: crate::types::ChatStreamReplay::default(),
        })
        .expect("serialize v3 tool approval request");

    let frames = parse_sse_json_frames(&bytes);
    assert!(
        frames.iter().any(|v| v["type"] == "content_block_delta"
            && v["delta"]["type"] == "text_delta"
            && v["delta"]["text"]
                .as_str()
                .is_some_and(|s| s.contains("[tool-approval-request]"))),
        "expected text_delta containing [tool-approval-request]: {frames:?}"
    );
}

#[test]
fn serializes_provider_tool_result_without_stream_start_with_message_start() {
    let converter = AnthropicEventConverter::new(create_test_config());

    let bytes = converter
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "anthropic:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "providerExecuted": true,
                "toolCallId": "srvtoolu_1",
                "toolName": "web_search",
                "result": [{ "type": "web_search_result", "title": "Rust", "url": "https://www.rust-lang.org" }]
            }),
        })
        .expect("serialize provider tool-result");

    let frames = parse_sse_json_frames(&bytes);
    assert_eq!(
        frames
            .first()
            .and_then(|value| value.get("type"))
            .and_then(|value| value.as_str()),
        Some("message_start"),
        "expected message_start first for provider custom event: {frames:?}"
    );
    assert!(
        frames.iter().any(|value| {
            value["type"] == "content_block_start"
                && value["content_block"]["type"] == "web_search_tool_result"
                && value["content_block"]["tool_use_id"] == "srvtoolu_1"
        }),
        "expected web_search_tool_result content_block_start: {frames:?}"
    );
    assert!(
        frames
            .iter()
            .any(|value| value["type"] == "content_block_stop"),
        "expected content_block_stop for provider custom event: {frames:?}"
    );
}
