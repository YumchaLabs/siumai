#![cfg(feature = "anthropic")]

use eventsource_stream::Event;
use siumai::prelude::unified::{
    ChatResponse, ChatStreamEvent, FinishReason, MessageContent, SseEventConverter, Usage,
};
use siumai::provider_ext::anthropic::AnthropicChatResponseExt;
use std::collections::HashMap;

fn parse_sse_json_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|chunk| {
            let line = chunk
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .map(str::trim)?;
            if line.is_empty() || line == "[DONE]" {
                return None;
            }
            serde_json::from_str::<serde_json::Value>(line).ok()
        })
        .collect()
}

async fn decode_frames(
    converter: &siumai::protocol::anthropic::streaming::AnthropicEventConverter,
    frames: &[serde_json::Value],
) -> Vec<ChatStreamEvent> {
    let mut events = Vec::new();

    for (index, frame) in frames.iter().enumerate() {
        let out = converter
            .convert_event(Event {
                event: String::new(),
                data: serde_json::to_string(frame).expect("serialize frame"),
                id: index.to_string(),
                retry: None,
            })
            .await;

        for item in out {
            events.push(item.expect("decode stream frame"));
        }
    }

    while let Some(item) = converter.handle_stream_end() {
        events.push(item.expect("finalize stream"));
    }

    events
}

#[tokio::test]
async fn anthropic_public_roundtrip_preserves_extended_usage_fields() {
    let converter = siumai::protocol::anthropic::streaming::AnthropicEventConverter::new(
        siumai::protocol::anthropic::params::AnthropicParams::default(),
    );

    let response = ChatResponse {
        id: Some("msg_issue_17".to_string()),
        model: Some("claude-sonnet-4-5".to_string()),
        content: MessageContent::Text(String::new()),
        usage: Some(
            Usage::builder()
                .prompt_tokens(17)
                .completion_tokens(1)
                .total_tokens(18)
                .with_input_cache_read_tokens(5)
                .build(),
        ),
        finish_reason: Some(FinishReason::Stop),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: Some("standard".to_string()),
        warnings: None,
        request: None,
        response: None,
        provider_metadata: Some(HashMap::from([(
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
    };

    let bytes = converter
        .serialize_event(&ChatStreamEvent::StreamEnd { response })
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

    let decoder = siumai::protocol::anthropic::streaming::AnthropicEventConverter::new(
        siumai::protocol::anthropic::params::AnthropicParams::default(),
    );
    let events = decode_frames(&decoder, &frames).await;
    let roundtripped = events
        .into_iter()
        .find_map(|event| match event {
            ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("stream end response");

    let metadata = roundtripped
        .anthropic_metadata()
        .expect("anthropic metadata");
    assert_eq!(metadata.cache_creation_input_tokens, Some(10));
    assert_eq!(metadata.cache_read_input_tokens, Some(5));
    assert_eq!(metadata.service_tier.as_deref(), Some("standard"));
    assert_eq!(
        metadata
            .server_tool_use
            .as_ref()
            .and_then(|usage| usage.web_search_requests),
        Some(2)
    );
}

#[tokio::test]
async fn anthropic_public_roundtrip_replays_raw_usage_without_provider_metadata() {
    let converter = siumai::protocol::anthropic::streaming::AnthropicEventConverter::new(
        siumai::protocol::anthropic::params::AnthropicParams::default(),
    );

    let raw_usage = serde_json::json!({
        "input_tokens": 17,
        "output_tokens": 1,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 5,
        "service_tier": "standard",
        "server_tool_use": {
            "code_execution_requests": 1,
            "future_tool_requests": 9
        }
    });

    let response = ChatResponse {
        id: Some("msg_issue_17_raw_usage".to_string()),
        model: Some("claude-sonnet-4-5".to_string()),
        content: MessageContent::Text(String::new()),
        usage: Some(
            Usage::builder()
                .prompt_tokens(17)
                .completion_tokens(1)
                .total_tokens(18)
                .with_input_cache_read_tokens(5)
                .with_raw_usage_value(raw_usage)
                .build(),
        ),
        finish_reason: Some(FinishReason::Stop),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: None,
    };

    let bytes = converter
        .serialize_event(&ChatStreamEvent::StreamEnd { response })
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
        message_delta["usage"]["server_tool_use"]["code_execution_requests"],
        serde_json::json!(1)
    );
    assert_eq!(
        message_delta["usage"]["server_tool_use"]["future_tool_requests"],
        serde_json::json!(9)
    );

    let decoder = siumai::protocol::anthropic::streaming::AnthropicEventConverter::new(
        siumai::protocol::anthropic::params::AnthropicParams::default(),
    );
    let events = decode_frames(&decoder, &frames).await;
    let roundtripped = events
        .into_iter()
        .find_map(|event| match event {
            ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("stream end response");

    assert_eq!(roundtripped.service_tier.as_deref(), Some("standard"));
    let metadata = roundtripped
        .anthropic_metadata()
        .expect("anthropic metadata");
    assert_eq!(
        metadata
            .server_tool_use
            .as_ref()
            .and_then(|usage| usage.code_execution_requests),
        Some(1)
    );
    assert_eq!(
        metadata
            .server_tool_use
            .as_ref()
            .and_then(|usage| usage.extra.get("future_tool_requests"))
            .and_then(|value| value.as_u64()),
        Some(9)
    );
}

#[test]
fn anthropic_public_message_metadata_surface_matches_narrow_typed_shape() {
    let mut response = ChatResponse {
        id: Some("msg_meta".to_string()),
        model: Some("claude-sonnet-4-5".to_string()),
        content: MessageContent::Text(String::new()),
        usage: None,
        finish_reason: Some(FinishReason::Stop),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: Some(HashMap::from([(
            "anthropic".to_string(),
            serde_json::json!({
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2
                },
                "stopSequence": null,
                "iterations": null,
                "container": null,
                "contextManagement": null
            }),
        )])),
    };

    let metadata = response
        .anthropic_message_metadata()
        .expect("anthropic message metadata");

    assert_eq!(
        serde_json::to_value(&metadata).expect("serialize message metadata"),
        serde_json::json!({
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2
            },
            "stopSequence": null,
            "iterations": null,
            "container": null,
            "contextManagement": null
        })
    );

    let wide = response.anthropic_metadata().expect("anthropic metadata");
    assert_eq!(
        wide.usage
            .as_ref()
            .and_then(|usage| usage.get("input_tokens"))
            .and_then(|value| value.as_u64()),
        Some(10)
    );
    response.provider_metadata = None;
    assert!(response.anthropic_message_metadata().is_none());
}

#[test]
fn anthropic_public_message_metadata_container_surface_is_typed() {
    let response = ChatResponse {
        id: Some("msg_meta_container".to_string()),
        model: Some("claude-sonnet-4-5".to_string()),
        content: MessageContent::Text(String::new()),
        usage: None,
        finish_reason: Some(FinishReason::Stop),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: Some(HashMap::from([(
            "anthropic".to_string(),
            serde_json::json!({
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2
                },
                "stopSequence": null,
                "iterations": null,
                "container": {
                    "id": "container_123",
                    "expiresAt": "2026-04-22T12:00:00Z",
                    "skills": [
                        {
                            "type": "anthropic",
                            "skillId": "pptx",
                            "version": "latest"
                        }
                    ]
                },
                "contextManagement": null
            }),
        )])),
    };

    let metadata = response
        .anthropic_message_metadata()
        .expect("anthropic message metadata");
    let container = metadata.container.expect("typed message container");
    assert_eq!(container.id, "container_123");
    assert_eq!(container.expires_at, "2026-04-22T12:00:00Z");
    let skills = container.skills.expect("typed message container skills");
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].kind, "anthropic");
    assert_eq!(skills[0].skill_id, "pptx");
    assert_eq!(skills[0].version, "latest");

    let wide = response.anthropic_metadata().expect("anthropic metadata");
    assert_eq!(
        wide.container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_123")
    );
}
