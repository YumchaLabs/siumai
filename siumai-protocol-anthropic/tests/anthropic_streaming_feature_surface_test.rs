#![cfg(feature = "anthropic-standard")]

use eventsource_stream::Event;
use siumai_protocol_anthropic::ChatResponse;
use siumai_protocol_anthropic::provider_metadata::anthropic::AnthropicChatResponseExt;
use siumai_protocol_anthropic::standards::anthropic::params::AnthropicParams;
use siumai_protocol_anthropic::standards::anthropic::streaming::AnthropicEventConverter;
use siumai_protocol_anthropic::streaming::{
    ChatStreamEvent, LanguageModelV3StreamPart, SseEventConverter,
};
use siumai_protocol_anthropic::types::{
    ChatStreamPart, ChatStreamToolCall, ChatStreamToolResult, FinishReason, MessageContent,
    ResponseMetadata,
};

fn create_test_config() -> AnthropicParams {
    AnthropicParams::default()
}

fn stream_part(event: &ChatStreamEvent) -> Option<LanguageModelV3StreamPart> {
    LanguageModelV3StreamPart::try_from_chat_event(event)
}

fn anthropic_provider_metadata(
    value: serde_json::Value,
) -> std::collections::HashMap<String, serde_json::Value> {
    std::collections::HashMap::from([("anthropic".to_string(), value)])
}

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
    converter: &AnthropicEventConverter,
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

    events
}

#[tokio::test]
async fn anthropic_public_feature_surface_roundtrips_provider_tool_stream_parts() {
    let encoder = AnthropicEventConverter::new(create_test_config());

    let tool_call_bytes = encoder
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                tool_call_id: "srvtoolu_1".to_string(),
                tool_name: "web_search".to_string(),
                input: r#"{"query":"rust"}"#.to_string(),
                provider_executed: Some(true),
                dynamic: None,
                provider_metadata: Some(anthropic_provider_metadata(serde_json::json!({
                    "contentBlockIndex": 0
                }))),
            }),
        })
        .expect("serialize provider tool-call part");
    let tool_result_bytes = encoder
        .serialize_event(&ChatStreamEvent::Part {
            part: ChatStreamPart::ToolResult(ChatStreamToolResult {
                tool_call_id: "srvtoolu_1".to_string(),
                tool_name: "web_search".to_string(),
                result: serde_json::json!([{
                    "type": "web_search_result",
                    "title": "Rust",
                    "url": "https://www.rust-lang.org",
                    "encryptedContent": "..."
                }]),
                is_error: Some(false),
                preliminary: None,
                dynamic: None,
                provider_metadata: Some(anthropic_provider_metadata(serde_json::json!({
                    "contentBlockIndex": 1
                }))),
            }),
        })
        .expect("serialize provider tool-result part");

    let call_frames = parse_sse_json_frames(&tool_call_bytes);
    let result_frames = parse_sse_json_frames(&tool_result_bytes);

    assert_eq!(call_frames[0]["type"], serde_json::json!("message_start"));
    assert!(call_frames.iter().any(|frame| {
        frame["type"] == serde_json::json!("content_block_start")
            && frame["content_block"]["type"] == serde_json::json!("server_tool_use")
            && frame["content_block"]["id"] == serde_json::json!("srvtoolu_1")
            && frame["content_block"]["name"] == serde_json::json!("web_search")
    }));
    assert!(
        result_frames
            .iter()
            .all(|frame| { frame["type"] != serde_json::json!("message_start") })
    );
    assert!(result_frames.iter().any(|frame| {
        frame["type"] == serde_json::json!("content_block_start")
            && frame["content_block"]["type"] == serde_json::json!("web_search_tool_result")
            && frame["content_block"]["tool_use_id"] == serde_json::json!("srvtoolu_1")
    }));

    let decoder = AnthropicEventConverter::new(create_test_config());
    let events = decode_frames(
        &decoder,
        &call_frames
            .into_iter()
            .chain(result_frames.into_iter())
            .collect::<Vec<_>>(),
    )
    .await;

    let provider_tool_calls = events
        .iter()
        .filter(|event| {
            matches!(
                stream_part(event),
                Some(LanguageModelV3StreamPart::ToolCall(ref call))
                    if call.tool_name == "web_search"
                        && call.provider_executed == Some(true)
            )
        })
        .count();
    let provider_tool_results = events
        .iter()
        .filter(|event| {
            matches!(
                stream_part(event),
                Some(LanguageModelV3StreamPart::ToolResult(ref result))
                    if result.tool_name == "web_search"
                        && result.is_error == Some(false)
            )
        })
        .count();
    let sources = events
        .iter()
        .filter(|event| {
            matches!(
                stream_part(event),
                Some(LanguageModelV3StreamPart::Source(
                    siumai_protocol_anthropic::streaming::LanguageModelV3Source::Url { url, .. }
                )) if url == "https://www.rust-lang.org"
            )
        })
        .count();

    assert_eq!(provider_tool_calls, 1);
    assert_eq!(provider_tool_results, 1);
    assert_eq!(sources, 1);
}

#[tokio::test]
async fn anthropic_public_feature_surface_preserves_thinking_signature_and_single_block() {
    let decoder = AnthropicEventConverter::new(create_test_config());

    let _ = decoder
        .convert_event(Event {
            event: String::new(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#
                .to_string(),
            id: "0".to_string(),
            retry: None,
        })
        .await;
    let signature_events = decoder
        .convert_event(Event {
            event: String::new(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}"#
                .to_string(),
            id: "1".to_string(),
            retry: None,
        })
        .await;

    assert!(signature_events.iter().any(|event| matches!(
        stream_part(event.as_ref().expect("decode signature frame")),
        Some(LanguageModelV3StreamPart::ReasoningDelta {
            ref id,
            ref delta,
            provider_metadata: Some(ref provider_metadata),
        }) if id == "0"
            && delta.is_empty()
            && provider_metadata
                .get("anthropic")
                .and_then(|meta| meta.get("signature"))
                == Some(&serde_json::json!("sig-1"))
    )));

    let stop_events = decoder
        .convert_event(Event {
            event: String::new(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "2".to_string(),
            retry: None,
        })
        .await;
    let stream_end = stop_events
        .into_iter()
        .find_map(|event| match event.expect("decode stop frame") {
            ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("stream end");

    let metadata = stream_end
        .anthropic_metadata()
        .expect("anthropic metadata on stream end");
    assert_eq!(metadata.thinking_signature.as_deref(), Some("sig-1"));

    let encoder = AnthropicEventConverter::new(create_test_config());
    let mut bytes = Vec::new();
    bytes.extend_from_slice(
        &encoder
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
            &encoder
                .serialize_event(&ChatStreamEvent::ThinkingDelta {
                    delta: delta.to_string(),
                })
                .expect("serialize thinking delta"),
        );
    }
    bytes.extend_from_slice(
        &encoder
            .serialize_event(&ChatStreamEvent::Part {
                part: ChatStreamPart::ReasoningStart {
                    id: "0".to_string(),
                    provider_metadata: Some(anthropic_provider_metadata(serde_json::json!({
                        "contentBlockIndex": 0
                    }))),
                },
            })
            .expect("serialize reasoning-start"),
    );
    bytes.extend_from_slice(
        &encoder
            .serialize_event(&ChatStreamEvent::Part {
                part: ChatStreamPart::ReasoningDelta {
                    id: "0".to_string(),
                    delta: String::new(),
                    provider_metadata: Some(anthropic_provider_metadata(serde_json::json!({
                        "contentBlockIndex": 0,
                        "signature": "sig-1"
                    }))),
                },
            })
            .expect("serialize reasoning signature"),
    );
    bytes.extend_from_slice(
        &encoder
            .serialize_event(&ChatStreamEvent::Part {
                part: ChatStreamPart::ReasoningEnd {
                    id: "0".to_string(),
                    provider_metadata: Some(anthropic_provider_metadata(serde_json::json!({
                        "contentBlockIndex": 0
                    }))),
                },
            })
            .expect("serialize reasoning-end"),
    );
    bytes.extend_from_slice(
        &encoder
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
                    response: None,
                    provider_metadata: None,
                },
            })
            .expect("serialize end"),
    );

    let frames = parse_sse_json_frames(&bytes);
    let thinking_starts = frames
        .iter()
        .filter(|frame| {
            frame["type"] == serde_json::json!("content_block_start")
                && frame["content_block"]["type"] == serde_json::json!("thinking")
        })
        .count();
    let thinking_deltas = frames
        .iter()
        .filter(|frame| {
            frame["type"] == serde_json::json!("content_block_delta")
                && frame["delta"]["type"] == serde_json::json!("thinking_delta")
        })
        .count();
    let thinking_stops = frames
        .iter()
        .filter(|frame| frame["type"] == serde_json::json!("content_block_stop"))
        .count();
    let signature_deltas = frames
        .iter()
        .filter(|frame| {
            frame["type"] == serde_json::json!("content_block_delta")
                && frame["delta"]["type"] == serde_json::json!("signature_delta")
                && frame["delta"]["signature"] == serde_json::json!("sig-1")
        })
        .count();

    assert_eq!(thinking_starts, 1);
    assert_eq!(thinking_deltas, 3);
    assert_eq!(thinking_stops, 1);
    assert_eq!(signature_deltas, 1);
}
