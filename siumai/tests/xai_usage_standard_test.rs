#![cfg(feature = "xai")]

use eventsource_stream::Event;
use siumai::execution::transformers::response::ResponseTransformer;
use siumai::providers::xai::streaming::XaiEventConverter;
use siumai::providers::xai::transformers::XaiResponseTransformer;
use siumai::streaming::SseEventConverter;
use siumai::types::ChatResponse;

#[test]
fn xai_response_usage_includes_detailed_fields() {
    let tx = XaiResponseTransformer;

    // XAI chat response with both top-level reasoning_tokens and
    // nested completion_tokens_details.reasoning_tokens. The nested
    // field should be preferred when constructing Usage.
    let raw = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_756_433_923,
        "model": "grok-beta",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello",
                "tool_calls": []
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 7,
            "prompt_tokens_details": {
                "cached_tokens": 3
            },
            "completion_tokens_details": {
                "reasoning_tokens": 11
            }
        },
        "system_fingerprint": null
    });

    let resp: ChatResponse = tx.transform_chat_response(&raw).expect("ok");
    let usage = resp.usage.expect("usage should be present");

    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 20);
    assert_eq!(usage.total_tokens, 30);

    // prompt_tokens_details.cached_tokens
    assert_eq!(
        usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens),
        Some(3)
    );

    // completion_tokens_details.reasoning_tokens should prefer nested value (11)
    let completion = usage
        .completion_tokens_details
        .as_ref()
        .expect("completion_tokens_details should be present");
    assert_eq!(completion.reasoning_tokens, Some(11));
}

#[tokio::test]
async fn xai_stream_usage_includes_detailed_fields() {
    let converter = XaiEventConverter::new();

    // Stream chunk with usage only (no content) to isolate UsageUpdate event.
    let event = Event {
        event: "message".to_string(),
        data: r#"{
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1756433923,
            "model": "grok-beta",
            "choices": [{
                "index": 0,
                "delta": { "content": "", "reasoning_content": null, "tool_calls": null },
                "finish_reason": null
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
                "reasoning_tokens": 2,
                "prompt_tokens_details": {
                    "cached_tokens": 1
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 4
                }
            }
        }"#
        .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let events = converter.convert_event(event).await;
    assert!(!events.is_empty());

    // Find the UsageUpdate event and validate its fields.
    let usage_event = events
        .into_iter()
        .find_map(|e| match e {
            Ok(siumai::streaming::ChatStreamEvent::UsageUpdate { usage }) => Some(usage),
            _ => None,
        })
        .expect("UsageUpdate event should be present");

    assert_eq!(usage_event.prompt_tokens, 5);
    assert_eq!(usage_event.completion_tokens, 7);
    assert_eq!(usage_event.total_tokens, 12);

    // prompt_tokens_details.cached_tokens from streaming
    assert_eq!(
        usage_event
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens),
        Some(1)
    );

    // completion_tokens_details.reasoning_tokens should prefer nested value (4)
    let completion = usage_event
        .completion_tokens_details
        .as_ref()
        .expect("completion_tokens_details should be present");
    assert_eq!(completion.reasoning_tokens, Some(4));
}
