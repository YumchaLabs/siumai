//! Anthropic streaming fixtures tests

use siumai::providers::anthropic::streaming::AnthropicEventConverter;
use siumai::stream::ChatStreamEvent;
use siumai::params::AnthropicParams;

#[path = "../support/stream_fixture.rs"]
mod support;

fn make_anthropic_converter() -> AnthropicEventConverter {
    let cfg = AnthropicParams::default();
    AnthropicEventConverter::new(cfg)
}

#[tokio::test]
async fn anthropic_message_start_deltas_stop_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/anthropic/message_start_deltas_stop.sse",
    )
    .expect("load fixture");

    let converter = make_anthropic_converter();
    let events = support::collect_sse_events(bytes, converter).await;

    let mut content = String::new();
    let mut saw_start = false;
    let mut saw_end = false;
    let mut usage_total = 0u32;
    for e in events {
        match e {
            ChatStreamEvent::StreamStart { .. } => saw_start = true,
            ChatStreamEvent::ContentDelta { delta, .. } => content.push_str(&delta),
            ChatStreamEvent::UsageUpdate { usage } => usage_total = usage.total_tokens,
            ChatStreamEvent::StreamEnd { .. } => saw_end = true,
            _ => {}
        }
    }
    assert!(saw_start, "expect stream start");
    assert_eq!(content, "Hello world");
    assert_eq!(usage_total, 12);
    assert!(saw_end, "expect stream end");
}

#[tokio::test]
async fn anthropic_thinking_and_text_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/anthropic/thinking_and_text.sse",
    )
    .expect("load fixture");

    let converter = make_anthropic_converter();
    let events = support::collect_sse_events(bytes, converter).await;

    let mut thinking = String::new();
    let mut content = String::new();
    let mut saw_end = false;
    for e in events {
        match e {
            ChatStreamEvent::ThinkingDelta { delta } => thinking.push_str(&delta),
            ChatStreamEvent::ContentDelta { delta, .. } => content.push_str(&delta),
            ChatStreamEvent::StreamEnd { .. } => saw_end = true,
            _ => {}
        }
    }
    assert!(thinking.contains("Reasoning"));
    assert_eq!(content, "Answer: 42");
    assert!(saw_end);
}

#[tokio::test]
async fn anthropic_error_event_fixture() {
    // Pending: convert Anthropic error JSON into ChatStreamEvent::Error instead of ParseError
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/anthropic/error_event.sse",
    )
    .expect("load fixture");
    let converter = make_anthropic_converter();
    let events = support::collect_sse_events(bytes, converter).await;
    assert!(events.iter().any(|e| matches!(e, ChatStreamEvent::Error { error } if error.contains("Invalid auth"))));
}

#[tokio::test]
async fn anthropic_partial_without_message_stop_fixture() {
    // Expect content deltas but no StreamEnd because message_stop/delta stop reason is missing.
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/anthropic/partial_without_message_stop.sse",
    )
    .expect("load fixture");

    let converter = make_anthropic_converter();
    let events = support::collect_sse_events(bytes, converter).await;
    assert!(events.iter().any(|e| matches!(e, ChatStreamEvent::ContentDelta { .. })), "expect content delta");
    assert!(!events.iter().any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. })), "no StreamEnd expected");
}

#[tokio::test]
async fn anthropic_refusal_stop_reason_maps_to_content_filter() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/anthropic/refusal_stop_reason.sse",
    )
    .expect("load fixture");

    let converter = make_anthropic_converter();
    let events = support::collect_sse_events(bytes, converter).await;
    let end = events.into_iter().find_map(|e| match e {
        ChatStreamEvent::StreamEnd { response } => Some(response),
        _ => None,
    }).expect("expect stream end");
    assert_eq!(end.finish_reason, Some(siumai::types::FinishReason::ContentFilter));
}
