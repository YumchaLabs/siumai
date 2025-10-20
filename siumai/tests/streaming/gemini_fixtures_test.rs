//! Gemini streaming fixtures tests

use siumai::providers::gemini::streaming::GeminiEventConverter;
use siumai::providers::gemini::types::GeminiConfig;
use siumai::stream::ChatStreamEvent;

#[path = "../support/stream_fixture.rs"]
mod support;

#[tokio::test]
async fn gemini_simple_text_then_finish_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/gemini/simple_text_then_finish.sse",
    )
    .expect("load fixture");

    let cfg = GeminiConfig::default();
    let converter = GeminiEventConverter::new(cfg);
    let events = support::collect_sse_events(bytes, converter).await;

    let mut content = String::new();
    let mut saw_end = false;
    for e in events {
        match e {
            ChatStreamEvent::ContentDelta { delta, .. } => content.push_str(&delta),
            ChatStreamEvent::StreamEnd { .. } => saw_end = true,
            _ => {}
        }
    }
    assert_eq!(content, "Hello world");
    assert!(saw_end, "expect stream end on finishReason");
}

#[tokio::test]
async fn gemini_thought_then_text_stop_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/gemini/thought_then_text_stop.sse",
    )
    .expect("load fixture");

    let cfg = GeminiConfig::default();
    let converter = GeminiEventConverter::new(cfg);
    let events = support::collect_sse_events(bytes, converter).await;

    let mut thought = String::new();
    let mut content = String::new();
    let mut saw_end = false;
    let mut saw_reasoning_tokens = false;
    for e in events {
        match e {
            ChatStreamEvent::ThinkingDelta { delta } => thought.push_str(&delta),
            ChatStreamEvent::ContentDelta { delta, .. } => content.push_str(&delta),
            ChatStreamEvent::UsageUpdate { usage } => {
                if usage.reasoning_tokens.is_some() { saw_reasoning_tokens = true; }
            }
            ChatStreamEvent::StreamEnd { .. } => saw_end = true,
            _ => {}
        }
    }
    assert!(thought.contains("Thinking"));
    assert!(content.contains("Final answer."));
    assert!(saw_reasoning_tokens, "expect reasoning tokens in usage update");
    assert!(saw_end);
}

#[tokio::test]
async fn gemini_max_tokens_finish_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/gemini/max_tokens_finish.sse",
    )
    .expect("load fixture");

    let cfg = GeminiConfig::default();
    let converter = GeminiEventConverter::new(cfg);
    let events = support::collect_sse_events(bytes, converter).await;

    assert!(events.iter().any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. })), "expect StreamEnd on MAX_TOKENS");
}

#[tokio::test]
async fn gemini_multi_candidates_non_first_fixture() {
    // Pending: support extracting from non-first candidate
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/gemini/multi_candidates_non_first.sse",
    )
    .expect("load fixture");
    let cfg = GeminiConfig::default();
    let converter = GeminiEventConverter::new(cfg);
    let events = support::collect_sse_events(bytes, converter).await;
    // Expect content from the second candidate is emitted once we scan candidates
    assert!(events.iter().any(|e| matches!(e, ChatStreamEvent::ContentDelta { delta, .. } if delta.contains("Visible from second"))));
}

#[tokio::test]
async fn gemini_safety_finish_maps_to_content_filter() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/gemini/safety_finish.sse",
    )
    .expect("load fixture");
    let cfg = GeminiConfig::default();
    let converter = GeminiEventConverter::new(cfg);
    let events = support::collect_sse_events(bytes, converter).await;
    let end = events.into_iter().find_map(|e| match e {
        ChatStreamEvent::StreamEnd { response } => Some(response),
        _ => None,
    }).expect("expect stream end");
    assert_eq!(end.finish_reason, Some(siumai::types::FinishReason::ContentFilter));
}

#[tokio::test]
async fn gemini_recitation_finish_maps_to_content_filter() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/gemini/recitation_finish.sse",
    )
    .expect("load fixture");
    let cfg = GeminiConfig::default();
    let converter = GeminiEventConverter::new(cfg);
    let events = support::collect_sse_events(bytes, converter).await;
    let end = events.into_iter().find_map(|e| match e {
        ChatStreamEvent::StreamEnd { response } => Some(response),
        _ => None,
    }).expect("expect stream end");
    assert_eq!(end.finish_reason, Some(siumai::types::FinishReason::ContentFilter));
}
