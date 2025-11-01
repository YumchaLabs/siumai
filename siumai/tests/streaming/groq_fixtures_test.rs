//! Groq streaming fixtures tests
//!
//! These tests verify Groq streaming responses using real API response fixtures.
//! All fixtures are based on official Groq API documentation:
//! https://console.groq.com/docs/api-reference

use siumai::providers::groq::streaming::GroqEventConverter;
use siumai::streaming::ChatStreamEvent;

use crate::support;

fn make_groq_converter() -> GroqEventConverter {
    GroqEventConverter::new()
}

#[tokio::test]
async fn groq_simple_content_stop_fixture() {
    let bytes = support::load_sse_fixture_as_bytes("tests/fixtures/groq/simple_content_stop.sse")
        .expect("load fixture");

    let converter = make_groq_converter();
    let events = support::collect_sse_events(bytes, converter).await;

    let mut content = String::new();
    let mut saw_start = false;
    let mut saw_end = false;
    let mut usage_total = 0u32;

    for e in events {
        match e {
            ChatStreamEvent::StreamStart { metadata } => {
                saw_start = true;
                assert_eq!(metadata.model, Some("llama-3.3-70b-versatile".to_string()));
                assert_eq!(metadata.provider, "groq");
                assert_eq!(metadata.id, Some("chatcmpl-abc123".to_string()));
            }
            ChatStreamEvent::ContentDelta { delta, .. } => content.push_str(&delta),
            ChatStreamEvent::UsageUpdate { usage } => usage_total = usage.total_tokens,
            ChatStreamEvent::StreamEnd { response } => {
                saw_end = true;
                assert_eq!(
                    response.finish_reason,
                    Some(siumai::types::FinishReason::Stop)
                );
            }
            _ => {}
        }
    }

    assert!(saw_start, "expect stream start");
    assert_eq!(content, "Hello world");
    assert_eq!(usage_total, 15); // 10 + 5
    assert!(saw_end, "expect stream end");
}
