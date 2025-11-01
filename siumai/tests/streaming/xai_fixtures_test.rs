//! xAI streaming fixtures tests
//!
//! These tests verify xAI streaming responses using real API response fixtures.
//! All fixtures are based on official xAI API documentation:
//! https://docs.x.ai/docs

use siumai::providers::xai::streaming::XaiEventConverter;
use siumai::streaming::ChatStreamEvent;

use crate::support;

fn make_xai_converter() -> XaiEventConverter {
    XaiEventConverter::new()
}

#[tokio::test]
async fn xai_simple_content_stop_fixture() {
    let bytes = support::load_sse_fixture_as_bytes("tests/fixtures/xai/simple_content_stop.sse")
        .expect("load fixture");

    let converter = make_xai_converter();
    let events = support::collect_sse_events(bytes, converter).await;

    let mut content = String::new();
    let mut saw_start = false;
    let mut saw_end = false;
    let mut usage_total = 0u32;

    for e in events {
        match e {
            ChatStreamEvent::StreamStart { metadata } => {
                saw_start = true;
                assert_eq!(metadata.model, Some("grok-beta".to_string()));
                assert_eq!(metadata.provider, "xai");
                assert_eq!(metadata.id, Some("chatcmpl-123".to_string()));
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
    assert_eq!(content, "Hello from Grok");
    assert_eq!(usage_total, 18); // 10 + 8
    assert!(saw_end, "expect stream end");
}
