//! xAI streaming fixtures tests
//!
//! These tests verify xAI streaming responses using real API response fixtures.
//! All fixtures are based on official xAI API documentation:
//! https://docs.x.ai/docs

use siumai::core::{ProviderContext, ProviderSpec};
use siumai::streaming::{ChatStreamEvent, SseEventConverter};
use siumai::types::{ChatMessage, ChatRequest, CommonParams};

use crate::support;

/// 使用运行时 XaiSpec（基于 OpenAI 标准）的 streaming transformer。
fn make_xai_converter() -> impl SseEventConverter + Clone + 'static {
    // 构造一个最小的 ChatRequest；具体内容对 streaming 形状不重要。
    let req = ChatRequest::builder()
        .messages(vec![ChatMessage::user("hello").build()])
        .common_params(CommonParams {
            model: "grok-beta".to_string(),
            ..Default::default()
        })
        .build();

    let ctx = ProviderContext::new(
        "xai",
        "https://api.x.ai/v1".to_string(),
        None,
        std::collections::HashMap::new(),
    );

    let spec = siumai::providers::xai::spec::XaiSpec;
    let bundle = spec.choose_chat_transformers(&req, &ctx);
    siumai::streaming::TransformerConverter(
        bundle
            .stream
            .expect("xAI should provide streaming transformers"),
    )
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
                assert_eq!(metadata.provider, "xai");
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
