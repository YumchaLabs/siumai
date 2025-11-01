//! Ollama streaming fixtures tests
//!
//! These tests verify Ollama streaming responses using real API response fixtures.
//! All fixtures are based on official Ollama API documentation:
//! https://github.com/ollama/ollama/blob/main/docs/api.md

use siumai::providers::ollama::streaming::OllamaEventConverter;
use siumai::streaming::ChatStreamEvent;

use crate::support;

fn make_ollama_converter() -> OllamaEventConverter {
    OllamaEventConverter::new()
}

#[tokio::test]
async fn ollama_simple_content_then_done_fixture() {
    let bytes = support::load_jsonl_fixture_as_bytes(
        "tests/fixtures/ollama/simple_content_then_done.jsonl",
    )
    .expect("load fixture");

    let converter = make_ollama_converter();
    let events = support::collect_json_events(bytes, converter).await;

    let mut content = String::new();
    let mut saw_start = false;
    let mut saw_end = false;
    let mut usage_total = 0u32;

    for e in events {
        match e {
            ChatStreamEvent::StreamStart { metadata } => {
                saw_start = true;
                assert_eq!(metadata.model, Some("llama3.2".to_string()));
                assert_eq!(metadata.provider, "ollama");
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
    assert_eq!(content, "Hello world!");
    assert_eq!(usage_total, 324); // 26 + 298
    assert!(saw_end, "expect stream end");
}

#[tokio::test]
async fn ollama_thinking_and_content_fixture() {
    let bytes =
        support::load_jsonl_fixture_as_bytes("tests/fixtures/ollama/thinking_and_content.jsonl")
            .expect("load fixture");

    let converter = make_ollama_converter();
    let events = support::collect_json_events(bytes, converter).await;

    let mut thinking = String::new();
    let mut content = String::new();
    let mut saw_end = false;

    for e in events {
        match e {
            ChatStreamEvent::ThinkingDelta { delta } => thinking.push_str(&delta),
            ChatStreamEvent::ContentDelta { delta, .. } => content.push_str(&delta),
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

    assert!(thinking.contains("Let me think about this"));
    assert!(thinking.contains("The answer is 42"));
    assert_eq!(content, "The answer is 42.");
    assert!(saw_end);
}
