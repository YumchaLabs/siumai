//! Groq streaming fixtures tests
//!
//! These tests verify Groq streaming responses using real API response fixtures.
//! All fixtures are based on official Groq API documentation:
//! https://console.groq.com/docs/api-reference

use siumai::standards::openai::compat::openai_config::OpenAiCompatibleConfig;
use siumai::standards::openai::compat::provider_registry::{
    ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
};
use siumai::standards::openai::compat::streaming::OpenAiCompatibleEventConverter;
use siumai::streaming::ChatStreamEvent;

use crate::support;

fn make_groq_converter() -> OpenAiCompatibleEventConverter {
    let provider_config = ProviderConfig {
        id: "groq".to_string(),
        name: "Groq".to_string(),
        base_url: "https://api.groq.com/openai/v1".to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec!["chat".to_string(), "streaming".to_string()],
        default_model: Some("llama-3.3-70b-versatile".to_string()),
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: Vec::new(),
    };
    let adapter = std::sync::Arc::new(ConfigurableAdapter::new(provider_config));
    let cfg = OpenAiCompatibleConfig::new("groq", "", "", adapter.clone())
        .with_model("llama-3.3-70b-versatile");
    OpenAiCompatibleEventConverter::new(cfg, adapter)
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
