#![cfg(feature = "deepseek")]

//! Alignment tests for Vercel `@ai-sdk/deepseek` Chat Completions response fixtures.

use serde_json::Value;
use siumai::prelude::unified::*;
use siumai_protocol_openai::execution::transformers::response::ResponseTransformer;
use siumai_provider_openai_compatible::providers::openai_compatible::transformers::CompatResponseTransformer;
use siumai_provider_openai_compatible::providers::openai_compatible::{
    ConfigurableAdapter, OpenAiCompatibleConfig, get_provider_config,
};
use std::path::Path;
use std::sync::Arc;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("deepseek")
        .join("chat")
}

fn read_json(path: impl AsRef<Path>) -> Value {
    let text = std::fs::read_to_string(path).expect("read fixture json");
    serde_json::from_str(&text).expect("parse fixture json")
}

fn make_transformer(model: &str) -> CompatResponseTransformer {
    let provider_config = get_provider_config("deepseek").expect("deepseek provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));
    let cfg = OpenAiCompatibleConfig::new(
        "deepseek",
        "sk-test",
        &provider_config.base_url,
        adapter.clone(),
    )
    .with_model(model);

    CompatResponseTransformer {
        config: cfg,
        adapter,
    }
}

#[test]
fn deepseek_reasoning_response_maps_reasoning_content_to_reasoning_part() {
    let raw = read_json(fixtures_dir().join("deepseek-reasoning.json"));
    let tx = make_transformer("deepseek-reasoner");
    let resp = tx
        .transform_chat_response(&raw)
        .expect("transform response");

    let parts = resp
        .content
        .as_multimodal()
        .expect("expected multimodal content with reasoning");

    assert!(
        parts.iter().any(|p| matches!(p, ContentPart::Text { .. })),
        "expected a text part"
    );
    let reasoning = parts.iter().find_map(|p| match p {
        ContentPart::Reasoning { text, .. } => Some(text.as_str()),
        _ => None,
    });
    assert!(
        matches!(reasoning, Some(t) if t.contains("strawberry") && t.contains("three")),
        "expected reasoning_content to be preserved"
    );

    let usage = resp.usage.expect("usage");
    assert_eq!(usage.prompt_tokens, 18);
    assert_eq!(usage.total_tokens, 363);
    assert_eq!(
        usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        Some(315)
    );
}

#[test]
fn deepseek_tool_call_response_maps_tool_calls_and_reasoning() {
    let raw = read_json(fixtures_dir().join("deepseek-tool-call.json"));
    let tx = make_transformer("deepseek-reasoner");
    let resp = tx
        .transform_chat_response(&raw)
        .expect("transform response");

    assert!(
        matches!(resp.finish_reason, Some(FinishReason::ToolCalls)),
        "expected tool_calls finish reason"
    );

    let parts = resp
        .content
        .as_multimodal()
        .expect("expected multimodal content with tool call");

    let tool_call = parts.iter().find_map(|p: &ContentPart| p.as_tool_call());
    let tool_call = tool_call.expect("expected tool call content part");
    assert_eq!(tool_call.tool_name, "weather");
    assert_eq!(
        tool_call
            .arguments
            .get("location")
            .and_then(|v: &serde_json::Value| v.as_str()),
        Some("San Francisco")
    );

    let reasoning = parts.iter().find_map(|p| match p {
        ContentPart::Reasoning { text, .. } => Some(text.as_str()),
        _ => None,
    });
    assert!(
        matches!(reasoning, Some(t) if t.contains("weather") && t.contains("San Francisco")),
        "expected reasoning part to be preserved"
    );
}
