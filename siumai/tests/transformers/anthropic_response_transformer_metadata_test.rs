#![cfg(feature = "anthropic")]

use siumai::execution::transformers::response::ResponseTransformer;
use siumai::providers::anthropic::transformers::AnthropicResponseTransformer;
use siumai::types::ChatResponse;

#[test]
fn anthropic_response_includes_usage_and_metadata() {
    let tx = AnthropicResponseTransformer::default();

    // Simulate a minimal Anthropic Messages response with usage and thinking.
    let raw = serde_json::json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "content": [
            { "type": "text", "text": "hello" },
            { "type": "thinking", "thinking": "Let me think..." }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 3
        }
    });

    let resp: ChatResponse = tx.transform_chat_response(&raw).expect("ok");

    // Content
    assert_eq!(resp.content_text().unwrap_or_default(), "hello");

    // Usage mapping stays consistent
    let usage = resp.usage.expect("usage should be present");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 20);
    assert_eq!(usage.total_tokens, 30);
    assert_eq!(
        usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens),
        Some(3)
    );

    // Provider metadata should be populated for Anthropic
    let meta = resp
        .anthropic_metadata()
        .expect("Anthropic metadata should be present");

    assert_eq!(meta.cache_creation_input_tokens, Some(5));
    assert_eq!(meta.cache_read_input_tokens, Some(3));
    assert_eq!(meta.thinking.as_deref(), Some("Let me think..."));
}

