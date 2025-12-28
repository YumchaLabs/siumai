//! Anthropic Prompt Cache fixtures-style tests
//!
//! Validates that message-level cache_control is injected into the request body
//! and that cached token usage is mapped from the response.

use siumai::traits::ChatCapability;
use siumai::types::{ChatMessage, CacheControl};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn anthropic_ok_response() -> serde_json::Value {
    serde_json::json!({
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [ { "type": "text", "text": "Hi!" } ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 10,
            "cache_read_input_tokens": 7
        }
    })
}

#[tokio::test]
async fn anthropic_message_level_cache_control_injected_and_usage_mapped() {
    let server = MockServer::start().await;

    // Expect /v1/messages with x-api-key and message-level cache_control
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(|req: &Request| {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                // basic fields
                if v.get("model").is_none() { return false; }
                // messages[0].cache_control.type == "ephemeral"
                let cc_type = v
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .and_then(|arr| arr.get(0))
                    .and_then(|m0| m0.get("cache_control"))
                    .and_then(|cc| cc.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                cc_type == "ephemeral"
            } else { false }
        })
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok_response()))
        .expect(1)
        .mount(&server)
        .await;

    // Build client via builder to set base_url and model
    let client = siumai::LlmBuilder::default()
        .anthropic()
        .api_key("test-key")
        .base_url(server.uri())
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await
        .expect("client");

    // Message with cache control
    let msg = ChatMessage::user("Hello")
        .cache_control(CacheControl::Ephemeral)
        .build();

    let resp = client.chat(vec![msg]).await.expect("chat ok");
    // Verify cached_tokens is mapped from cache_read_input_tokens
    let usage = resp.usage.expect("usage");
    assert_eq!(usage.cached_tokens, Some(7));
}

#[tokio::test]
async fn anthropic_content_level_cache_control_injected_for_parts() {
    let server = MockServer::start().await;

    // Expect the first content part includes cache_control.type = ephemeral
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(|req: &Request| {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                let first_part_cc_type = v
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .and_then(|arr| arr.get(0))
                    .and_then(|m0| m0.get("content"))
                    .and_then(|c| c.as_array())
                    .and_then(|parts| parts.get(0))
                    .and_then(|p| p.get("cache_control"))
                    .and_then(|cc| cc.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                first_part_cc_type == "ephemeral"
            } else { false }
        })
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = siumai::LlmBuilder::default()
        .anthropic()
        .api_key("test-key")
        .base_url(server.uri())
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await
        .expect("client");

    // Build a multimodal message, mark part[0] for content-level cache via builder API
    let msg = ChatMessage::user("text part")
        .with_image("data:image/jpeg;base64,xxx".into(), None)
        .cache_control_for_part(0, CacheControl::Ephemeral)
        .build();

    let _ = client.chat(vec![msg]).await.expect("chat ok");
}

#[tokio::test]
async fn anthropic_content_level_cache_control_multi_parts_via_builder() {
    let server = MockServer::start().await;

    // Expect both part[0] and part[1] have cache_control.type = ephemeral
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(|req: &Request| {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                let parts = v
                    .get("messages").and_then(|m| m.as_array()).and_then(|arr| arr.get(0))
                    .and_then(|m0| m0.get("content")).and_then(|c| c.as_array())
                    .cloned().unwrap_or_default();
                if parts.len() < 2 { return false; }
                let cc0 = parts[0].get("cache_control").and_then(|cc| cc.get("type")).and_then(|t| t.as_str()).unwrap_or("");
                let cc1 = parts[1].get("cache_control").and_then(|cc| cc.get("type")).and_then(|t| t.as_str()).unwrap_or("");
                cc0 == "ephemeral" && cc1 == "ephemeral"
            } else { false }
        })
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = siumai::LlmBuilder::default()
        .anthropic()
        .api_key("test-key")
        .base_url(server.uri())
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await
        .expect("client");

    // Build a multimodal message with two parts: text + image
    let msg = ChatMessage::user("text part")
        .with_image("data:image/jpeg;base64,xxx".into(), None)
        .cache_control_for_parts([0, 1], CacheControl::Ephemeral)
        .build();

    let _ = client.chat(vec![msg]).await.expect("chat ok");
}
