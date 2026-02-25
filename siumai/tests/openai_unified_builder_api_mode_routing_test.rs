#![cfg(feature = "openai")]

use siumai::prelude::*;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn unified_builder_openai_defaults_to_responses_api() {
    let server = MockServer::start().await;

    let body = include_str!("fixtures/openai/responses/response/basic-text/response.json");
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_raw(body, "application/json"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .openai()
        .api_key("test-api-key")
        .base_url(format!("{}/v1", server.uri()))
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("build");

    let resp = client.chat(vec![user!("hi")]).await.expect("chat ok");
    assert_eq!(resp.content_text().unwrap_or_default(), "answer text");
}

#[tokio::test]
async fn unified_builder_openai_chat_uses_chat_completions() {
    let server = MockServer::start().await;

    let body = serde_json::json!({
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": { "role": "assistant", "content": "hello from chat completions" },
                "finish_reason": "stop"
            }
        ],
        "usage": { "prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = Siumai::builder()
        .openai_chat()
        .api_key("test-api-key")
        .base_url(format!("{}/v1", server.uri()))
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("build");

    let resp = client.chat(vec![user!("hi")]).await.expect("chat ok");
    assert_eq!(
        resp.content_text().unwrap_or_default(),
        "hello from chat completions"
    );
}
