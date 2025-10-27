//! OpenAI Responses API prompt_cache_key fixtures-style tests
//!
//! Validates that when using Responses API, provider params include
//! `prompt_cache_key` flattened into the request body.

use siumai::providers::openai::{OpenAiClient, OpenAiConfig};
use siumai::traits::ChatCapability;
use siumai::types::ChatMessage;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn responses_ok() -> serde_json::Value {
    serde_json::json!({
        "id": "resp_123",
        "model": "gpt-4o-mini",
        "output": [ { "content": [ { "type": "output_text", "text": "ok" } ] } ],
        "usage": { "prompt_tokens": 5, "output_tokens": 1, "total_tokens": 6 }
    })
}

#[tokio::test]
async fn openai_responses_includes_prompt_cache_key() {
    let server = MockServer::start().await;

    // Expect POST /v1/responses with Authorization and prompt_cache_key in body
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("openai-organization", "org-123"))
        .and(header("openai-project", "proj-456"))
        .and(|req: &Request| {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                v.get("prompt_cache_key") == Some(&serde_json::Value::String("cache-123".into()))
            } else { false }
        })
        .respond_with(ResponseTemplate::new(200).set_body_json(responses_ok()))
        .expect(1)
        .mount(&server)
        .await;

    // Build config
    let mut cfg = OpenAiConfig::new("test-key")
        .with_base_url(format!("{}/v1", server.uri()))
        .with_model("gpt-4o-mini")
        .with_organization("org-123")
        .with_project("proj-456")
        .with_responses_api(true);

    // Set prompt_cache_key via OpenAiParams builder
    let openai_params = siumai::params::OpenAiParams::builder()
        .prompt_cache_key("cache-123")
        .build();
    cfg.openai_params = openai_params;

    let client = OpenAiClient::new(cfg, reqwest::Client::new());

    let msg = ChatMessage::user("Hello").build();
    let _ = client.chat(vec![msg]).await.expect("chat ok");
}
