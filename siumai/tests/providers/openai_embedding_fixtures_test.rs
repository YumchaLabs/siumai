//! OpenAI Embedding fixtures-style tests (inspired by Vercel AI)
//!
//! Validates request shape, headers, and usage extraction for OpenAI embeddings.

use siumai::providers::openai::OpenAiClient;
use siumai::providers::openai::OpenAiConfig;
use siumai::traits::EmbeddingExtensions;
use siumai::types::{EmbeddingFormat, EmbeddingRequest, HttpConfig};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn embed_response_json() -> serde_json::Value {
    serde_json::json!({
        "object": "list",
        "data": [
            { "object": "embedding", "index": 0, "embedding": [0.1, 0.2] },
            { "object": "embedding", "index": 1, "embedding": [0.3, 0.4] }
        ],
        "model": "text-embedding-3-large",
        "usage": { "prompt_tokens": 20, "total_tokens": 20 }
    })
}

#[tokio::test]
async fn openai_embedding_request_shape_and_headers() {
    let server = MockServer::start().await;
    // Expect POST /v1/embeddings with auth + org/project headers
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("openai-organization", "org-123"))
        .and(header("openai-project", "proj-456"))
        // Validate request body shape
        .and(|req: &Request| {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                if v.get("model") != Some(&serde_json::Value::String("text-embedding-3-large".into())) { return false; }
                // inputs
                if v.get("input").and_then(|i| i.as_array()).map(|a| a.len()) != Some(2) { return false; }
                // encoding_format present as 'float'
                if v.get("encoding_format") != Some(&serde_json::Value::String("float".into())) { return false; }
                // dimensions
                if v.get("dimensions") != Some(&serde_json::Value::Number(64u32.into())) { return false; }
                true
            } else { false }
        })
        .respond_with(ResponseTemplate::new(200).set_body_json(embed_response_json()))
        .expect(1)
        .mount(&server)
        .await;

    // Build client against mock server
    let cfg = OpenAiConfig::new("test-key")
        .with_base_url(format!("{}/v1", server.uri()))
        .with_organization("org-123")
        .with_project("proj-456")
        .with_model("text-embedding-3-large");
    let client = OpenAiClient::new(cfg, reqwest::Client::new());

    // Build request with encoding_format + custom dimensions
    let req = EmbeddingRequest::new(vec!["A".into(), "B".into()])
        .with_encoding_format(EmbeddingFormat::Float)
        .with_dimensions(64);

    let out = <OpenAiClient as EmbeddingExtensions>::embed_with_config(&client, req)
        .await
        .expect("embed ok");
    assert_eq!(out.embeddings.len(), 2);
    assert!(out.usage.is_some());
    let usage = out.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 20);
    assert_eq!(usage.total_tokens, 20);
}

