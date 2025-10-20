//! Gemini Embedding fixtures-style tests (inspired by Vercel AI)
//!
//! These tests validate request shapes and headers for
//! - single embedding: `models/{model}:embedContent`
//! - batch embeddings: `models/{model}:batchEmbedContents`
//!
//! The goal is to mirror Vercel AI's coverage for:
//! - flattened provider options (`taskType`, `outputDimensionality`, `title`)
//! - correct endpoint selection (single vs batch)
//! - header injection behavior (x-goog-api-key vs Authorization)
//!
//! Note: Unlike Vercel AI SDK (which exposes raw response metadata),
//! our EmbeddingResponse does not currently surface raw headers/body.
//! Therefore, we only assert outgoing request headers and body.

use siumai::providers::gemini::GeminiClient;
use siumai::types::{EmbeddingRequest, EmbeddingTaskType, HttpConfig};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn single_embed_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "embedding": { "values": [0.1, 0.2, 0.3] }
    }))
}

fn batch_embed_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "embeddings": [
            { "values": [0.1, 0.2] },
            { "values": [0.3, 0.4] }
        ]
    }))
}

#[tokio::test]
async fn gemini_single_embedding_request_shape_and_headers() {
    // Arrange mock server and endpoints
    let server = MockServer::start().await;
    // Expect POST to /v1beta/models/gemini-embedding-001:embedContent with flattened fields
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-embedding-001:embedContent"))
        // Ensure API key header is present
        .and(header("x-goog-api-key", "test-key"))
        // Validate request body structure (flattened, no embeddingConfig)
        .and(|req: &Request| {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                // model
                if v.get("model") != Some(&serde_json::Value::String("models/gemini-embedding-001".into())) {
                    return false;
                }
                // content.parts
                if v.get("content").and_then(|c| c.get("parts")).and_then(|p| p.as_array()).map(|a| a.len()) != Some(1) {
                    return false;
                }
                // role should be absent on single
                if v.get("content").and_then(|c| c.get("role")).is_some() {
                    return false;
                }
                // flattened fields present
                if v.get("taskType") != Some(&serde_json::Value::String("RETRIEVAL_QUERY".into())) {
                    return false;
                }
                if v.get("title") != Some(&serde_json::Value::String("My Title".into())) {
                    return false;
                }
                if v.get("outputDimensionality") != Some(&serde_json::Value::Number(768u32.into())) {
                    return false;
                }
                // must NOT include embeddingConfig wrapper
                if v.get("embeddingConfig").is_some() {
                    return false;
                }
                true
            } else {
                false
            }
        })
        .respond_with(single_embed_response())
        .expect(1)
        .mount(&server)
        .await;

    // Build Gemini client
    let mut http_cfg = HttpConfig::default();
    // Also test provider-level custom header pass-through
    http_cfg
        .headers
        .insert("Custom-Provider-Header".into(), "provider-header-value".into());

    let config = siumai::providers::gemini::types::GeminiConfig {
        api_key: "test-key".into(),
        base_url: format!("{}/v1beta", server.uri()),
        model: "gemini-embedding-001".into(),
        generation_config: None,
        safety_settings: None,
        timeout: Some(30),
        http_config: Some(http_cfg),
        token_provider: None,
    };

    let client = GeminiClient::with_http_client(config, reqwest::Client::new())
        .expect("client");

    // Act
    let request = EmbeddingRequest::new(vec!["Hello".to_string()])
        .with_dimensions(768)
        .with_task_type(EmbeddingTaskType::RetrievalQuery)
        .with_provider_param("title", serde_json::Value::String("My Title".into()));

    let out = <GeminiClient as siumai::traits::EmbeddingExtensions>::embed_with_config(&client, request)
        .await
        .expect("embed should succeed");

    // Assert minimal response validation
    assert_eq!(out.embeddings.len(), 1);
}

#[tokio::test]
async fn gemini_batch_embedding_request_shape_and_role() {
    // Arrange mock server and endpoints
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-embedding-001:batchEmbedContents"))
        .and(header("x-goog-api-key", "test-key"))
        .and(|req: &Request| {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                // requests array
                let reqs = match v.get("requests").and_then(|r| r.as_array()) {
                    Some(a) => a,
                    None => return false,
                };
                if reqs.len() != 2 { return false; }
                for (idx, item) in reqs.iter().enumerate() {
                    // model
                    if item.get("model") != Some(&serde_json::Value::String("models/gemini-embedding-001".into())) {
                        return false;
                    }
                    // role should be user in batch
                    if item.get("content").and_then(|c| c.get("role")) != Some(&serde_json::Value::String("user".into())) {
                        return false;
                    }
                    // parts text
                    let expected_text = if idx == 0 { "A" } else { "B" };
                    if item
                        .get("content")
                        .and_then(|c| c.get("parts"))
                        .and_then(|p| p.as_array())
                        .and_then(|a| a.get(0))
                        .and_then(|t| t.get("text"))
                        != Some(&serde_json::Value::String(expected_text.into()))
                    {
                        return false;
                    }
                    // flattened fields
                    if item.get("taskType") != Some(&serde_json::Value::String("SEMANTIC_SIMILARITY".into())) {
                        return false;
                    }
                    if item.get("outputDimensionality") != Some(&serde_json::Value::Number(64u32.into())) {
                        return false;
                    }
                    // must NOT include embeddingConfig wrapper
                    if item.get("embeddingConfig").is_some() {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        })
        .respond_with(batch_embed_response())
        .expect(1)
        .mount(&server)
        .await;

    let config = siumai::providers::gemini::types::GeminiConfig {
        api_key: "test-key".into(),
        base_url: format!("{}/v1beta", server.uri()),
        model: "gemini-embedding-001".into(),
        generation_config: None,
        safety_settings: None,
        timeout: Some(30),
        http_config: Some(HttpConfig::default()),
        token_provider: None,
    };

    let client = GeminiClient::with_http_client(config, reqwest::Client::new())
        .expect("client");

    let request = EmbeddingRequest::new(vec!["A".to_string(), "B".to_string()])
        .with_dimensions(64)
        .with_task_type(EmbeddingTaskType::SemanticSimilarity);

    let out = <GeminiClient as siumai::traits::EmbeddingExtensions>::embed_with_config(&client, request)
        .await
        .expect("embed should succeed");

    assert_eq!(out.embeddings.len(), 2);
}

#[tokio::test]
async fn gemini_embedding_authorization_header_overrides_api_key() {
    // Arrange mock server and endpoint
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-embedding-001:embedContent"))
        // Should contain Authorization header
        .and(header("authorization", "Bearer ya29.test-token"))
        // Should NOT contain x-goog-api-key when Authorization is present
        .and(|req: &Request| !req.headers.contains_key("x-goog-api-key"))
        .respond_with(single_embed_response())
        .expect(1)
        .mount(&server)
        .await;

    // Build config with token provider (simulating Vertex-style auth)
    let token_provider = std::sync::Arc::new(siumai::auth::StaticTokenProvider::new("ya29.test-token"));
    let config = siumai::providers::gemini::types::GeminiConfig {
        api_key: String::new(), // ignored when Authorization is provided
        base_url: format!("{}/v1beta", server.uri()),
        model: "gemini-embedding-001".into(),
        generation_config: None,
        safety_settings: None,
        timeout: Some(30),
        http_config: Some(HttpConfig::default()),
        token_provider: Some(token_provider),
    };

    let client = GeminiClient::with_http_client(config, reqwest::Client::new())
        .expect("client");

    let out = client.embed(vec!["Hello".into()]).await.expect("embed ok");
    assert_eq!(out.embeddings.len(), 1);
}

#[tokio::test]
async fn gemini_embedding_too_many_values_error() {
    // Prepare client (server won't be hit due to pre-check)
    let config = siumai::providers::gemini::types::GeminiConfig {
        api_key: "test-key".into(),
        base_url: "http://localhost/v1beta".into(),
        model: "gemini-embedding-001".into(),
        generation_config: None,
        safety_settings: None,
        timeout: Some(5),
        http_config: Some(HttpConfig::default()),
        token_provider: None,
    };

    let client = GeminiClient::with_http_client(config, reqwest::Client::new())
        .expect("client");

    // Build 2049 inputs (exceeds 2048 limit)
    let values = vec!["x".to_string(); 2049];

    let result = client.embed(values).await;
    assert!(result.is_err(), "should error for too many inputs");
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Too many values for a single embedding call"));
}
