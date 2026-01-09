#![cfg(feature = "google-vertex")]

use serde_json::json;
use siumai::experimental::core::ProviderContext;
use siumai::experimental::execution::executors::embedding::{
    EmbeddingExecutor, EmbeddingExecutorBuilder,
};
use siumai::prelude::unified::EmbeddingRequest;
use std::collections::HashMap;
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn vertex_embedding_executor_exposes_response_headers() {
    let server = MockServer::start().await;

    let base_url = format!(
        "{}/aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google",
        server.uri()
    );

    Mock::given(method("POST"))
        .and(path(
            "/aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google/models/textembedding-gecko@001:predict",
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .insert_header("test-header", "test-value")
                .set_body_json(json!({
                    "predictions": [
                        { "embeddings": { "values": [0.1, 0.2, 0.3], "statistics": { "token_count": 1 } } },
                        { "embeddings": { "values": [0.4, 0.5, 0.6], "statistics": { "token_count": 1 } } }
                    ]
                })),
        )
        .mount(&server)
        .await;

    let ctx = ProviderContext::new("vertex", base_url, None, HashMap::new());
    let spec = Arc::new(
        siumai::experimental::providers::google_vertex::standards::vertex_embedding::VertexEmbeddingStandard::new()
            .create_spec("vertex"),
    );

    let req = EmbeddingRequest::new(vec!["test text one".into(), "test text two".into()])
        .with_model("textembedding-gecko@001");

    let exec = EmbeddingExecutorBuilder::new("vertex", reqwest::Client::new())
        .with_spec(spec)
        .with_context(ctx)
        .build_for_request(&req);

    let before = chrono::Utc::now();
    let out = EmbeddingExecutor::execute(&*exec, req)
        .await
        .expect("execute ok");
    let after = chrono::Utc::now();

    let resp = out.response.expect("expected response envelope");
    assert!(resp.timestamp >= before && resp.timestamp <= after);
    assert_eq!(resp.model_id.as_deref(), Some("textembedding-gecko@001"));
    assert_eq!(
        resp.headers.get("test-header").map(|s| s.as_str()),
        Some("test-value")
    );
    assert!(
        resp.headers.contains_key("content-type"),
        "expected content-type header in response"
    );
}

#[tokio::test]
async fn vertex_embedding_executor_uses_custom_base_url_prefix() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/textembedding-gecko@001:predict"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [
                { "embeddings": { "values": [0.1], "statistics": { "token_count": 1 } } }
            ]
        })))
        .mount(&server)
        .await;

    let ctx = ProviderContext::new("vertex", server.uri(), None, HashMap::new());
    let spec = Arc::new(
        siumai::experimental::providers::google_vertex::standards::vertex_embedding::VertexEmbeddingStandard::new()
            .create_spec("vertex"),
    );

    let req = EmbeddingRequest::new(vec!["test".into()]).with_model("textembedding-gecko@001");
    let exec = EmbeddingExecutorBuilder::new("vertex", reqwest::Client::new())
        .with_spec(spec)
        .with_context(ctx)
        .build_for_request(&req);

    let _ = EmbeddingExecutor::execute(&*exec, req)
        .await
        .expect("execute ok");

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url.path(),
        "/models/textembedding-gecko@001:predict"
    );
}
