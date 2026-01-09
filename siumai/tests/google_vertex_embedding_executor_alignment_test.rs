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
async fn vertex_embedding_executor_merges_headers_and_sets_user_agent() {
    let server = MockServer::start().await;

    let base_url = format!(
        "{}/aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google",
        server.uri()
    );

    Mock::given(method("POST"))
        .and(path(
            "/aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google/models/textembedding-gecko@001:predict",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [
                { "embeddings": { "values": [0.1, 0.2, 0.3], "statistics": { "token_count": 1 } } },
                { "embeddings": { "values": [0.4, 0.5, 0.6], "statistics": { "token_count": 1 } } }
            ]
        })))
        .mount(&server)
        .await;

    let mut extra_headers = HashMap::new();
    extra_headers.insert("X-Custom-Header".to_string(), "custom-value".to_string());
    let ctx = ProviderContext::new("vertex", base_url, None, extra_headers);

    let spec = Arc::new(
        siumai::experimental::providers::google_vertex::standards::vertex_embedding::VertexEmbeddingStandard::new()
            .create_spec("vertex"),
    );

    let mut req = EmbeddingRequest::new(vec!["test text one".into(), "test text two".into()])
        .with_model("textembedding-gecko@001");
    req = req.with_header("X-Request-Header", "request-value");

    let exec = EmbeddingExecutorBuilder::new("vertex", reqwest::Client::new())
        .with_spec(spec)
        .with_context(ctx)
        .build_for_request(&req);

    let _ = EmbeddingExecutor::execute(&*exec, req)
        .await
        .expect("execute ok");

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 1);
    let got = &requests[0];

    let header_value = |name: &str| -> Option<String> {
        got.headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    };

    assert_eq!(
        header_value("content-type").as_deref(),
        Some("application/json")
    );
    assert_eq!(
        header_value("x-custom-header").as_deref(),
        Some("custom-value")
    );
    assert_eq!(
        header_value("x-request-header").as_deref(),
        Some("request-value")
    );

    let ua = header_value("user-agent").unwrap_or_default();
    let expected = format!("siumai/google-vertex/{}", env!("CARGO_PKG_VERSION"));
    assert!(
        ua.contains(&expected),
        "expected user-agent to contain {expected}, got {ua}"
    );
}
