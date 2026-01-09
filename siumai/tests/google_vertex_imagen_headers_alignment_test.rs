#![cfg(feature = "google-vertex")]

use serde_json::json;
use siumai::experimental::core::ProviderContext;
use siumai::experimental::execution::executors::image::{ImageExecutor, ImageExecutorBuilder};
use siumai::prelude::unified::ImageGenerationRequest;
use std::collections::HashMap;
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn vertex_imagen_executor_merges_headers_and_sets_user_agent() {
    let server = MockServer::start().await;
    let base_url = format!(
        "{}/aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google",
        server.uri()
    );

    Mock::given(method("POST"))
        .and(path(
            "/aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/imagen-3.0-generate-002:predict",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [{
                "mimeType": "image/png",
                "prompt": "revised prompt",
                "bytesBase64Encoded": "AAA"
            }]
        })))
        .mount(&server)
        .await;

    let mut extra_headers = HashMap::new();
    extra_headers.insert("Authorization".to_string(), "Bearer ok".to_string());
    extra_headers.insert("X-Custom-Header".to_string(), "custom-value".to_string());
    let ctx = ProviderContext::new("vertex", base_url, None, extra_headers);

    let spec = Arc::new(
        siumai::experimental::providers::google_vertex::standards::vertex_imagen::VertexImagenStandard::new()
            .create_spec("vertex"),
    );
    let exec = ImageExecutorBuilder::new("vertex", reqwest::Client::new())
        .with_spec(spec)
        .with_context(ctx)
        .build_for_request(&ImageGenerationRequest {
            model: Some("imagen-3.0-generate-002".to_string()),
            ..Default::default()
        });

    let mut req = ImageGenerationRequest {
        prompt: "p".to_string(),
        model: Some("imagen-3.0-generate-002".to_string()),
        count: 1,
        ..Default::default()
    };
    let mut http = siumai::prelude::unified::HttpConfig::default();
    http.headers
        .insert("X-Request-Header".to_string(), "request-value".to_string());
    req.http_config = Some(http);

    let _ = ImageExecutor::execute(&*exec, req)
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
    assert_eq!(header_value("authorization").as_deref(), Some("Bearer ok"));
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
