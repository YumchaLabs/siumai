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
async fn vertex_imagen_executor_exposes_response_envelope_headers_and_model_id() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/imagen-3.0-generate-002:predict"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .insert_header("request-id", "test-request-id")
                .insert_header("x-goog-quota-remaining", "123")
                .set_body_json(json!({
                    "predictions": [
                        {
                            "mimeType": "image/png",
                            "prompt": "revised prompt 1",
                            "bytesBase64Encoded": "base64-image-1"
                        }
                    ]
                })),
        )
        .mount(&server)
        .await;

    let ctx = ProviderContext::new("vertex", server.uri(), None, HashMap::new());
    let spec = Arc::new(
        siumai::experimental::providers::google_vertex::standards::vertex_imagen::VertexImagenStandard::new()
            .create_spec("vertex"),
    );

    let req = ImageGenerationRequest {
        prompt: "test prompt".to_string(),
        count: 1,
        model: Some("imagen-3.0-generate-002".to_string()),
        response_format: Some("b64_json".to_string()),
        ..Default::default()
    };

    let exec = ImageExecutorBuilder::new("vertex", reqwest::Client::new())
        .with_spec(spec)
        .with_context(ctx)
        .build_for_request(&req);

    let before = chrono::Utc::now();
    let out = ImageExecutor::execute(&*exec, req)
        .await
        .expect("execute ok");
    let after = chrono::Utc::now();

    let resp = out.response.expect("expected response envelope");
    assert!(resp.timestamp >= before && resp.timestamp <= after);
    assert_eq!(resp.model_id.as_deref(), Some("imagen-3.0-generate-002"));
    assert_eq!(
        resp.headers.get("request-id").map(|s| s.as_str()),
        Some("test-request-id")
    );
    assert_eq!(
        resp.headers
            .get("x-goog-quota-remaining")
            .map(|s| s.as_str()),
        Some("123")
    );
    assert!(
        resp.headers.contains_key("content-type"),
        "expected content-type header in response"
    );
}
