#![cfg(feature = "google-vertex")]

use serde_json::json;
use siumai::experimental::core::ProviderContext;
use siumai::experimental::execution::executors::image::{ImageExecutor, ImageExecutorBuilder};
use siumai::prelude::unified::ImageGenerationRequest;
use siumai::prelude::unified::Warning;
use std::sync::Arc;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn vertex_imagen_executor_returns_response_envelope_and_size_warning() {
    let server = MockServer::start().await;
    let base_url = format!(
        "{}/aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google",
        server.uri()
    );

    Mock::given(method("POST"))
        .and(path(
            "/aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/imagen-3.0-generate-002:predict",
        ))
        .and(header("authorization", "Bearer ok"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .insert_header("request-id", "test-request-id")
                .insert_header("x-goog-quota-remaining", "123")
                .set_body_json(json!({
                    "predictions": [{
                        "mimeType": "image/png",
                        "prompt": "revised prompt",
                        "bytesBase64Encoded": "AAA"
                    }]
                })),
        )
        .mount(&server)
        .await;

    let mut extra_headers = std::collections::HashMap::new();
    extra_headers.insert("Authorization".to_string(), "Bearer ok".to_string());
    let ctx = ProviderContext::new("vertex", base_url, None, extra_headers);

    let spec = Arc::new(
        siumai::experimental::providers::google_vertex::standards::vertex_imagen::VertexImagenStandard::new().create_spec("vertex"),
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
        size: Some("1024x1024".to_string()),
        ..Default::default()
    };
    req.extra_params
        .insert("aspectRatio".to_string(), json!("1:1"));

    let before = chrono::Utc::now();
    let out = ImageExecutor::execute(&*exec, req)
        .await
        .expect("execute ok");
    let after = chrono::Utc::now();

    assert_eq!(out.images.len(), 1);
    assert_eq!(out.images[0].b64_json.as_deref(), Some("AAA"));

    let warnings = out.warnings.expect("warnings");
    assert_eq!(warnings.len(), 1);
    assert_eq!(
        warnings[0],
        Warning::unsupported_setting(
            "size",
            Some("This model does not support the `size` option. Use `aspectRatio` instead.")
        )
    );

    let resp = out.response.expect("response envelope");
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
}
