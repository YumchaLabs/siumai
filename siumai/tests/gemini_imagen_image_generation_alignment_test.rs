#![cfg(feature = "google")]

use serde_json::json;
use siumai::prelude::*;
use siumai_core::types::Warning;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn gemini_imagen_defaults_aspect_ratio_and_emits_warnings() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/imagen-3.0-generate-002:predict"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(json!({
            "instances": [{ "prompt": "a cat" }],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "1:1"
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "predictions": [{ "bytesBase64Encoded": "b64-image" }]
        })))
        .mount(&mock_server)
        .await;

    let client = Siumai::builder()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("imagen-3.0-generate-002")
        .build()
        .await
        .unwrap();

    let mut req = ImageGenerationRequest::default();
    req.prompt = "a cat".to_string();
    req.count = 1;
    req.size = Some("1024x1024".to_string());
    req.seed = Some(42);

    let resp = client.generate_images(req).await.unwrap();

    assert_eq!(resp.images.len(), 1);
    assert_eq!(
        resp.images[0].b64_json.as_deref(),
        Some("b64-image"),
        "expected Imagen prediction bytes to map to b64_json"
    );

    assert_eq!(
        resp.warnings.unwrap_or_default(),
        vec![
            Warning::unsupported_setting(
                "size",
                Some("This model does not support the `size` option. Use `aspectRatio` instead.")
            ),
            Warning::unsupported_setting(
                "seed",
                Some("This model does not support the `seed` option through this provider.")
            ),
        ]
    );
}
