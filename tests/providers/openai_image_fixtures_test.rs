//! OpenAI Image Generation fixtures-style tests (inspired by Vercel AI)
//!
//! Validates request shape, headers and response extraction for image generation.

use siumai::providers::openai::{OpenAiClient, OpenAiConfig};
use siumai::traits::ImageGenerationCapability;
use siumai::types::ImageGenerationRequest;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn image_generate_response() -> serde_json::Value {
    serde_json::json!({
        "created": 1733837122,
        "data": [
            { "b64_json": "base64-image-1", "revised_prompt": "A vivid cat." },
            { "b64_json": "base64-image-2" }
        ]
    })
}

#[tokio::test]
async fn openai_dalle3_request_shape_and_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/images/generations"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("openai-organization", "org-123"))
        .and(header("openai-project", "proj-456"))
        .and(|req: &Request| {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                if v.get("model") != Some(&serde_json::Value::String("dall-e-3".into())) { return false; }
                if v.get("prompt") != Some(&serde_json::Value::String("A cat".into())) { return false; }
                if v.get("n") != Some(&serde_json::Value::Number(1u32.into())) { return false; }
                if v.get("size") != Some(&serde_json::Value::String("1024x1024".into())) { return false; }
                if v.get("style") != Some(&serde_json::Value::String("vivid".into())) { return false; }
                // We explicitly set response_format in request
                if v.get("response_format") != Some(&serde_json::Value::String("b64_json".into())) { return false; }
                true
            } else { false }
        })
        .respond_with(ResponseTemplate::new(200).set_body_json(image_generate_response()))
        .expect(1)
        .mount(&server)
        .await;

    let cfg = OpenAiConfig::new("test-key")
        .with_base_url(format!("{}/v1", server.uri()))
        .with_organization("org-123")
        .with_project("proj-456")
        .with_model("dall-e-3");
    let client = OpenAiClient::new(cfg, reqwest::Client::new());

    let mut req = ImageGenerationRequest::default();
    req.model = Some("dall-e-3".into());
    req.prompt = "A cat".into();
    req.count = 1;
    req.size = Some("1024x1024".into());
    req.style = Some("vivid".into());
    req.response_format = Some("b64_json".into());

    let out = client.generate_images(req).await.expect("image ok");
    assert_eq!(out.images.len(), 2);
    assert!(out.images.iter().all(|i| i.b64_json.is_some()));
    assert!(out.metadata.contains_key("created"));
}

#[tokio::test]
async fn openai_gpt_image1_does_not_include_response_format_if_not_set() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/images/generations"))
        .and(|req: &Request| {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                // Model set
                if v.get("model") != Some(&serde_json::Value::String("gpt-image-1".into())) { return false; }
                // response_format should be absent if not provided
                if v.get("response_format").is_some() { return false; }
                true
            } else { false }
        })
        .respond_with(ResponseTemplate::new(200).set_body_json(image_generate_response()))
        .expect(1)
        .mount(&server)
        .await;

    let cfg = OpenAiConfig::new("test-key").with_base_url(format!("{}/v1", server.uri()));
    let client = OpenAiClient::new(cfg, reqwest::Client::new());

    let mut req = ImageGenerationRequest::default();
    req.model = Some("gpt-image-1".into());
    req.prompt = "An otter".into();
    req.count = 1;

    let out = client.generate_images(req).await.expect("image ok");
    assert_eq!(out.images.len(), 2);
}

