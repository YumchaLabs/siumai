#![cfg(feature = "togetherai")]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::compat::Provider;
use siumai::experimental::client::LlmClient;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
    HttpTransportStreamResponse,
};
use siumai::prelude::extensions::types::{ImageEditInput, ImageEditRequest};
use siumai::prelude::unified::*;
use siumai::provider_ext::togetherai::{TogetherAiImageOptions, TogetherAiImageRequestExt};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct JsonCaptureTransport {
    request: Arc<Mutex<Option<HttpTransportRequest>>>,
    response: serde_json::Value,
}

impl JsonCaptureTransport {
    fn new(response: serde_json::Value) -> Self {
        Self {
            request: Arc::new(Mutex::new(None)),
            response,
        }
    }

    fn take(&self) -> Option<HttpTransportRequest> {
        self.request.lock().expect("lock request").take()
    }
}

#[async_trait]
impl HttpTransport for JsonCaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.request.lock().expect("lock request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: serde_json::to_vec(&self.response).expect("serialize response"),
        })
    }

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        *self.request.lock().expect("lock request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_bytes(
                serde_json::to_vec(&self.response).expect("serialize response"),
            ),
        })
    }
}

fn togetherai_image_response() -> serde_json::Value {
    serde_json::json!({
        "data": [
            {
                "b64_json": "aGVsbG8=",
                "revised_prompt": "a tiny robot"
            }
        ]
    })
}

#[tokio::test]
async fn togetherai_public_builder_routes_image_generation_to_provider_owned_endpoint() {
    let base_url = "https://example.com/together";
    let transport = JsonCaptureTransport::new(togetherai_image_response());

    let client = Provider::togetherai()
        .api_key("test-key")
        .base_url(base_url)
        .model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build togetherai client");

    let response = client
        .as_image_generation_capability()
        .expect("image capability")
        .generate_images(
            ImageGenerationRequest {
                prompt: "a tiny robot".to_string(),
                size: Some("1024x768".to_string()),
                aspect_ratio: Some("4:3".to_string()),
                response_format: Some("url".to_string()),
                ..Default::default()
            }
            .with_togetherai_image_options(
                TogetherAiImageOptions::new()
                    .with_steps(28)
                    .with_guidance(3.5)
                    .with_negative_prompt("blurry"),
            ),
        )
        .await
        .expect("image generation ok");

    let request = transport.take().expect("captured request");
    assert_eq!(request.url, format!("{base_url}/images/generations"));
    assert_eq!(
        request.body["model"],
        serde_json::json!("black-forest-labs/FLUX.1-schnell")
    );
    assert_eq!(request.body["prompt"], serde_json::json!("a tiny robot"));
    assert_eq!(request.body["width"], serde_json::json!(1024));
    assert_eq!(request.body["height"], serde_json::json!(768));
    assert_eq!(request.body["response_format"], serde_json::json!("base64"));
    assert_eq!(request.body["steps"], serde_json::json!(28));
    assert_eq!(request.body["guidance"], serde_json::json!(3.5));
    assert_eq!(request.body["negative_prompt"], serde_json::json!("blurry"));
    assert!(request.body.get("size").is_none());
    assert!(request.body.get("aspect_ratio").is_none());

    assert_eq!(response.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        response.warnings,
        Some(vec![Warning::unsupported(
            "aspectRatio",
            Some("This model does not support the `aspectRatio` option. Use `size` instead.")
        )])
    );
}

#[tokio::test]
async fn togetherai_public_builder_routes_image_edit_to_provider_owned_generation_endpoint() {
    let base_url = "https://example.com/together";
    let transport = JsonCaptureTransport::new(togetherai_image_response());

    let client = Provider::togetherai()
        .api_key("test-key")
        .base_url(base_url)
        .model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build togetherai client");

    let response = client
        .as_image_extras()
        .expect("image extras")
        .edit_image(
            ImageEditRequest {
                images: vec![
                    ImageEditInput::url("https://example.com/input.png"),
                    ImageEditInput::file_with_media_type(vec![137, 80, 78, 71], "image/png"),
                ],
                mask: None,
                prompt: "turn this into pixel art".to_string(),
                model: None,
                count: Some(2),
                size: Some("640x480".to_string()),
                aspect_ratio: None,
                seed: Some(7),
                response_format: Some("url".to_string()),
                extra_params: Default::default(),
                provider_options_map: Default::default(),
                http_config: None,
            }
            .with_togetherai_image_options(
                TogetherAiImageOptions::new()
                    .with_steps(12)
                    .with_disable_safety_checker(true),
            ),
        )
        .await
        .expect("image edit ok");

    let request = transport.take().expect("captured request");
    assert_eq!(request.url, format!("{base_url}/images/generations"));
    assert_eq!(
        request.body["model"],
        serde_json::json!("black-forest-labs/FLUX.1-schnell")
    );
    assert_eq!(
        request.body["image_url"],
        serde_json::json!("https://example.com/input.png")
    );
    assert_eq!(
        request.body["prompt"],
        serde_json::json!("turn this into pixel art")
    );
    assert_eq!(request.body["n"], serde_json::json!(2));
    assert_eq!(request.body["seed"], serde_json::json!(7));
    assert_eq!(request.body["width"], serde_json::json!(640));
    assert_eq!(request.body["height"], serde_json::json!(480));
    assert_eq!(
        request.body["disable_safety_checker"],
        serde_json::json!(true)
    );
    assert_eq!(request.body["steps"], serde_json::json!(12));
    assert_eq!(request.body["response_format"], serde_json::json!("base64"));
    assert!(request.body.get("mask").is_none());

    assert_eq!(response.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        response.warnings,
        Some(vec![Warning::other(
            "Together AI only supports a single input image. Additional images are ignored."
        )])
    );
}

#[tokio::test]
async fn togetherai_public_builder_rejects_mask_based_image_editing() {
    let base_url = "https://example.com/together";
    let transport = JsonCaptureTransport::new(togetherai_image_response());

    let client = Provider::togetherai()
        .api_key("test-key")
        .base_url(base_url)
        .model("black-forest-labs/FLUX.1-schnell")
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build togetherai client");

    let error = client
        .as_image_extras()
        .expect("image extras")
        .edit_image(ImageEditRequest {
            images: vec![ImageEditInput::url("https://example.com/input.png")],
            mask: Some(ImageEditInput::file_with_media_type(
                vec![255, 255, 255, 0],
                "image/png",
            )),
            prompt: "edit with mask".to_string(),
            model: None,
            count: Some(1),
            size: None,
            aspect_ratio: None,
            seed: None,
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        })
        .await
        .expect_err("mask edit should be rejected");

    assert!(matches!(error, LlmError::UnsupportedOperation(_)));
    assert!(transport.take().is_none());
}
