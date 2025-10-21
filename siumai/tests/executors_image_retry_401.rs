use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use serde_json::json;
use siumai::error::LlmError;
use siumai::executors::image::{HttpImageExecutor, ImageExecutor};
use siumai::transformers::{
    request::{ImageHttpBody, RequestTransformer},
    response::ResponseTransformer,
};
use siumai::types::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct TestReqTx;
impl RequestTransformer for TestReqTx {
    fn provider_id(&self) -> &str {
        "test"
    }
    fn transform_chat(
        &self,
        _req: &siumai::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        unreachable!()
    }
    fn transform_image(
        &self,
        _req: &ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Ok(json!({"ok":true}))
    }
    fn transform_image_edit(&self, _req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        Ok(ImageHttpBody::Json(json!({"edit":true})))
    }
    fn transform_image_variation(
        &self,
        _req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        Ok(ImageHttpBody::Json(json!({"var":true})))
    }
}

struct TestRespTx;
impl ResponseTransformer for TestRespTx {
    fn provider_id(&self) -> &str {
        "test"
    }
    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let images = raw["images"].as_array().cloned().unwrap_or_default();
        let images = images
            .into_iter()
            .map(|_| siumai::types::GeneratedImage {
                url: Some("http://x".to_string()),
                b64_json: None,
                format: None,
                width: None,
                height: None,
                revised_prompt: None,
                metadata: Default::default(),
            })
            .collect();
        Ok(ImageGenerationResponse {
            images,
            metadata: Default::default(),
        })
    }
}

fn headers_builder_factory(
    counter: Arc<AtomicUsize>,
) -> impl Fn() -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, LlmError>> + Send>,
> + Send
+ Sync
+ 'static {
    move || {
        let counter = counter.clone();
        Box::pin(async move {
            use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
            let n = counter.fetch_add(1, Ordering::SeqCst);
            let token = if n == 0 { "bad" } else { "ok" };
            let mut h = HeaderMap::new();
            h.insert(
                HeaderName::from_static("authorization"),
                HeaderValue::from_str(&format!("Bearer {}", token)).unwrap(),
            );
            h.insert(
                HeaderName::from_static("content-type"),
                HeaderValue::from_static("application/json"),
            );
            Ok(h)
        })
    }
}

#[tokio::test]
async fn image_executor_retries_on_401() {
    let server = MockServer::start().await;

    let unauthorized = ResponseTemplate::new(401).set_body_json(json!({"error": {"code":401}}));
    let ok = ResponseTemplate::new(200).set_body_json(json!({"images":[{"url":"http://x"}]}));

    // single endpoint for simplicity
    Mock::given(method("POST"))
        .and(path("/image"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ok)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/image"))
        .respond_with(unauthorized)
        .mount(&server)
        .await;

    let counter = Arc::new(AtomicUsize::new(0));
    let exec = HttpImageExecutor {
        provider_id: "test".to_string(),
        http_client: reqwest::Client::new(),
        request_transformer: Arc::new(TestReqTx),
        response_transformer: Arc::new(TestRespTx),
        build_url: Box::new({
            let base = server.uri();
            move || format!("{}/image", base)
        }),
        build_headers: Box::new(headers_builder_factory(counter)),
        before_send: None,
    };

    // execute
    let out = exec
        .execute(ImageGenerationRequest {
            prompt: "p".into(),
            count: 1,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(out.images.len(), 1);

    // edit
    let out = exec
        .execute_edit(ImageEditRequest {
            image: vec![],
            mask: None,
            prompt: "p".into(),
            count: None,
            size: None,
            response_format: None,
            extra_params: Default::default(),
        })
        .await
        .unwrap();
    assert_eq!(out.images.len(), 1);

    // variation
    let out = exec
        .execute_variation(ImageVariationRequest {
            image: vec![],
            count: None,
            size: None,
            response_format: None,
            extra_params: Default::default(),
        })
        .await
        .unwrap();
    assert_eq!(out.images.len(), 1);
}
