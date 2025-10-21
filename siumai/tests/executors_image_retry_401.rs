use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use serde_json::json;
use siumai::error::LlmError;
use siumai::executors::image::{HttpImageExecutor, ImageExecutor};
use siumai::provider_core::{ImageTransformers, ProviderContext, ProviderSpec};
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

// Test ProviderSpec that returns different headers based on counter
struct TestImageSpec {
    counter: Arc<AtomicUsize>,
    base_url: String,
}

impl ProviderSpec for TestImageSpec {
    fn id(&self) -> &'static str {
        "test"
    }

    fn capabilities(&self) -> siumai::traits::ProviderCapabilities {
        siumai::traits::ProviderCapabilities::new()
    }

    fn build_headers(
        &self,
        _ctx: &ProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
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
    }

    fn chat_url(
        &self,
        _stream: bool,
        _req: &siumai::types::ChatRequest,
        _ctx: &ProviderContext,
    ) -> String {
        unreachable!("chat not used in this test")
    }

    fn choose_chat_transformers(
        &self,
        _req: &siumai::types::ChatRequest,
        _ctx: &ProviderContext,
    ) -> siumai::provider_core::ChatTransformers {
        unreachable!("chat not used in this test")
    }

    fn image_url(&self, _req: &ImageGenerationRequest, _ctx: &ProviderContext) -> String {
        format!("{}/image", self.base_url)
    }

    fn choose_image_transformers(
        &self,
        _req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        ImageTransformers {
            request: Arc::new(TestReqTx),
            response: Arc::new(TestRespTx),
        }
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
    let spec = Arc::new(TestImageSpec {
        counter: counter.clone(),
        base_url: server.uri(),
    });
    let ctx = ProviderContext::new("test", server.uri(), None, Default::default());
    let exec = HttpImageExecutor {
        provider_id: "test".to_string(),
        http_client: reqwest::Client::new(),
        provider_spec: spec,
        provider_context: ctx,
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
