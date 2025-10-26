//! Image generation executor traits

use crate::error::LlmError;
use crate::execution::transformers::{
    request::{ImageHttpBody, RequestTransformer},
    response::ResponseTransformer,
};
use crate::types::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};
use std::sync::Arc;

#[async_trait::async_trait]
pub trait ImageExecutor: Send + Sync {
    async fn execute(
        &self,
        req: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError>;
    async fn execute_edit(
        &self,
        req: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError>;
    async fn execute_variation(
        &self,
        req: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError>;
}

/// Generic HTTP-based Image executor
pub struct HttpImageExecutor {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub request_transformer: Arc<dyn RequestTransformer>,
    pub response_transformer: Arc<dyn ResponseTransformer>,
    pub provider_spec: Arc<dyn crate::core::ProviderSpec>,
    pub provider_context: crate::core::ProviderContext,
    /// Execution policy
    pub policy: crate::execution::ExecutionPolicy,
}

/// Builder for creating HttpImageExecutor instances
pub struct ImageExecutorBuilder {
    provider_id: String,
    http_client: reqwest::Client,
    spec: Option<Arc<dyn crate::core::ProviderSpec>>,
    context: Option<crate::core::ProviderContext>,
    request_transformer: Option<Arc<dyn RequestTransformer>>,
    response_transformer: Option<Arc<dyn ResponseTransformer>>,
    policy: crate::execution::ExecutionPolicy,
}

impl ImageExecutorBuilder {
    pub fn new(provider_id: impl Into<String>, http_client: reqwest::Client) -> Self {
        Self {
            provider_id: provider_id.into(),
            http_client,
            spec: None,
            context: None,
            request_transformer: None,
            response_transformer: None,
            policy: crate::execution::ExecutionPolicy::new(),
        }
    }

    pub fn with_spec(mut self, spec: Arc<dyn crate::core::ProviderSpec>) -> Self {
        self.spec = Some(spec);
        self
    }

    pub fn with_context(mut self, context: crate::core::ProviderContext) -> Self {
        self.context = Some(context);
        self
    }

    pub fn with_transformers(
        mut self,
        request: Arc<dyn RequestTransformer>,
        response: Arc<dyn ResponseTransformer>,
    ) -> Self {
        self.request_transformer = Some(request);
        self.response_transformer = Some(response);
        self
    }

    pub fn with_before_send(mut self, hook: crate::execution::executors::BeforeSendHook) -> Self {
        self.policy.before_send = Some(hook);
        self
    }

    pub fn with_interceptors(
        mut self,
        interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    ) -> Self {
        self.policy.interceptors = interceptors;
        self
    }

    pub fn with_retry_options(mut self, retry_options: crate::retry_api::RetryOptions) -> Self {
        self.policy.retry_options = Some(retry_options);
        self
    }

    pub fn build(self) -> Arc<HttpImageExecutor> {
        Arc::new(HttpImageExecutor {
            provider_id: self.provider_id,
            http_client: self.http_client,
            request_transformer: self
                .request_transformer
                .expect("request_transformer is required"),
            response_transformer: self
                .response_transformer
                .expect("response_transformer is required"),
            provider_spec: self.spec.expect("provider_spec is required"),
            provider_context: self.context.expect("provider_context is required"),
            policy: self.policy,
        })
    }
}

#[async_trait::async_trait]
impl ImageExecutor for HttpImageExecutor {
    async fn execute(
        &self,
        req: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        // Capability guard: image generation is a custom feature
        let caps = self.provider_spec.capabilities();
        if !caps.supports("image_generation") {
            return Err(LlmError::UnsupportedOperation(
                "Image generation is not supported by this provider".to_string(),
            ));
        }
        // 1. Transform request to JSON
        let mut body = self.request_transformer.transform_image(&req)?;

        // 2. Apply before_send hook if present
        if let Some(cb) = &self.policy.before_send {
            body = cb(&body)?;
        }

        // 3. Get URL from provider spec
        let url = self.provider_spec.image_url(&req, &self.provider_context);

        // 4. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 5. Execute request using common HTTP layer
        let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
        let result = crate::execution::executors::common::execute_json_request(
            &config,
            &url,
            crate::execution::executors::common::HttpBody::Json(body),
            per_request_headers,
            false, // stream = false
        )
        .await?;

        // 6. Transform response
        self.response_transformer
            .transform_image_response(&result.json)
    }

    async fn execute_edit(
        &self,
        req: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("image_generation") {
            return Err(LlmError::UnsupportedOperation(
                "Image editing is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL from provider spec
        let url = self
            .provider_spec
            .image_edit_url(&req, &self.provider_context);

        // 2. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 3. Transform request and execute based on body type
        let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
        let body = self.request_transformer.transform_image_edit(&req)?;
        let result = match body {
            ImageHttpBody::Json(json) => {
                // Use JSON request path
                crate::execution::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(json),
                    per_request_headers,
                    false, // stream = false
                )
                .await?
            }
            ImageHttpBody::Multipart(_) => {
                // Use multipart request path
                let req_clone = req.clone();
                crate::execution::executors::common::execute_multipart_request(
                    &config,
                    &url,
                    || {
                        self.request_transformer
                            .transform_image_edit(&req_clone)
                            .and_then(|body| match body {
                                ImageHttpBody::Multipart(form) => Ok(form),
                                _ => Err(LlmError::InvalidParameter(
                                    "Expected multipart body".into(),
                                )),
                            })
                    },
                    per_request_headers,
                )
                .await?
            }
        };

        // 4. Transform response
        self.response_transformer
            .transform_image_response(&result.json)
    }

    async fn execute_variation(
        &self,
        req: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("image_generation") {
            return Err(LlmError::UnsupportedOperation(
                "Image variation is not supported by this provider".to_string(),
            ));
        }
        // 1. Get URL from provider spec
        let url = self
            .provider_spec
            .image_variation_url(&req, &self.provider_context);

        // 2. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 3. Transform request and execute based on body type
        let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
        let body = self.request_transformer.transform_image_variation(&req)?;
        let result = match body {
            ImageHttpBody::Json(json) => {
                // Use JSON request path
                crate::execution::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(json),
                    per_request_headers,
                    false, // stream = false
                )
                .await?
            }
            ImageHttpBody::Multipart(_) => {
                // Use multipart request path
                let req_clone = req.clone();
                crate::execution::executors::common::execute_multipart_request(
                    &config,
                    &url,
                    || {
                        self.request_transformer
                            .transform_image_variation(&req_clone)
                            .and_then(|body| match body {
                                ImageHttpBody::Multipart(form) => Ok(form),
                                _ => Err(LlmError::InvalidParameter(
                                    "Expected multipart body".into(),
                                )),
                            })
                    },
                    per_request_headers,
                )
                .await?
            }
        };

        // 4. Transform response
        self.response_transformer
            .transform_image_response(&result.json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    use std::sync::{Arc, Mutex};

    // Minimal ProviderSpec for image
    #[derive(Clone, Copy)]
    struct TestSpec;
    impl crate::core::ProviderSpec for TestSpec {
        fn id(&self) -> &'static str {
            "test"
        }
        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            crate::traits::ProviderCapabilities::new().with_custom_feature("image_generation", true)
        }
        fn build_headers(
            &self,
            _ctx: &crate::core::ProviderContext,
        ) -> Result<HeaderMap, LlmError> {
            let mut h = HeaderMap::new();
            h.insert(
                HeaderName::from_static("x-base"),
                HeaderValue::from_static("B"),
            );
            Ok(h)
        }
        fn chat_url(
            &self,
            _s: bool,
            _r: &crate::types::ChatRequest,
            _c: &crate::core::ProviderContext,
        ) -> String {
            unreachable!()
        }
        fn choose_chat_transformers(
            &self,
            _r: &crate::types::ChatRequest,
            _c: &crate::core::ProviderContext,
        ) -> crate::core::ChatTransformers {
            unreachable!()
        }
    }

    // Request/response transformers
    struct ImgReq;
    impl crate::execution::transformers::request::RequestTransformer for ImgReq {
        fn provider_id(&self) -> &str {
            "test"
        }
        fn transform_chat(
            &self,
            _req: &crate::types::ChatRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({}))
        }
        fn transform_image(
            &self,
            req: &crate::types::ImageGenerationRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({"prompt": req.prompt, "tag":"orig"}))
        }
        fn transform_image_edit(
            &self,
            _req: &crate::types::ImageEditRequest,
        ) -> Result<ImageHttpBody, LlmError> {
            Ok(ImageHttpBody::Json(serde_json::json!({"edit":true})))
        }
        fn transform_image_variation(
            &self,
            _req: &crate::types::ImageVariationRequest,
        ) -> Result<ImageHttpBody, LlmError> {
            Ok(ImageHttpBody::Json(serde_json::json!({"var":true})))
        }
    }
    struct ImgResp;
    impl crate::execution::transformers::response::ResponseTransformer for ImgResp {
        fn provider_id(&self) -> &str {
            "test"
        }
        fn transform_image_response(
            &self,
            _raw: &serde_json::Value,
        ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
            Err(LlmError::InvalidParameter("abort".into()))
        }
    }

    // Interceptor to capture
    struct CaptureHeaders {
        seen: Arc<Mutex<Option<HeaderMap>>>,
    }
    impl HttpInterceptor for CaptureHeaders {
        fn on_before_send(
            &self,
            _ctx: &HttpRequestContext,
            _rb: reqwest::RequestBuilder,
            _body: &serde_json::Value,
            headers: &HeaderMap,
        ) -> Result<reqwest::RequestBuilder, LlmError> {
            *self.seen.lock().unwrap() = Some(headers.clone());
            Err(LlmError::InvalidParameter("abort".into()))
        }
    }

    #[tokio::test]
    async fn image_before_send_applied_and_aborts() {
        let http = reqwest::Client::new();
        let spec = Arc::new(TestSpec);
        let ctx =
            crate::core::ProviderContext::new("test", "http://127.0.0.1", None, Default::default());
        let exec = HttpImageExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer: Arc::new(ImgReq),
            response_transformer: Arc::new(ImgResp),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new().with_before_send(Arc::new(|body| {
                assert_eq!(body["tag"], serde_json::json!("orig"));
                Err(LlmError::InvalidParameter("abort".into()))
            })),
        };
        let mut req = crate::types::ImageGenerationRequest::default();
        req.prompt = "hello".into();
        let err = exec.execute(req).await.unwrap_err();
        match err {
            LlmError::InvalidParameter(m) => assert_eq!(m, "abort"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn image_interceptor_merges_headers() {
        let http = reqwest::Client::new();
        let spec = Arc::new(TestSpec);
        let ctx =
            crate::core::ProviderContext::new("test", "http://127.0.0.1", None, Default::default());
        let seen = Arc::new(Mutex::new(None::<HeaderMap>));
        let interceptor = Arc::new(CaptureHeaders { seen: seen.clone() });
        let exec = HttpImageExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer: Arc::new(ImgReq),
            response_transformer: Arc::new(ImgResp),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new().with_interceptors(vec![interceptor]),
        };
        let mut req = crate::types::ImageGenerationRequest::default();
        req.prompt = "hello".into();
        let mut hc = crate::types::HttpConfig::default();
        hc.headers.insert("x-req".into(), "R".into());
        req.http_config = Some(hc);
        let _ = exec.execute(req).await; // expect abort
        let headers = seen.lock().unwrap().clone().unwrap();
        assert_eq!(
            headers.get("x-base").unwrap(),
            &HeaderValue::from_static("B")
        );
        assert_eq!(
            headers.get("x-req").unwrap(),
            &HeaderValue::from_static("R")
        );
    }
}
