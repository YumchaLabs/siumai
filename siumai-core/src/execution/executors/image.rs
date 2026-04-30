//! Image generation executor traits

use crate::error::LlmError;
use crate::execution::transformers::{
    request::{ImageHttpBody, RequestTransformer},
    response::ResponseTransformer,
};
use crate::types::{HttpResponseInfo, Warning};
use crate::types::{
    ImageEditFileData, ImageEditInput, ImageEditRequest, ImageGenerationRequest,
    ImageGenerationResponse, ImageVariationRequest,
};
use base64::Engine;
use reqwest::header::CONTENT_TYPE;
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use std::sync::Arc;

fn parse_image_data_url(url: &str, label: &str) -> Result<(Vec<u8>, Option<String>), LlmError> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Err(LlmError::InvalidParameter(format!(
            "Expected a data URL for {label}"
        )));
    };
    let Some((meta, data)) = payload.split_once(',') else {
        return Err(LlmError::InvalidParameter(format!(
            "Invalid data URL for {label}"
        )));
    };

    let Some(meta) = meta.strip_suffix(";base64") else {
        return Err(LlmError::InvalidParameter(format!(
            "{label} data URLs must use base64 encoding"
        )));
    };

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|err| {
            LlmError::InvalidParameter(format!("Invalid base64 payload in {label} data URL: {err}"))
        })?;
    let media_type = (!meta.is_empty()).then_some(meta.to_string());
    Ok((bytes, media_type))
}

async fn materialize_url_backed_image_input(
    http_client: &reqwest::Client,
    input: &ImageEditInput,
    label: &str,
) -> Result<ImageEditInput, LlmError> {
    match input {
        ImageEditInput::File { .. } => Ok(input.clone()),
        ImageEditInput::Url {
            url,
            provider_options_map,
        } => {
            let (bytes, media_type) = if url.starts_with("data:") {
                let (bytes, media_type) = parse_image_data_url(url, label)?;
                let media_type =
                    media_type.unwrap_or_else(|| crate::utils::guess_mime(Some(&bytes), Some(url)));
                (bytes, media_type)
            } else if url.starts_with("http://") || url.starts_with("https://") {
                let response = http_client.get(url).send().await.map_err(|err| {
                    LlmError::HttpError(format!("Failed to download {label}: {err}"))
                })?;
                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(LlmError::ApiError {
                        code: status.as_u16(),
                        message: format!("Failed to download {label} from {url}"),
                        details: Some(serde_json::json!({ "url": url, "body": body })),
                    });
                }

                let content_type = response
                    .headers()
                    .get(CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.split(';').next())
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string);
                let bytes = response.bytes().await.map_err(|err| {
                    LlmError::HttpError(format!("Failed to read {label} bytes: {err}"))
                })?;
                let media_type = content_type.unwrap_or_else(|| {
                    crate::utils::guess_mime(Some(bytes.as_ref()), Some(url.as_str()))
                });
                (bytes.to_vec(), media_type)
            } else {
                return Err(LlmError::InvalidParameter(format!(
                    "Unsupported URL scheme for {label}. Only data:, http:, and https: URLs can be materialized on this image path."
                )));
            };

            Ok(ImageEditInput::File {
                data: ImageEditFileData::binary(bytes),
                media_type: Some(media_type),
                provider_options_map: provider_options_map.clone(),
            })
        }
    }
}

async fn materialize_url_backed_image_edit_request(
    http_client: &reqwest::Client,
    mut req: ImageEditRequest,
) -> Result<ImageEditRequest, LlmError> {
    if !req.images.iter().any(ImageEditInput::is_url)
        && !req.mask.as_ref().is_some_and(ImageEditInput::is_url)
    {
        return Ok(req);
    }

    let mut materialized_images = Vec::with_capacity(req.images.len());
    for (index, image) in req.images.iter().enumerate() {
        materialized_images.push(
            materialize_url_backed_image_input(
                http_client,
                image,
                &format!("image input {}", index + 1),
            )
            .await?,
        );
    }
    req.images = materialized_images;

    if let Some(mask) = req.mask.as_ref() {
        req.mask = Some(materialize_url_backed_image_input(http_client, mask, "mask").await?);
    }

    Ok(req)
}

async fn materialize_url_backed_image_variation_request(
    http_client: &reqwest::Client,
    mut req: ImageVariationRequest,
) -> Result<ImageVariationRequest, LlmError> {
    if !req.image.is_url() {
        return Ok(req);
    }

    req.image =
        materialize_url_backed_image_input(http_client, &req.image, "variation image").await?;
    Ok(req)
}

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

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.policy.transport = Some(transport);
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

    /// Build the executor, selecting transformers from the ProviderSpec if they were not set.
    ///
    /// # Panics
    /// Panics if required fields (spec, context) are not set.
    pub fn build_for_request(mut self, request: &ImageGenerationRequest) -> Arc<HttpImageExecutor> {
        let spec = self.spec.take().expect("provider_spec is required");
        let ctx = self.context.take().expect("provider_context is required");

        if self.request_transformer.is_none() || self.response_transformer.is_none() {
            let bundle = spec.choose_image_transformers(request, &ctx);
            self.request_transformer = Some(bundle.request);
            self.response_transformer = Some(bundle.response);
        }

        self.spec = Some(spec);
        self.context = Some(ctx);
        self.build()
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
        let retry_options = self.policy.retry_options.clone();
        let run_once = move || {
            let req = req.clone();
            async move {
                let warnings = self
                    .provider_spec
                    .image_warnings(&req, &self.provider_context);
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
                    transport: self.policy.transport.clone(),
                    provider_spec: self.provider_spec.clone(),
                    provider_context: self.provider_context.clone(),
                    interceptors: self.policy.interceptors.clone(),
                    retry_options: self.policy.retry_options.clone(),
                };

                // 5. Execute request using common HTTP layer
                let result = crate::execution::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(body),
                    req.http_config.as_ref(),
                    false, // stream = false
                )
                .await?;

                // 6. Transform response
                let mut out = self
                    .response_transformer
                    .transform_image_response(&result.json)?;

                out.warnings = merge_warnings(out.warnings, warnings);
                out.response = Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: req.model.clone().filter(|m| !m.is_empty()),
                    headers: headers_to_map(&result.headers),
                    body: None,
                });

                Ok(out)
            }
        };

        if let Some(opts) = retry_options {
            crate::retry_api::retry_with(run_once, opts).await
        } else {
            run_once().await
        }
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
        let req = if self
            .provider_spec
            .materialize_image_edit_urls(&req, &self.provider_context)
        {
            materialize_url_backed_image_edit_request(&self.http_client, req).await?
        } else {
            req
        };
        // 1. Get URL from provider spec
        let url = self
            .provider_spec
            .image_edit_url(&req, &self.provider_context);

        // 2. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            transport: self.policy.transport.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 3. Transform request and execute based on body type
        let body = self.request_transformer.transform_image_edit(&req)?;
        let warnings = self
            .provider_spec
            .image_edit_warnings(&req, &self.provider_context);
        let result = match body {
            ImageHttpBody::Json(json) => {
                // Use JSON request path
                crate::execution::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(json),
                    req.http_config.as_ref(),
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
                    req.http_config.as_ref(),
                )
                .await?
            }
        };

        // 4. Transform response
        let mut out = self
            .response_transformer
            .transform_image_response(&result.json)?;
        out.warnings = merge_warnings(out.warnings, warnings);
        out.response = Some(HttpResponseInfo {
            timestamp: chrono::Utc::now(),
            model_id: req.model.clone().filter(|m| !m.is_empty()),
            headers: headers_to_map(&result.headers),
            body: None,
        });
        Ok(out)
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
        let req = if self
            .provider_spec
            .materialize_image_variation_urls(&req, &self.provider_context)
        {
            materialize_url_backed_image_variation_request(&self.http_client, req).await?
        } else {
            req
        };
        // 1. Get URL from provider spec
        let url = self
            .provider_spec
            .image_variation_url(&req, &self.provider_context);

        // 2. Build execution config for common HTTP layer
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            transport: self.policy.transport.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        // 3. Transform request and execute based on body type
        let body = self.request_transformer.transform_image_variation(&req)?;
        let warnings = self
            .provider_spec
            .image_variation_warnings(&req, &self.provider_context);
        let result = match body {
            ImageHttpBody::Json(json) => {
                // Use JSON request path
                crate::execution::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(json),
                    req.http_config.as_ref(),
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
                    req.http_config.as_ref(),
                )
                .await?
            }
        };

        // 4. Transform response
        let mut out = self
            .response_transformer
            .transform_image_response(&result.json)?;
        out.warnings = merge_warnings(out.warnings, warnings);
        out.response = Some(HttpResponseInfo {
            timestamp: chrono::Utc::now(),
            model_id: req.model.clone().filter(|m| !m.is_empty()),
            headers: headers_to_map(&result.headers),
            body: None,
        });
        Ok(out)
    }
}

fn headers_to_map(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(k, v)| Some((k.as_str().to_string(), v.to_str().ok()?.to_string())))
        .collect()
}

fn merge_warnings(
    existing: Option<Vec<Warning>>,
    extra: Option<Vec<Warning>>,
) -> Option<Vec<Warning>> {
    match (existing, extra) {
        (None, None) => None,
        (Some(w), None) | (None, Some(w)) => Some(w),
        (Some(mut a), Some(b)) => {
            a.extend(b);
            Some(a)
        }
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
            crate::traits::ProviderCapabilities::new().with_image_generation()
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

    struct OkImgResp;
    impl crate::execution::transformers::response::ResponseTransformer for OkImgResp {
        fn provider_id(&self) -> &str {
            "test"
        }
        fn transform_image_response(
            &self,
            _raw: &serde_json::Value,
        ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
            Ok(crate::types::ImageGenerationResponse {
                images: vec![],
                metadata: HashMap::new(),
                warnings: None,
                response: None,
            })
        }
    }

    struct InspectMaterializedEditReq;
    impl crate::execution::transformers::request::RequestTransformer for InspectMaterializedEditReq {
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
            _req: &crate::types::ImageGenerationRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({}))
        }

        fn transform_image_edit(
            &self,
            req: &crate::types::ImageEditRequest,
        ) -> Result<ImageHttpBody, LlmError> {
            let first = req.images.first().expect("expected first image");
            let data = first.file_data().expect("image should be materialized");
            assert_eq!(data.as_bytes().expect("image bytes"), vec![1, 2, 3, 4]);
            assert_eq!(first.media_type(), Some("image/png"));
            assert_eq!(
                first.provider_options_map().get("openai"),
                Some(&serde_json::json!({ "detail": "high" }))
            );

            let mask = req.mask.as_ref().expect("mask should exist");
            let mask_data = mask.file_data().expect("mask should be materialized");
            assert_eq!(mask_data.as_bytes().expect("mask bytes"), vec![5, 6, 7, 8]);
            assert_eq!(mask.media_type(), Some("image/png"));

            Ok(ImageHttpBody::Json(serde_json::json!({ "ok": true })))
        }
    }

    struct InspectMaterializedVariationReq;
    impl crate::execution::transformers::request::RequestTransformer
        for InspectMaterializedVariationReq
    {
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
            _req: &crate::types::ImageGenerationRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({}))
        }

        fn transform_image_variation(
            &self,
            req: &crate::types::ImageVariationRequest,
        ) -> Result<ImageHttpBody, LlmError> {
            let data = req
                .image
                .file_data()
                .expect("variation image should be materialized");
            assert_eq!(data.as_bytes().expect("variation bytes"), vec![9, 8, 7, 6]);
            assert_eq!(req.image.media_type(), Some("image/png"));
            assert_eq!(
                req.image.provider_options_map().get("openai"),
                Some(&serde_json::json!({ "detail": "low" }))
            );

            Ok(ImageHttpBody::Json(serde_json::json!({ "ok": true })))
        }
    }

    #[derive(Clone, Copy)]
    struct TestSpecNoMaterialize;
    impl crate::core::ProviderSpec for TestSpecNoMaterialize {
        fn id(&self) -> &'static str {
            "test"
        }

        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            crate::traits::ProviderCapabilities::new().with_image_generation()
        }

        fn build_headers(
            &self,
            _ctx: &crate::core::ProviderContext,
        ) -> Result<HeaderMap, LlmError> {
            let mut h = HeaderMap::new();
            h.insert(
                reqwest::header::CONTENT_TYPE,
                reqwest::header::HeaderValue::from_static("application/json"),
            );
            Ok(h)
        }

        fn materialize_image_edit_urls(
            &self,
            _req: &crate::types::ImageEditRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> bool {
            false
        }

        fn materialize_image_variation_urls(
            &self,
            _req: &crate::types::ImageVariationRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> bool {
            false
        }
    }

    struct InspectUnmaterializedEditReq;
    impl crate::execution::transformers::request::RequestTransformer for InspectUnmaterializedEditReq {
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
            _req: &crate::types::ImageGenerationRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({}))
        }

        fn transform_image_edit(
            &self,
            req: &crate::types::ImageEditRequest,
        ) -> Result<ImageHttpBody, LlmError> {
            let first = req.images.first().expect("expected first image");
            assert!(
                first.is_url(),
                "expected URL input to remain unmaterialized"
            );
            assert_eq!(first.as_url(), Some("https://example.com/input.png"));

            Ok(ImageHttpBody::Json(serde_json::json!({ "ok": true })))
        }
    }

    struct InspectUnmaterializedVariationReq;
    impl crate::execution::transformers::request::RequestTransformer
        for InspectUnmaterializedVariationReq
    {
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
            _req: &crate::types::ImageGenerationRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({}))
        }

        fn transform_image_variation(
            &self,
            req: &crate::types::ImageVariationRequest,
        ) -> Result<ImageHttpBody, LlmError> {
            assert!(
                req.image.is_url(),
                "expected URL variation input to remain unmaterialized"
            );
            assert_eq!(req.image.as_url(), Some("https://example.com/input.png"));

            Ok(ImageHttpBody::Json(serde_json::json!({ "ok": true })))
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
        let req = crate::types::ImageGenerationRequest {
            prompt: "hello".into(),
            ..Default::default()
        };
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
        let mut req = crate::types::ImageGenerationRequest {
            prompt: "hello".into(),
            ..Default::default()
        };
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

    #[tokio::test]
    async fn image_executor_populates_response_and_warnings() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/images/generations")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_header("request-id", "test-request-id")
            .with_header("x-goog-quota-remaining", "123")
            .with_body("{\"ok\":true}")
            .create_async()
            .await;

        #[derive(Clone, Copy)]
        struct WarningSpec;
        impl crate::core::ProviderSpec for WarningSpec {
            fn id(&self) -> &'static str {
                "test"
            }
            fn capabilities(&self) -> crate::traits::ProviderCapabilities {
                crate::traits::ProviderCapabilities::new().with_image_generation()
            }
            fn build_headers(
                &self,
                _ctx: &crate::core::ProviderContext,
            ) -> Result<HeaderMap, LlmError> {
                Ok(HeaderMap::new())
            }
            fn image_warnings(
                &self,
                req: &crate::types::ImageGenerationRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> Option<Vec<Warning>> {
                if req.size.is_some() {
                    return Some(vec![Warning::unsupported_setting(
                        "size",
                        Some(
                            "This model does not support the `size` option. Use `aspectRatio` instead.",
                        ),
                    )]);
                }
                None
            }
        }

        let http = reqwest::Client::new();
        let ctx = crate::core::ProviderContext::new(
            "test",
            server.url(),
            None,
            std::collections::HashMap::new(),
        );
        let exec = HttpImageExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer: Arc::new(ImgReq),
            response_transformer: Arc::new(OkImgResp),
            provider_spec: Arc::new(WarningSpec),
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new(),
        };

        let before = chrono::Utc::now();
        let req = crate::types::ImageGenerationRequest {
            prompt: "hello".into(),
            model: Some("imagen-3.0-generate-002".into()),
            size: Some("1024x1024".into()),
            ..Default::default()
        };
        let out = exec.execute(req).await.expect("execute ok");
        let after = chrono::Utc::now();

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

        let warnings = out.warnings.expect("warnings");
        assert_eq!(
            warnings[0],
            Warning::unsupported(
                "size",
                Some("This model does not support the `size` option. Use `aspectRatio` instead.")
            )
        );
    }

    #[tokio::test]
    async fn image_edit_executor_materializes_url_backed_inputs_before_transformer() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/images/edits")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{\"ok\":true}")
            .create_async()
            .await;

        let exec = HttpImageExecutor {
            provider_id: "test".into(),
            http_client: reqwest::Client::new(),
            request_transformer: Arc::new(InspectMaterializedEditReq),
            response_transformer: Arc::new(OkImgResp),
            provider_spec: Arc::new(TestSpec),
            provider_context: crate::core::ProviderContext::new(
                "test",
                server.url(),
                None,
                Default::default(),
            ),
            policy: crate::execution::ExecutionPolicy::new(),
        };

        let request = crate::types::ImageEditRequest {
            images: vec![
                crate::types::ImageEditInput::url("data:image/png;base64,AQIDBA==")
                    .with_provider_option("openai", serde_json::json!({ "detail": "high" })),
            ],
            mask: Some(crate::types::ImageEditInput::url(
                "data:image/png;base64,BQYHCA==",
            )),
            prompt: "edit".to_string(),
            model: Some("gpt-image-1".to_string()),
            count: Some(1),
            size: None,
            aspect_ratio: None,
            seed: None,
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        exec.execute_edit(request)
            .await
            .expect("edit request should succeed");
    }

    #[tokio::test]
    async fn image_variation_executor_materializes_url_backed_input_before_transformer() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/images/variations")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{\"ok\":true}")
            .create_async()
            .await;

        let exec = HttpImageExecutor {
            provider_id: "test".into(),
            http_client: reqwest::Client::new(),
            request_transformer: Arc::new(InspectMaterializedVariationReq),
            response_transformer: Arc::new(OkImgResp),
            provider_spec: Arc::new(TestSpec),
            provider_context: crate::core::ProviderContext::new(
                "test",
                server.url(),
                None,
                Default::default(),
            ),
            policy: crate::execution::ExecutionPolicy::new(),
        };

        let request = crate::types::ImageVariationRequest {
            image: crate::types::ImageEditInput::url("data:image/png;base64,CQgHBg==")
                .with_provider_option("openai", serde_json::json!({ "detail": "low" })),
            model: Some("gpt-image-1".to_string()),
            count: Some(1),
            size: None,
            aspect_ratio: None,
            seed: None,
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        exec.execute_variation(request)
            .await
            .expect("variation request should succeed");
    }

    #[tokio::test]
    async fn image_edit_executor_can_preserve_url_backed_inputs_when_provider_opt_outs() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/images/edits")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{\"ok\":true}")
            .create_async()
            .await;

        let exec = HttpImageExecutor {
            provider_id: "test".into(),
            http_client: reqwest::Client::new(),
            request_transformer: Arc::new(InspectUnmaterializedEditReq),
            response_transformer: Arc::new(OkImgResp),
            provider_spec: Arc::new(TestSpecNoMaterialize),
            provider_context: crate::core::ProviderContext::new(
                "test",
                server.url(),
                None,
                Default::default(),
            ),
            policy: crate::execution::ExecutionPolicy::new(),
        };

        let request = crate::types::ImageEditRequest {
            images: vec![crate::types::ImageEditInput::url(
                "https://example.com/input.png",
            )],
            mask: None,
            prompt: "edit".to_string(),
            model: Some("gemini-image".to_string()),
            count: Some(1),
            size: None,
            aspect_ratio: None,
            seed: None,
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        exec.execute_edit(request)
            .await
            .expect("edit request should succeed");
    }

    #[tokio::test]
    async fn image_variation_executor_can_preserve_url_backed_inputs_when_provider_opt_outs() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/images/variations")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{\"ok\":true}")
            .create_async()
            .await;

        let exec = HttpImageExecutor {
            provider_id: "test".into(),
            http_client: reqwest::Client::new(),
            request_transformer: Arc::new(InspectUnmaterializedVariationReq),
            response_transformer: Arc::new(OkImgResp),
            provider_spec: Arc::new(TestSpecNoMaterialize),
            provider_context: crate::core::ProviderContext::new(
                "test",
                server.url(),
                None,
                Default::default(),
            ),
            policy: crate::execution::ExecutionPolicy::new(),
        };

        let request = crate::types::ImageVariationRequest {
            image: crate::types::ImageEditInput::url("https://example.com/input.png"),
            model: Some("gemini-image".to_string()),
            count: Some(1),
            size: None,
            aspect_ratio: None,
            seed: None,
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        exec.execute_variation(request)
            .await
            .expect("variation request should succeed");
    }
}
