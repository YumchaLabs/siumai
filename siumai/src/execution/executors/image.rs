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
    /// HTTP interceptors for request/response observation and modification
    pub interceptors: Vec<Arc<dyn crate::utils::http_interceptor::HttpInterceptor>>,
    /// Optional external parameter transformer (plugin-like), applied to JSON bodies only
    pub before_send: Option<crate::execution::executors::BeforeSendHook>,
    /// Optional retry options for controlling retry behavior (including 401 retry)
    /// If None, uses default behavior (401 retry enabled)
    pub retry_options: Option<crate::retry_api::RetryOptions>,
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
        if let Some(cb) = &self.before_send {
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
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
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
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
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
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
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
