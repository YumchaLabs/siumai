//! Image generation executor traits

use crate::error::LlmError;
use crate::transformers::{
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
    pub provider_spec: Arc<dyn crate::provider_core::ProviderSpec>,
    pub provider_context: crate::provider_core::ProviderContext,
    /// Optional external parameter transformer (plugin-like), applied to JSON bodies only
    pub before_send: Option<crate::executors::BeforeSendHook>,
}

#[async_trait::async_trait]
impl ImageExecutor for HttpImageExecutor {
    async fn execute(
        &self,
        req: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let url = self.provider_spec.image_url(&req, &self.provider_context);
        let headers = self.provider_spec.build_headers(&self.provider_context)?;

        let body = self.request_transformer.transform_image(&req)?;
        let body = if let Some(cb) = &self.before_send {
            cb(&body)?
        } else {
            body
        };

        let mut resp = self
            .http_client
            .post(&url)
            .headers(headers.clone())
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                // Retry once with rebuilt headers
                let headers = self.provider_spec.build_headers(&self.provider_context)?;
                resp = self
                    .http_client
                    .post(&url)
                    .headers(headers)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?;
            } else {
                let headers = resp.headers().clone();
                let text = resp.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    &self.provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Use parse_json_with_repair for automatic JSON repair when enabled
        let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer.transform_image_response(&json)
    }

    async fn execute_edit(
        &self,
        req: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let url = self
            .provider_spec
            .image_edit_url(&req, &self.provider_context);
        let headers = self.provider_spec.build_headers(&self.provider_context)?;

        let body = self.request_transformer.transform_image_edit(&req)?;
        let builder = self.http_client.post(&url).headers(headers.clone());
        let mut resp = match body {
            ImageHttpBody::Json(json) => builder.json(&json).send().await,
            ImageHttpBody::Multipart(form) => builder.multipart(form).send().await,
        }
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                // Retry once with rebuilt headers
                let headers = self.provider_spec.build_headers(&self.provider_context)?;
                let body = self.request_transformer.transform_image_edit(&req)?;
                let builder = self.http_client.post(&url).headers(headers);
                resp = match body {
                    ImageHttpBody::Json(json) => builder.json(&json).send().await,
                    ImageHttpBody::Multipart(form) => builder.multipart(form).send().await,
                }
                .map_err(|e| LlmError::HttpError(e.to_string()))?;
            } else {
                let headers = resp.headers().clone();
                let text = resp.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    &self.provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Use parse_json_with_repair for automatic JSON repair when enabled
        let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer.transform_image_response(&json)
    }

    async fn execute_variation(
        &self,
        req: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let url = self
            .provider_spec
            .image_variation_url(&req, &self.provider_context);
        let headers = self.provider_spec.build_headers(&self.provider_context)?;

        let body = self.request_transformer.transform_image_variation(&req)?;
        let builder = self.http_client.post(&url).headers(headers.clone());
        let mut resp = match body {
            ImageHttpBody::Json(json) => builder.json(&json).send().await,
            ImageHttpBody::Multipart(form) => builder.multipart(form).send().await,
        }
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                // Retry once with rebuilt headers
                let headers = self.provider_spec.build_headers(&self.provider_context)?;
                let body = self.request_transformer.transform_image_variation(&req)?;
                let builder = self.http_client.post(&url).headers(headers);
                resp = match body {
                    ImageHttpBody::Json(json) => builder.json(&json).send().await,
                    ImageHttpBody::Multipart(form) => builder.multipart(form).send().await,
                }
                .map_err(|e| LlmError::HttpError(e.to_string()))?;
            } else {
                let headers = resp.headers().clone();
                let text = resp.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    &self.provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Use parse_json_with_repair for automatic JSON repair when enabled
        let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer.transform_image_response(&json)
    }
}
