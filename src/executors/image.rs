//! Image generation executor traits

use crate::error::LlmError;
use crate::transformers::{
    request::{ImageHttpBody, RequestTransformer},
    response::ResponseTransformer,
};
use crate::types::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};
use reqwest::header::HeaderMap;
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
    pub build_url: Box<dyn Fn() -> String + Send + Sync>,
    pub build_headers: Box<dyn Fn() -> Result<HeaderMap, LlmError> + Send + Sync>,
}

#[async_trait::async_trait]
impl ImageExecutor for HttpImageExecutor {
    async fn execute(
        &self,
        req: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let body = self.request_transformer.transform_image(&req)?;
        let url = (self.build_url)();
        let headers = (self.build_headers)()?;

        let resp = self
            .http_client
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer.transform_image_response(&json)
    }

    async fn execute_edit(
        &self,
        req: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let body = self.request_transformer.transform_image_edit(&req)?;
        let url = (self.build_url)();
        let headers = (self.build_headers)()?;
        let builder = self.http_client.post(url).headers(headers);
        let resp = match body {
            ImageHttpBody::Json(json) => builder.json(&json).send().await,
            ImageHttpBody::Multipart(form) => builder.multipart(form).send().await,
        }
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer.transform_image_response(&json)
    }

    async fn execute_variation(
        &self,
        req: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let body = self.request_transformer.transform_image_variation(&req)?;
        let url = (self.build_url)();
        let headers = (self.build_headers)()?;
        let builder = self.http_client.post(url).headers(headers);
        let resp = match body {
            ImageHttpBody::Json(json) => builder.json(&json).send().await,
            ImageHttpBody::Multipart(form) => builder.multipart(form).send().await,
        }
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer.transform_image_response(&json)
    }
}
