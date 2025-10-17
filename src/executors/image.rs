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
    /// Optional external parameter transformer (plugin-like), applied to JSON bodies only
    pub before_send: Option<crate::executors::BeforeSendHook>,
}

#[async_trait::async_trait]
impl ImageExecutor for HttpImageExecutor {
    async fn execute(
        &self,
        req: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let url = (self.build_url)();
        let build_and_send = || async {
            let body0 = self.request_transformer.transform_image(&req)?;
            let headers0 = (self.build_headers)()?;
            let body0 = if let Some(cb) = &self.before_send {
                cb(&body0)?
            } else {
                body0
            };
            let resp0 = self
                .http_client
                .post(&url)
                .headers(headers0)
                .json(&body0)
                .send()
                .await
                .map_err(|e| LlmError::HttpError(e.to_string()))?;
            Ok::<reqwest::Response, LlmError>(resp0)
        };

        let mut resp = build_and_send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                // Retry once with rebuilt headers/body
                resp = build_and_send().await?;
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
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer.transform_image_response(&json)
    }

    async fn execute_edit(
        &self,
        req: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let url = (self.build_url)();
        let build_and_send = || async {
            let body0 = self.request_transformer.transform_image_edit(&req)?;
            let headers0 = (self.build_headers)()?;
            let builder0 = self.http_client.post(&url).headers(headers0);
            let resp0 = match body0 {
                ImageHttpBody::Json(json) => builder0.json(&json).send().await,
                ImageHttpBody::Multipart(form) => builder0.multipart(form).send().await,
            }
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
            Ok::<reqwest::Response, LlmError>(resp0)
        };
        let mut resp = build_and_send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                resp = build_and_send().await?;
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
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer.transform_image_response(&json)
    }

    async fn execute_variation(
        &self,
        req: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let url = (self.build_url)();
        let build_and_send = || async {
            let body0 = self.request_transformer.transform_image_variation(&req)?;
            let headers0 = (self.build_headers)()?;
            let builder0 = self.http_client.post(&url).headers(headers0);
            let resp0 = match body0 {
                ImageHttpBody::Json(json) => builder0.json(&json).send().await,
                ImageHttpBody::Multipart(form) => builder0.multipart(form).send().await,
            }
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
            Ok::<reqwest::Response, LlmError>(resp0)
        };
        let mut resp = build_and_send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                resp = build_and_send().await?;
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
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer.transform_image_response(&json)
    }
}
