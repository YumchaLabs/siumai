//! Embedding executor traits

use crate::error::LlmError;
use crate::transformers::{request::RequestTransformer, response::ResponseTransformer};
use crate::types::{EmbeddingRequest, EmbeddingResponse};
use reqwest::header::HeaderMap;
use std::sync::Arc;

#[async_trait::async_trait]
pub trait EmbeddingExecutor: Send + Sync {
    async fn execute(&self, req: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError>;
}

/// Generic HTTP-based Embedding executor
pub struct HttpEmbeddingExecutor {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub request_transformer: Arc<dyn RequestTransformer>,
    pub response_transformer: Arc<dyn ResponseTransformer>,
    pub build_url: Box<dyn Fn(&EmbeddingRequest) -> String + Send + Sync>,
    pub build_headers: Box<dyn Fn() -> Result<HeaderMap, LlmError> + Send + Sync>,
    /// Optional external parameter transformer (plugin-like), applied to JSON body
    pub before_send: Option<
        Arc<dyn Fn(&serde_json::Value) -> Result<serde_json::Value, LlmError> + Send + Sync>,
    >,
}

#[async_trait::async_trait]
impl EmbeddingExecutor for HttpEmbeddingExecutor {
    async fn execute(&self, req: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let body = self.request_transformer.transform_embedding(&req)?;
        let url = (self.build_url)(&req);
        let headers = (self.build_headers)()?;

        let body = if let Some(cb) = &self.before_send {
            cb(&body)?
        } else {
            body
        };

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
        self.response_transformer
            .transform_embedding_response(&json)
    }
}
