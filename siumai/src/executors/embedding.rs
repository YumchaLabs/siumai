//! Embedding executor traits

use crate::error::LlmError;
use crate::transformers::{request::RequestTransformer, response::ResponseTransformer};
use crate::types::{EmbeddingRequest, EmbeddingResponse};
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
    pub provider_spec: Arc<dyn crate::provider_core::ProviderSpec>,
    pub provider_context: crate::provider_core::ProviderContext,
    /// Optional external parameter transformer (plugin-like), applied to JSON body
    pub before_send: Option<crate::executors::BeforeSendHook>,
}

#[async_trait::async_trait]
impl EmbeddingExecutor for HttpEmbeddingExecutor {
    async fn execute(&self, req: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let body = self.request_transformer.transform_embedding(&req)?;
        let url = self
            .provider_spec
            .embedding_url(&req, &self.provider_context);
        let headers = self.provider_spec.build_headers(&self.provider_context)?;

        let body = if let Some(cb) = &self.before_send {
            cb(&body)?
        } else {
            body
        };

        let resp = self
            .http_client
            .post(&url)
            .headers(headers.clone())
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let resp = if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                // Retry once with rebuilt headers
                let headers = self.provider_spec.build_headers(&self.provider_context)?;
                self.http_client
                    .post(&url)
                    .headers(headers)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?
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
        } else {
            resp
        };
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Use parse_json_with_repair for automatic JSON repair when enabled
        let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.response_transformer
            .transform_embedding_response(&json)
    }
}
