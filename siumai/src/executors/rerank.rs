//! Rerank executor traits

use crate::error::LlmError;
use crate::transformers::rerank_request::RerankRequestTransformer;
use crate::transformers::rerank_response::RerankResponseTransformer;
use crate::types::{RerankRequest, RerankResponse};
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

#[async_trait::async_trait]
pub trait RerankExecutor: Send + Sync {
    async fn execute(&self, req: RerankRequest) -> Result<RerankResponse, LlmError>;
}

/// Generic HTTP-based Rerank executor
pub struct HttpRerankExecutor {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub request_transformer: Arc<dyn RerankRequestTransformer>,
    pub response_transformer: Arc<dyn RerankResponseTransformer>,
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    pub retry_options: Option<crate::retry_api::RetryOptions>,
    pub url: String,
    pub headers: reqwest::header::HeaderMap,
    pub before_send: Option<crate::executors::BeforeSendHook>,
}

#[async_trait::async_trait]
impl RerankExecutor for HttpRerankExecutor {
    async fn execute(&self, req: RerankRequest) -> Result<RerankResponse, LlmError> {
        let mut body = self.request_transformer.transform(&req)?;

        // Apply before_send hook if present
        if let Some(cb) = &self.before_send {
            body = cb(&body)?;
        }

        // Create request context for interceptors
        let ctx = crate::utils::http_interceptor::HttpRequestContext {
            provider_id: self.provider_id.clone(),
            url: self.url.clone(),
            stream: false,
        };

        // Build request with interceptors
        let mut builder = self
            .http_client
            .post(&self.url)
            .headers(self.headers.clone())
            .json(&body);

        // Apply interceptors (before send)
        for interceptor in &self.interceptors {
            builder = interceptor.on_before_send(&ctx, builder, &body, &self.headers)?;
        }

        // Send request
        let resp = builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Handle non-success status
        let resp = if !resp.status().is_success() {
            let status = resp.status();
            let should_retry_401 = self
                .retry_options
                .as_ref()
                .map(|opts| opts.retry_401)
                .unwrap_or(true);

            if status.as_u16() == 401 && should_retry_401 {
                // Retry once with same headers
                self.http_client
                    .post(&self.url)
                    .headers(self.headers.clone())
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

        // Parse response
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Use parse_json_with_repair for automatic JSON repair when enabled
        let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        // Transform response
        self.response_transformer.transform(json)
    }
}
