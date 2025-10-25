//! Rerank executor traits

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::transformers::rerank_request::RerankRequestTransformer;
use crate::execution::transformers::rerank_response::RerankResponseTransformer;
use crate::types::{RerankRequest, RerankResponse};
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
    pub before_send: Option<crate::execution::executors::BeforeSendHook>,
}

#[async_trait::async_trait]
impl RerankExecutor for HttpRerankExecutor {
    async fn execute(&self, req: RerankRequest) -> Result<RerankResponse, LlmError> {
        let mut body = self.request_transformer.transform(&req)?;

        // Apply before_send hook if present
        if let Some(cb) = &self.before_send {
            body = cb(&body)?;
        }
        let result = crate::execution::executors::common::execute_json_request_with_headers(
            &self.http_client,
            &self.provider_id,
            &self.url,
            self.headers.clone(),
            body,
            &self.interceptors,
            self.retry_options.clone(),
            None,
            false,
        )
        .await?;
        self.response_transformer.transform(result.json)
    }
}
