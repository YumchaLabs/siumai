//! Rerank executor traits

use crate::error::LlmError;
// use crate::execution::http::interceptor::HttpInterceptor;
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
    /// Execution policy
    pub policy: crate::execution::ExecutionPolicy,
    pub url: String,
    pub headers: reqwest::header::HeaderMap,
    // Rerank doesn't commonly use JSON mutations; keep for symmetry
    pub before_send: Option<crate::execution::executors::BeforeSendHook>,
}

/// Builder for creating HttpRerankExecutor instances
pub struct RerankExecutorBuilder {
    provider_id: String,
    http_client: reqwest::Client,
    request_transformer: Option<Arc<dyn RerankRequestTransformer>>,
    response_transformer: Option<Arc<dyn RerankResponseTransformer>>,
    policy: crate::execution::ExecutionPolicy,
    url: Option<String>,
    headers: Option<reqwest::header::HeaderMap>,
    before_send: Option<crate::execution::executors::BeforeSendHook>,
}

impl RerankExecutorBuilder {
    pub fn new(provider_id: impl Into<String>, http_client: reqwest::Client) -> Self {
        Self {
            provider_id: provider_id.into(),
            http_client,
            request_transformer: None,
            response_transformer: None,
            policy: crate::execution::ExecutionPolicy::new(),
            url: None,
            headers: None,
            before_send: None,
        }
    }

    pub fn with_transformers(
        mut self,
        request: Arc<dyn RerankRequestTransformer>,
        response: Arc<dyn RerankResponseTransformer>,
    ) -> Self {
        self.request_transformer = Some(request);
        self.response_transformer = Some(response);
        self
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    pub fn with_headers(mut self, headers: reqwest::header::HeaderMap) -> Self {
        self.headers = Some(headers);
        self
    }

    pub fn with_before_send(mut self, hook: crate::execution::executors::BeforeSendHook) -> Self {
        self.before_send = Some(hook);
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

    pub fn build(self) -> Arc<HttpRerankExecutor> {
        Arc::new(HttpRerankExecutor {
            provider_id: self.provider_id,
            http_client: self.http_client,
            request_transformer: self
                .request_transformer
                .expect("rerank request transformer is required"),
            response_transformer: self
                .response_transformer
                .expect("rerank response transformer is required"),
            policy: self.policy,
            url: self.url.expect("rerank url is required"),
            headers: self.headers.unwrap_or_default(),
            before_send: self.before_send,
        })
    }
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
            &self.policy.interceptors,
            self.policy.retry_options.clone(),
            None,
            false,
        )
        .await?;
        self.response_transformer.transform(result.json)
    }
}
