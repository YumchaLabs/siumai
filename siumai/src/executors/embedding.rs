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
    pub provider_spec: Arc<dyn crate::core::ProviderSpec>,
    pub provider_context: crate::core::ProviderContext,
    /// HTTP interceptors for request/response observation and modification
    pub interceptors: Vec<Arc<dyn crate::utils::http_interceptor::HttpInterceptor>>,
    /// Optional external parameter transformer (plugin-like), applied to JSON body
    pub before_send: Option<crate::executors::BeforeSendHook>,
    /// Optional retry options for controlling retry behavior (including 401 retry)
    /// If None, uses default behavior (401 retry enabled)
    pub retry_options: Option<crate::retry_api::RetryOptions>,
}

#[async_trait::async_trait]
impl EmbeddingExecutor for HttpEmbeddingExecutor {
    async fn execute(&self, req: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        // Capability guard to avoid calling unimplemented ProviderSpec defaults
        let caps = self.provider_spec.capabilities();
        if !caps.supports("embedding") {
            return Err(LlmError::UnsupportedOperation(
                "Embedding is not supported by this provider".to_string(),
            ));
        }
        // 1. Transform request to JSON
        let mut body = self.request_transformer.transform_embedding(&req)?;

        // 2. Apply before_send hook if present
        if let Some(cb) = &self.before_send {
            body = cb(&body)?;
        }

        // 3. Get URL from provider spec
        let url = self
            .provider_spec
            .embedding_url(&req, &self.provider_context);

        // 4. Build execution config for common HTTP layer
        let config = crate::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };

        // 5. Execute request using common HTTP layer
        let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
        let result = crate::executors::common::execute_json_request(
            &config,
            &url,
            crate::executors::common::HttpBody::Json(body),
            per_request_headers,
            false, // stream = false for embedding
        )
        .await?;

        // 6. Transform response
        self.response_transformer
            .transform_embedding_response(&result.json)
    }
}
