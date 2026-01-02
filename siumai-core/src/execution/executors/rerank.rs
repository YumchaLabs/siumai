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
    pub provider_spec: Arc<dyn crate::core::ProviderSpec>,
    pub provider_context: crate::core::ProviderContext,
    /// Execution policy
    pub policy: crate::execution::ExecutionPolicy,
}

/// Builder for creating HttpRerankExecutor instances
pub struct RerankExecutorBuilder {
    provider_id: String,
    http_client: reqwest::Client,
    spec: Option<Arc<dyn crate::core::ProviderSpec>>,
    context: Option<crate::core::ProviderContext>,
    request_transformer: Option<Arc<dyn RerankRequestTransformer>>,
    response_transformer: Option<Arc<dyn RerankResponseTransformer>>,
    policy: crate::execution::ExecutionPolicy,
}

impl RerankExecutorBuilder {
    pub fn new(provider_id: impl Into<String>, http_client: reqwest::Client) -> Self {
        Self {
            provider_id: provider_id.into(),
            http_client,
            spec: None,
            context: None,
            request_transformer: None,
            response_transformer: None,
            policy: crate::execution::ExecutionPolicy::new(),
        }
    }

    pub fn with_spec(mut self, spec: Arc<dyn crate::core::ProviderSpec>) -> Self {
        self.spec = Some(spec);
        self
    }

    pub fn with_context(mut self, context: crate::core::ProviderContext) -> Self {
        self.context = Some(context);
        self
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

    pub fn with_before_send(mut self, hook: crate::execution::executors::BeforeSendHook) -> Self {
        self.policy.before_send = Some(hook);
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
        let spec = self.spec.expect("provider_spec is required");
        let context = self.context.expect("provider_context is required");
        Arc::new(HttpRerankExecutor {
            provider_id: self.provider_id,
            http_client: self.http_client,
            request_transformer: self
                .request_transformer
                .expect("rerank request transformer is required"),
            response_transformer: self
                .response_transformer
                .expect("rerank response transformer is required"),
            provider_spec: spec,
            provider_context: context,
            policy: self.policy,
        })
    }

    /// Build the executor, selecting transformers from ProviderSpec if they were not set.
    ///
    /// This matches the `*_ExecutorBuilder::build_for_request` pattern used by other executors.
    ///
    /// # Panics
    /// Panics if required fields are not set.
    pub fn build_for_request(mut self, request: &RerankRequest) -> Arc<HttpRerankExecutor> {
        let spec = self.spec.take().expect("provider_spec is required");
        let ctx = self.context.take().expect("provider_context is required");

        if self.request_transformer.is_none() || self.response_transformer.is_none() {
            let bundle = spec.choose_rerank_transformers(request, &ctx);
            self.request_transformer = Some(bundle.request);
            self.response_transformer = Some(bundle.response);
        }

        // Restore spec/context: url/headers are computed via ProviderSpec at execution time.
        self.spec = Some(spec);
        self.context = Some(ctx);
        self.build()
    }
}

#[async_trait::async_trait]
impl RerankExecutor for HttpRerankExecutor {
    async fn execute(&self, req: RerankRequest) -> Result<RerankResponse, LlmError> {
        // Capability guard to avoid calling unimplemented ProviderSpec defaults
        let caps = self.provider_spec.capabilities();
        if !caps.supports("rerank") {
            return Err(LlmError::UnsupportedOperation(
                "Rerank is not supported by this provider".to_string(),
            ));
        }

        let retry_options = self.policy.retry_options.clone();
        let run_once = move || {
            let req = req.clone();
            async move {
                // 1. Transform request
                let mut body = self.request_transformer.transform(&req)?;

                // 2. Apply ProviderSpec-level hook, then policy hook (mirrors EmbeddingExecutor)
                if let Some(cb) = self
                    .provider_spec
                    .rerank_before_send(&req, &self.provider_context)
                {
                    body = cb(&body)?;
                }
                if let Some(cb) = &self.policy.before_send {
                    body = cb(&body)?;
                }

                // 3. Resolve URL via ProviderSpec
                let url = self.provider_spec.rerank_url(&req, &self.provider_context);

                // 4. Execute request via the common HTTP layer
                let config = crate::execution::executors::common::HttpExecutionConfig {
                    provider_id: self.provider_id.clone(),
                    http_client: self.http_client.clone(),
                    provider_spec: self.provider_spec.clone(),
                    provider_context: self.provider_context.clone(),
                    interceptors: self.policy.interceptors.clone(),
                    retry_options: self.policy.retry_options.clone(),
                };
                let result = crate::execution::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(body),
                    None,
                    false,
                )
                .await?;

                // 5. Transform response
                self.response_transformer.transform(result.json)
            }
        };

        if let Some(opts) = retry_options {
            crate::retry_api::retry_with(run_once, opts).await
        } else {
            run_once().await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderMap;
    use std::collections::HashMap;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[derive(Clone, Copy)]
    struct NoRerankSpec;
    impl crate::core::ProviderSpec for NoRerankSpec {
        fn id(&self) -> &'static str {
            "test"
        }
        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            crate::traits::ProviderCapabilities::new()
        }
        fn build_headers(
            &self,
            _ctx: &crate::core::ProviderContext,
        ) -> Result<HeaderMap, LlmError> {
            Ok(HeaderMap::new())
        }
        fn chat_url(
            &self,
            _stream: bool,
            _req: &crate::types::ChatRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> String {
            unreachable!()
        }
        fn choose_chat_transformers(
            &self,
            _req: &crate::types::ChatRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> crate::core::ChatTransformers {
            unreachable!()
        }
    }

    #[derive(Clone, Copy)]
    struct RerankSpec;
    impl crate::core::ProviderSpec for RerankSpec {
        fn id(&self) -> &'static str {
            "test"
        }
        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            crate::traits::ProviderCapabilities::new().with_rerank()
        }
        fn build_headers(
            &self,
            _ctx: &crate::core::ProviderContext,
        ) -> Result<HeaderMap, LlmError> {
            Ok(HeaderMap::new())
        }
        fn chat_url(
            &self,
            _stream: bool,
            _req: &crate::types::ChatRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> String {
            unreachable!()
        }
        fn choose_chat_transformers(
            &self,
            _req: &crate::types::ChatRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> crate::core::ChatTransformers {
            unreachable!()
        }
    }

    struct MarkingReqTx {
        called: Arc<AtomicBool>,
        error: Option<LlmError>,
    }
    impl crate::execution::transformers::rerank_request::RerankRequestTransformer for MarkingReqTx {
        fn transform(&self, _req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
            self.called.store(true, Ordering::SeqCst);
            if let Some(e) = self.error.clone() {
                return Err(e);
            }
            Ok(serde_json::json!({}))
        }
    }

    struct NoopRespTx;
    impl crate::execution::transformers::rerank_response::RerankResponseTransformer for NoopRespTx {
        fn transform(&self, _raw: serde_json::Value) -> Result<RerankResponse, LlmError> {
            Err(LlmError::InvalidParameter("abort".into()))
        }
    }

    fn ctx() -> crate::core::ProviderContext {
        crate::core::ProviderContext::new("test", "http://example.invalid", None, HashMap::new())
    }

    #[tokio::test]
    async fn rerank_guard_blocks_unsupported_provider_before_transform() {
        let called = Arc::new(AtomicBool::new(false));
        let exec = RerankExecutorBuilder::new("test", reqwest::Client::new())
            .with_spec(Arc::new(NoRerankSpec))
            .with_context(ctx())
            .with_transformers(
                Arc::new(MarkingReqTx {
                    called: called.clone(),
                    error: None,
                }),
                Arc::new(NoopRespTx),
            )
            .build();

        let req = RerankRequest::new(
            "model".to_string(),
            "q".to_string(),
            vec!["doc".to_string()],
        );
        let err = exec.execute(req).await.unwrap_err();

        assert!(
            matches!(err, LlmError::UnsupportedOperation(_)),
            "expected UnsupportedOperation, got {err:?}"
        );
        assert!(
            !called.load(Ordering::SeqCst),
            "request transformer should not run when rerank is unsupported"
        );
    }

    #[tokio::test]
    async fn rerank_guard_allows_supported_provider_and_calls_transformer() {
        let called = Arc::new(AtomicBool::new(false));
        let exec = RerankExecutorBuilder::new("test", reqwest::Client::new())
            .with_spec(Arc::new(RerankSpec))
            .with_context(ctx())
            .with_transformers(
                Arc::new(MarkingReqTx {
                    called: called.clone(),
                    error: Some(LlmError::InvalidParameter("expected".into())),
                }),
                Arc::new(NoopRespTx),
            )
            .build();

        let req = RerankRequest::new(
            "model".to_string(),
            "q".to_string(),
            vec!["doc".to_string()],
        );
        let err = exec.execute(req).await.unwrap_err();

        assert!(
            matches!(err, LlmError::InvalidParameter(_)),
            "expected transformer error, got {err:?}"
        );
        assert!(
            called.load(Ordering::SeqCst),
            "request transformer should run when rerank is supported"
        );
    }
}
