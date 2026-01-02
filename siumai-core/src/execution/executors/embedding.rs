//! Embedding executor traits

use crate::error::LlmError;
use crate::execution::transformers::{request::RequestTransformer, response::ResponseTransformer};
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
    /// Execution policy
    pub policy: crate::execution::ExecutionPolicy,
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
        let retry_options = self.policy.retry_options.clone();
        let run_once = move || {
            let req = req.clone();
            async move {
                // 1. Transform request to JSON
                let mut body = self.request_transformer.transform_embedding(&req)?;

                // 2. Apply ProviderSpec-level embedding_before_send, then policy hook
                if let Some(cb) = self
                    .provider_spec
                    .embedding_before_send(&req, &self.provider_context)
                {
                    body = cb(&body)?;
                }
                if let Some(cb) = &self.policy.before_send {
                    body = cb(&body)?;
                }

                // 3. Get URL from provider spec
                let url = self
                    .provider_spec
                    .embedding_url(&req, &self.provider_context);

                // 4. Build execution config for common HTTP layer
                let config = crate::execution::executors::common::HttpExecutionConfig {
                    provider_id: self.provider_id.clone(),
                    http_client: self.http_client.clone(),
                    provider_spec: self.provider_spec.clone(),
                    provider_context: self.provider_context.clone(),
                    interceptors: self.policy.interceptors.clone(),
                    retry_options: self.policy.retry_options.clone(),
                };

                // 5. Execute request using common HTTP layer
                let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
                let result = crate::execution::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(body),
                    per_request_headers,
                    false, // stream = false for embedding
                )
                .await?;

                // 6. Transform response
                self.response_transformer
                    .transform_embedding_response(&result.json)
            }
        };

        if let Some(opts) = retry_options {
            crate::retry_api::retry_with(run_once, opts).await
        } else {
            run_once().await
        }
    }
}

/// Builder for creating HttpEmbeddingExecutor instances
///
/// Mirrors ChatExecutorBuilder to provide a consistent construction API
/// for non-streaming embedding executors.
pub struct EmbeddingExecutorBuilder {
    provider_id: String,
    http_client: reqwest::Client,
    spec: Option<Arc<dyn crate::core::ProviderSpec>>,
    context: Option<crate::core::ProviderContext>,
    request_transformer: Option<Arc<dyn RequestTransformer>>,
    response_transformer: Option<Arc<dyn ResponseTransformer>>,
    policy: crate::execution::ExecutionPolicy,
}

impl EmbeddingExecutorBuilder {
    /// Create a new builder with required fields
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

    /// Set the provider spec
    pub fn with_spec(mut self, spec: Arc<dyn crate::core::ProviderSpec>) -> Self {
        self.spec = Some(spec);
        self
    }

    /// Set the provider context
    pub fn with_context(mut self, context: crate::core::ProviderContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Set the transformers
    pub fn with_transformers(
        mut self,
        request: Arc<dyn RequestTransformer>,
        response: Arc<dyn ResponseTransformer>,
    ) -> Self {
        self.request_transformer = Some(request);
        self.response_transformer = Some(response);
        self
    }

    /// Set the before_send hook
    pub fn with_before_send(mut self, hook: crate::execution::executors::BeforeSendHook) -> Self {
        self.policy.before_send = Some(hook);
        self
    }

    /// Set HTTP interceptors
    pub fn with_interceptors(
        mut self,
        interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    ) -> Self {
        self.policy.interceptors = interceptors;
        self
    }

    /// Set retry options
    pub fn with_retry_options(mut self, retry_options: crate::retry_api::RetryOptions) -> Self {
        self.policy.retry_options = Some(retry_options);
        self
    }

    /// Build the HttpEmbeddingExecutor
    ///
    /// # Panics
    /// Panics if required fields (spec, context, transformers) are not set
    pub fn build(self) -> Arc<HttpEmbeddingExecutor> {
        Arc::new(HttpEmbeddingExecutor {
            provider_id: self.provider_id,
            http_client: self.http_client,
            request_transformer: self
                .request_transformer
                .expect("request_transformer is required"),
            response_transformer: self
                .response_transformer
                .expect("response_transformer is required"),
            provider_spec: self.spec.expect("provider_spec is required"),
            provider_context: self.context.expect("provider_context is required"),
            policy: self.policy,
        })
    }

    /// Build the executor, selecting transformers from the ProviderSpec if they were not set.
    ///
    /// This keeps providers from duplicating the `choose_embedding_transformers` wiring while still
    /// allowing providers to override transformers explicitly when needed (e.g. config-aware Gemini).
    ///
    /// # Panics
    /// Panics if required fields (spec, context) are not set.
    pub fn build_for_request(mut self, request: &EmbeddingRequest) -> Arc<HttpEmbeddingExecutor> {
        let spec = self.spec.take().expect("provider_spec is required");
        let ctx = self.context.take().expect("provider_context is required");

        if self.request_transformer.is_none() || self.response_transformer.is_none() {
            let bundle = spec.choose_embedding_transformers(request, &ctx);
            self.request_transformer = Some(bundle.request);
            self.response_transformer = Some(bundle.response);
        }

        self.spec = Some(spec);
        self.context = Some(ctx);
        self.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    use std::sync::{Arc, Mutex};

    // Minimal ProviderSpec for embedding
    #[derive(Clone, Copy)]
    struct TestSpec;
    impl crate::core::ProviderSpec for TestSpec {
        fn id(&self) -> &'static str {
            "test"
        }
        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            crate::traits::ProviderCapabilities::new().with_embedding()
        }
        fn build_headers(
            &self,
            _ctx: &crate::core::ProviderContext,
        ) -> Result<HeaderMap, LlmError> {
            let mut h = HeaderMap::new();
            h.insert(
                HeaderName::from_static("x-base"),
                HeaderValue::from_static("B"),
            );
            Ok(h)
        }
        fn chat_url(
            &self,
            _s: bool,
            _r: &crate::types::ChatRequest,
            _c: &crate::core::ProviderContext,
        ) -> String {
            unreachable!()
        }
        fn choose_chat_transformers(
            &self,
            _r: &crate::types::ChatRequest,
            _c: &crate::core::ProviderContext,
        ) -> crate::core::ChatTransformers {
            unreachable!()
        }
    }

    // No-op transformers
    struct EchoReq;
    impl crate::execution::transformers::request::RequestTransformer for EchoReq {
        fn provider_id(&self) -> &str {
            "test"
        }
        fn transform_chat(
            &self,
            _req: &crate::types::ChatRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({}))
        }
        fn transform_embedding(
            &self,
            req: &crate::types::EmbeddingRequest,
        ) -> Result<serde_json::Value, LlmError> {
            Ok(serde_json::json!({"model": req.model.clone().unwrap_or_default(), "mark": "orig"}))
        }
    }
    struct NoopResp;
    impl crate::execution::transformers::response::ResponseTransformer for NoopResp {
        fn provider_id(&self) -> &str {
            "test"
        }
        fn transform_embedding_response(
            &self,
            _raw: &serde_json::Value,
        ) -> Result<crate::types::EmbeddingResponse, LlmError> {
            Err(LlmError::InvalidParameter("abort".into()))
        }
    }

    // Interceptor capturing headers and abort
    struct CaptureHeaders {
        seen: Arc<Mutex<Option<HeaderMap>>>,
    }
    impl HttpInterceptor for CaptureHeaders {
        fn on_before_send(
            &self,
            _ctx: &HttpRequestContext,
            _rb: reqwest::RequestBuilder,
            _body: &serde_json::Value,
            headers: &HeaderMap,
        ) -> Result<reqwest::RequestBuilder, LlmError> {
            *self.seen.lock().unwrap() = Some(headers.clone());
            Err(LlmError::InvalidParameter("abort".into()))
        }
    }

    #[tokio::test]
    async fn embedding_before_send_applied_and_aborts() {
        let http = reqwest::Client::new();
        let spec = Arc::new(TestSpec);
        let ctx =
            crate::core::ProviderContext::new("test", "http://127.0.0.1", None, Default::default());
        let exec = HttpEmbeddingExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer: Arc::new(EchoReq),
            response_transformer: Arc::new(NoopResp),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new().with_before_send(Arc::new(|body| {
                assert_eq!(body["mark"], serde_json::json!("orig"));
                Err(LlmError::InvalidParameter("abort".into()))
            })),
        };
        let req = crate::types::EmbeddingRequest::new(vec!["hi".into()]).with_model("m");
        let err = exec.execute(req).await.unwrap_err();
        match err {
            LlmError::InvalidParameter(m) => assert_eq!(m, "abort"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn embedding_interceptor_merges_headers() {
        let http = reqwest::Client::new();
        let spec = Arc::new(TestSpec);
        let ctx =
            crate::core::ProviderContext::new("test", "http://127.0.0.1", None, Default::default());
        let seen = Arc::new(Mutex::new(None::<HeaderMap>));
        let interceptor = Arc::new(CaptureHeaders { seen: seen.clone() });
        let exec = HttpEmbeddingExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer: Arc::new(EchoReq),
            response_transformer: Arc::new(NoopResp),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new().with_interceptors(vec![interceptor]),
        };
        let mut req = crate::types::EmbeddingRequest::new(vec!["hi".into()]).with_model("m");
        let mut hc = crate::types::HttpConfig::default();
        hc.headers.insert("x-req".into(), "R".into());
        req.http_config = Some(hc);
        let _ = exec.execute(req).await; // expect abort
        let headers = seen.lock().unwrap().clone().unwrap();
        assert_eq!(
            headers.get("x-base").unwrap(),
            &HeaderValue::from_static("B")
        );
        assert_eq!(
            headers.get("x-req").unwrap(),
            &HeaderValue::from_static("R")
        );
    }

    #[tokio::test]
    async fn spec_before_send_runs_before_policy_hook() {
        // Spec that sets mark:"spec" via embedding_before_send
        #[derive(Clone, Copy)]
        struct SpecWithHook;
        impl crate::core::ProviderSpec for SpecWithHook {
            fn id(&self) -> &'static str {
                "test"
            }
            fn capabilities(&self) -> crate::traits::ProviderCapabilities {
                crate::traits::ProviderCapabilities::new().with_embedding()
            }
            fn build_headers(
                &self,
                _ctx: &crate::core::ProviderContext,
            ) -> Result<HeaderMap, LlmError> {
                Ok(HeaderMap::new())
            }
            fn chat_url(
                &self,
                _s: bool,
                _r: &crate::types::ChatRequest,
                _c: &crate::core::ProviderContext,
            ) -> String {
                unreachable!()
            }
            fn choose_chat_transformers(
                &self,
                _r: &crate::types::ChatRequest,
                _c: &crate::core::ProviderContext,
            ) -> crate::core::ChatTransformers {
                unreachable!()
            }
            fn embedding_before_send(
                &self,
                _req: &EmbeddingRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> Option<crate::execution::executors::BeforeSendHook> {
                Some(Arc::new(|body: &serde_json::Value| {
                    let mut out = body.clone();
                    out["mark"] = serde_json::json!("spec");
                    Ok(out)
                }))
            }
        }

        // Request transformer sets mark:"orig"
        struct ReqTx;
        impl crate::execution::transformers::request::RequestTransformer for ReqTx {
            fn provider_id(&self) -> &str {
                "test"
            }
            fn transform_chat(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(serde_json::json!({}))
            }
            fn transform_embedding(
                &self,
                req: &EmbeddingRequest,
            ) -> Result<serde_json::Value, LlmError> {
                Ok(
                    serde_json::json!({"model": req.model.clone().unwrap_or_default(), "mark": "orig"}),
                )
            }
        }

        // Response transformer aborts, we only care about hooks order
        struct AbortResp;
        impl crate::execution::transformers::response::ResponseTransformer for AbortResp {
            fn provider_id(&self) -> &str {
                "test"
            }
            fn transform_embedding_response(
                &self,
                _raw: &serde_json::Value,
            ) -> Result<crate::types::EmbeddingResponse, LlmError> {
                Err(LlmError::InvalidParameter("abort".into()))
            }
        }

        // Policy hook captures body mark value then aborts
        let seen = Arc::new(Mutex::new(None::<String>));
        let seen2 = seen.clone();
        let policy_hook: crate::execution::executors::BeforeSendHook =
            Arc::new(move |body: &serde_json::Value| {
                *seen2.lock().unwrap() = body
                    .get("mark")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                Ok(body.clone())
            });

        let http = reqwest::Client::new();
        let spec = Arc::new(SpecWithHook);
        let ctx =
            crate::core::ProviderContext::new("test", "http://127.0.0.1", None, Default::default());
        let exec = super::HttpEmbeddingExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer: Arc::new(ReqTx),
            response_transformer: Arc::new(AbortResp),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new().with_before_send(policy_hook),
        };

        let _ = exec
            .execute(EmbeddingRequest::new(vec!["hello".into()]).with_model("m"))
            .await;
        let captured = seen.lock().unwrap().clone().unwrap();
        assert_eq!(captured, "spec");
    }
}
