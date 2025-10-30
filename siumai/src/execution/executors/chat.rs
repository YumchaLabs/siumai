//! Chat executor traits
#![allow(clippy::items_after_test_module)]
//!
//! Defines the abstraction that will drive chat operations using transformers
//! and HTTP. For now this is an interface stub for the refactor.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::{
    GenerateAsyncFn, LanguageModelMiddleware, StreamAsyncFn, apply_post_generate_chain,
    apply_transform_chain, try_pre_generate, try_pre_stream,
};
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::streaming::ChatStream;
// use crate::streaming::SseEventConverter; // not needed explicitly here
// Telemetry event emission is handled via execution::telemetry helpers
use crate::types::{ChatRequest, ChatResponse};
use std::sync::Arc;

// -----------------------------------------------------------------------------
// Module-scoped stream converter wrappers (SSE/JSON) to avoid inline duplicates
// -----------------------------------------------------------------------------

use crate::streaming::adapters::{
    InterceptingConverter, MiddlewareConverter, MiddlewareJsonConverter, TransformerConverter,
};

// -----------------------------------------------------------------------------
// Helper builders for SSE/JSON streaming sources (no behavior change)
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn create_sse_stream_with_middlewares(
    provider_id: String,
    url: String,
    http: reqwest::Client,
    headers_base: reqwest::header::HeaderMap,
    transformed: serde_json::Value,
    sse_tx: Arc<dyn crate::execution::transformers::stream::StreamChunkTransformer>,
    interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    middlewares: Vec<
        Arc<dyn crate::execution::middleware::language_model::LanguageModelMiddleware>,
    >,
    req_in: crate::types::ChatRequest,
    disable_compression: bool,
    retry_options: Option<crate::retry_api::RetryOptions>,
) -> Result<crate::streaming::ChatStream, LlmError> {
    let converter = TransformerConverter(sse_tx.clone());
    let mw_wrapped = MiddlewareConverter {
        middlewares: middlewares.clone(),
        req: req_in.clone(),
        convert: converter,
    };
    let intercepting = InterceptingConverter {
        interceptors: interceptors.clone(),
        ctx: crate::execution::http::interceptor::HttpRequestContext {
            provider_id: provider_id.clone(),
            url: url.clone(),
            stream: true,
        },
        convert: mw_wrapped,
    };
    // Prepare body (OpenAI stream flags) before calling common helper
    let mut body_for_send = transformed.clone();
    if provider_id.starts_with("openai") {
        body_for_send["stream"] = serde_json::Value::Bool(true);
        if body_for_send.get("stream_options").is_none() {
            body_for_send["stream_options"] = serde_json::json!({"include_usage": true});
        } else if let Some(obj) = body_for_send["stream_options"].as_object_mut() {
            obj.entry("include_usage")
                .or_insert(serde_json::Value::Bool(true));
        }
    }

    crate::execution::executors::common::execute_sse_stream_request_with_headers(
        &http,
        &provider_id,
        &url,
        headers_base,
        body_for_send,
        &interceptors,
        retry_options,
        req_in.http_config.as_ref().map(|hc| hc.headers.clone()),
        intercepting,
        disable_compression,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn create_json_stream_with_middlewares(
    provider_id: String,
    url: String,
    http: reqwest::Client,
    headers_base: reqwest::header::HeaderMap,
    transformed: serde_json::Value,
    json_conv: Arc<dyn crate::streaming::JsonEventConverter>,
    interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    middlewares: Vec<
        Arc<dyn crate::execution::middleware::language_model::LanguageModelMiddleware>,
    >,
    req_in: crate::types::ChatRequest,
    disable_compression: bool,
) -> Result<crate::streaming::ChatStream, LlmError> {
    let mw = MiddlewareJsonConverter {
        middlewares: middlewares.clone(),
        req: req_in.clone(),
        convert: json_conv.clone(),
    };
    let per_req_headers = req_in.http_config.as_ref().map(|hc| &hc.headers);
    crate::execution::executors::common::execute_json_stream_request_with_headers(
        &http,
        &provider_id,
        &url,
        headers_base,
        transformed,
        &interceptors,
        None,
        per_req_headers,
        mw,
        disable_compression,
    )
    .await
}

// -----------------------------------------------------------------------------
// Small helper to build chat JSON body with middlewares + before_send
// -----------------------------------------------------------------------------

fn build_chat_body(
    request_tx: &Arc<dyn RequestTransformer>,
    middlewares: &[Arc<dyn LanguageModelMiddleware>],
    before_send: &Option<crate::execution::executors::BeforeSendHook>,
    req_in: &crate::types::ChatRequest,
) -> Result<serde_json::Value, LlmError> {
    let mut body = request_tx.transform_chat(req_in)?;
    crate::execution::middleware::language_model::apply_json_body_transform_chain(
        middlewares,
        req_in,
        &mut body,
    )?;
    let out = if let Some(cb) = before_send {
        cb(&body)?
    } else {
        body
    };
    Ok(out)
}

#[async_trait::async_trait]
pub trait ChatExecutor: Send + Sync {
    async fn execute(&self, req: ChatRequest) -> Result<ChatResponse, LlmError>;
    async fn execute_stream(&self, req: ChatRequest) -> Result<ChatStream, LlmError>;
}

/// Generic HTTP-based ChatExecutor that wires transformers and HTTP
pub struct HttpChatExecutor {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub request_transformer: Arc<dyn RequestTransformer>,
    pub response_transformer: Arc<dyn ResponseTransformer>,
    pub stream_transformer: Option<Arc<dyn StreamChunkTransformer>>,
    /// Optional JSON streaming converter for providers that emit JSON lines
    pub json_stream_converter: Option<Arc<dyn crate::streaming::JsonEventConverter>>,
    /// Execution policy (interceptors/retry/before_send/stream flags)
    pub policy: crate::execution::ExecutionPolicy,
    /// Optional model-level middlewares (transform ChatRequest before mapping)
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Provider spec for building headers and URLs
    pub provider_spec: Arc<dyn crate::core::ProviderSpec>,
    /// Provider context for header/URL construction
    pub provider_context: crate::core::ProviderContext,
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use reqwest::header::HeaderMap;

    struct EchoRequestTransformer;
    impl crate::execution::transformers::request::RequestTransformer for EchoRequestTransformer {
        fn provider_id(&self) -> &str {
            "test"
        }
        fn transform_chat(
            &self,
            req: &crate::types::ChatRequest,
        ) -> Result<serde_json::Value, crate::error::LlmError> {
            Ok(serde_json::json!({
                "model": req.common_params.model,
                "messages_len": req.messages.len(),
            }))
        }
    }

    struct NoopResponseTransformer;
    impl crate::execution::transformers::response::ResponseTransformer for NoopResponseTransformer {
        fn provider_id(&self) -> &str {
            "test"
        }
    }

    // Test ProviderSpec
    #[derive(Clone, Copy)]
    struct TestProviderSpec;
    impl crate::core::ProviderSpec for TestProviderSpec {
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
            "http://127.0.0.1/never".to_string()
        }
        fn choose_chat_transformers(
            &self,
            _req: &crate::types::ChatRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> crate::core::ChatTransformers {
            crate::core::ChatTransformers {
                request: Arc::new(EchoRequestTransformer),
                response: Arc::new(NoopResponseTransformer),
                stream: None,
                json: None,
            }
        }
        fn chat_before_send(
            &self,
            _req: &crate::types::ChatRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> Option<crate::execution::executors::BeforeSendHook> {
            None
        }
    }

    #[tokio::test]
    async fn applies_model_middlewares_before_mapping() {
        struct AppendSuffix;
        impl crate::execution::middleware::language_model::LanguageModelMiddleware for AppendSuffix {
            fn transform_params(
                &self,
                mut req: crate::types::ChatRequest,
            ) -> crate::types::ChatRequest {
                req.common_params.model.push_str("-mw");
                req
            }
        }

        let http = reqwest::Client::new();
        let request_transformer = Arc::new(EchoRequestTransformer);
        let response_transformer = Arc::new(NoopResponseTransformer);

        let seen = Arc::new(std::sync::Mutex::new(None::<String>));
        let seen_clone = seen.clone();
        let hook: crate::execution::executors::BeforeSendHook =
            Arc::new(move |body: &serde_json::Value| {
                let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("");
                *seen_clone.lock().unwrap() = Some(model.to_string());
                Err(crate::error::LlmError::InvalidParameter("abort".into()))
            });

        let provider_context = crate::core::ProviderContext::new(
            "test",
            "http://127.0.0.1",
            None,
            std::collections::HashMap::new(),
        );
        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: None,
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new()
                .with_stream_disable_compression(true)
                .with_retry_options(None),
            middlewares: vec![Arc::new(AppendSuffix)],
            provider_spec: Arc::new(TestProviderSpec),
            provider_context,
            // attach before_send via policy; create a new executor with updated policy
        };

        // Rebuild with before_send hook (immutable struct fields)
        let exec = HttpChatExecutor {
            policy: exec.policy.clone().with_before_send(hook),
            ..exec
        };

        let mut req = crate::types::ChatRequest::new(vec![]);
        req.common_params.model = "base".to_string();
        let err = exec.execute(req).await.unwrap_err();
        match err {
            crate::error::LlmError::InvalidParameter(msg) => assert_eq!(msg, "abort"),
            other => panic!("unexpected error: {:?}", other),
        }
        let got = seen.lock().unwrap().clone().unwrap();
        assert_eq!(got, "base-mw");
    }

    #[tokio::test]
    async fn wrap_stream_async_orders_and_short_circuits() {
        use futures::StreamExt;

        // Dummy stream transformer (unused due to short-circuit)
        struct DummyStreamTx;
        impl crate::execution::transformers::stream::StreamChunkTransformer for DummyStreamTx {
            fn provider_id(&self) -> &str {
                "test"
            }
            fn convert_event(
                &self,
                _event: eventsource_stream::Event,
            ) -> crate::execution::transformers::stream::StreamEventFuture<'_> {
                Box::pin(async { Vec::new() })
            }
        }

        // Outer appends suffix to final StreamEnd
        struct Outer;
        impl crate::execution::middleware::language_model::LanguageModelMiddleware for Outer {
            fn wrap_stream_async(
                &self,
                next: std::sync::Arc<crate::execution::middleware::language_model::StreamAsyncFn>,
            ) -> std::sync::Arc<crate::execution::middleware::language_model::StreamAsyncFn>
            {
                std::sync::Arc::new(move |req: crate::types::ChatRequest| {
                    let next = next.clone();
                    Box::pin(async move {
                        let s = next(req).await?;
                        let mapped = s.map(|res| {
                            res.map(|ev| match ev {
                                crate::types::ChatStreamEvent::StreamEnd { mut response } => {
                                    if let Some(t) = response.content_text() {
                                        response.content = crate::types::MessageContent::Text(
                                            format!("{}-outer", t),
                                        );
                                    }
                                    crate::types::ChatStreamEvent::StreamEnd { response }
                                }
                                other => other,
                            })
                        });
                        Ok(Box::pin(mapped) as crate::streaming::ChatStream)
                    })
                })
            }
        }

        // Inner short-circuits and returns a synthetic stream
        struct Inner;
        impl crate::execution::middleware::language_model::LanguageModelMiddleware for Inner {
            fn wrap_stream_async(
                &self,
                _next: std::sync::Arc<crate::execution::middleware::language_model::StreamAsyncFn>,
            ) -> std::sync::Arc<crate::execution::middleware::language_model::StreamAsyncFn>
            {
                std::sync::Arc::new(|_req: crate::types::ChatRequest| {
                    Box::pin(async move {
                        let one = futures::stream::once(async move {
                            Ok(crate::types::ChatStreamEvent::StreamEnd {
                                response: crate::types::ChatResponse::new(
                                    crate::types::MessageContent::Text("inner".into()),
                                ),
                            })
                        });
                        Ok(Box::pin(one) as crate::streaming::ChatStream)
                    })
                })
            }
        }

        let http = reqwest::Client::new();
        let request_transformer = std::sync::Arc::new(EchoRequestTransformer);
        let response_transformer = std::sync::Arc::new(NoopResponseTransformer);

        let provider_context = crate::core::ProviderContext::new(
            "test",
            "http://127.0.0.1",
            None,
            std::collections::HashMap::new(),
        );
        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: Some(std::sync::Arc::new(DummyStreamTx)),
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new().with_stream_disable_compression(true),
            middlewares: vec![std::sync::Arc::new(Outer), std::sync::Arc::new(Inner)],
            provider_spec: Arc::new(TestProviderSpec),
            provider_context,
        };

        let req = crate::types::ChatRequest::new(vec![]);
        let stream = exec
            .execute_stream(req)
            .await
            .expect("stream should short-circuit");
        let events: Vec<_> = stream.collect().await;
        let end_text = events.iter().find_map(|e| match e {
            Ok(crate::types::ChatStreamEvent::StreamEnd { response }) => {
                response.content_text().map(|s| s.to_string())
            }
            _ => None,
        });
        assert_eq!(end_text.unwrap_or_default(), "inner-outer");
    }
    #[tokio::test]
    async fn wrap_generate_async_orders_and_short_circuits() {
        use crate::execution::middleware::language_model::GenerateAsyncFn;
        use futures::future::BoxFuture;
        use std::sync::Arc;

        struct Outer;
        impl crate::execution::middleware::language_model::LanguageModelMiddleware for Outer {
            fn wrap_generate_async(&self, next: Arc<GenerateAsyncFn>) -> Arc<GenerateAsyncFn> {
                Arc::new(
                    move |req: crate::types::ChatRequest| -> BoxFuture<
                        'static,
                        Result<crate::types::ChatResponse, crate::error::LlmError>,
                    > {
                        let next = next.clone();
                        Box::pin(async move {
                            let mut resp = next(req).await?;
                            if let Some(t) = resp.content_text() {
                                resp.content =
                                    crate::types::MessageContent::Text(format!("{}-outer", t));
                            }
                            Ok(resp)
                        })
                    },
                )
            }
        }

        struct Inner;
        impl crate::execution::middleware::language_model::LanguageModelMiddleware for Inner {
            fn wrap_generate_async(&self, _next: Arc<GenerateAsyncFn>) -> Arc<GenerateAsyncFn> {
                Arc::new(
                    |_req: crate::types::ChatRequest| -> BoxFuture<
                        'static,
                        Result<crate::types::ChatResponse, crate::error::LlmError>,
                    > {
                        Box::pin(async move {
                            Ok(crate::types::ChatResponse::new(
                                crate::types::MessageContent::Text("inner".into()),
                            ))
                        })
                    },
                )
            }
        }

        let http = reqwest::Client::new();
        let request_transformer = Arc::new(EchoRequestTransformer);
        let response_transformer = Arc::new(NoopResponseTransformer);

        let provider_context = crate::core::ProviderContext::new(
            "test",
            "http://127.0.0.1",
            None,
            std::collections::HashMap::new(),
        );
        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: None,
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new().with_stream_disable_compression(true),
            middlewares: vec![Arc::new(Outer), Arc::new(Inner)],
            provider_spec: Arc::new(TestProviderSpec),
            provider_context,
        };

        let req = crate::types::ChatRequest::new(vec![]);
        let resp = exec
            .execute(req)
            .await
            .expect("wrapped should short-circuit");
        assert_eq!(resp.content_text().unwrap_or_default(), "inner-outer");
    }

    #[tokio::test]
    async fn pre_generate_short_circuits_before_http() {
        // Middleware that short-circuits generate and returns a fixed response
        struct PreMw;
        impl crate::execution::middleware::language_model::LanguageModelMiddleware for PreMw {
            fn pre_generate(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Option<Result<crate::types::ChatResponse, crate::error::LlmError>> {
                Some(Ok(crate::types::ChatResponse::new(
                    crate::types::MessageContent::Text("mw".to_string()),
                )))
            }
        }

        let http = reqwest::Client::new();
        let request_transformer = Arc::new(EchoRequestTransformer);
        let response_transformer = Arc::new(NoopResponseTransformer);

        // Hook that would abort if HTTP is attempted
        let hook: crate::execution::executors::BeforeSendHook =
            Arc::new(move |_body: &serde_json::Value| {
                Err(crate::error::LlmError::InvalidParameter(
                    "should not be called".into(),
                ))
            });

        let provider_context = crate::core::ProviderContext::new(
            "test",
            "http://127.0.0.1",
            None,
            std::collections::HashMap::new(),
        );
        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: None,
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new()
                .with_stream_disable_compression(true)
                .with_before_send(hook),
            middlewares: vec![Arc::new(PreMw)],
            provider_spec: Arc::new(TestProviderSpec),
            provider_context,
        };

        let req = crate::types::ChatRequest::new(vec![]);
        let resp = exec
            .execute(req)
            .await
            .expect("pre mw should short-circuit");
        assert_eq!(resp.content_text().unwrap_or_default(), "mw");
    }

    #[tokio::test]
    async fn pre_stream_short_circuits_before_http() {
        // Middleware that short-circuits stream and returns a fixed one-shot stream
        struct PreMwStream;
        impl crate::execution::middleware::language_model::LanguageModelMiddleware for PreMwStream {
            fn pre_stream(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Option<Result<crate::streaming::ChatStream, crate::error::LlmError>> {
                let end = crate::types::ChatStreamEvent::StreamEnd {
                    response: crate::types::ChatResponse::new(crate::types::MessageContent::Text(
                        "mw-stream".to_string(),
                    )),
                };
                let s = futures::stream::iter(vec![Ok(end)]);
                Some(Ok(Box::pin(s)))
            }
        }

        // Dummy stream transformer (won't be used due to pre short-circuit)
        struct DummyTx;
        impl crate::execution::transformers::stream::StreamChunkTransformer for DummyTx {
            fn provider_id(&self) -> &str {
                "test"
            }
            fn convert_event(
                &self,
                _event: eventsource_stream::Event,
            ) -> crate::execution::transformers::stream::StreamEventFuture<'_> {
                Box::pin(async { vec![] })
            }
        }

        let http = reqwest::Client::new();
        let request_transformer = Arc::new(EchoRequestTransformer);
        let response_transformer = Arc::new(NoopResponseTransformer);

        let provider_context = crate::core::ProviderContext::new(
            "test",
            "http://127.0.0.1",
            None,
            std::collections::HashMap::new(),
        );
        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: Some(Arc::new(DummyTx)),
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new().with_stream_disable_compression(true),
            middlewares: vec![Arc::new(PreMwStream)],
            provider_spec: Arc::new(TestProviderSpec),
            provider_context,
        };

        let req = crate::types::ChatRequest::new(vec![]);
        let stream = exec
            .execute_stream(req)
            .await
            .expect("pre mw should short-circuit stream");
        let events: Vec<_> = stream.collect().await;
        assert!(events.iter().any(|e| match e {
            Ok(crate::types::ChatStreamEvent::StreamEnd { response }) => {
                response.content_text().unwrap_or_default() == "mw-stream"
            }
            _ => false,
        }));
    }

    // Interceptor that captures headers and aborts before network
    struct CaptureHeadersInterceptor {
        seen: std::sync::Arc<std::sync::Mutex<Option<reqwest::header::HeaderMap>>>,
    }
    impl crate::execution::http::interceptor::HttpInterceptor for CaptureHeadersInterceptor {
        fn on_before_send(
            &self,
            _ctx: &crate::execution::http::interceptor::HttpRequestContext,
            _rb: reqwest::RequestBuilder,
            _body: &serde_json::Value,
            headers: &reqwest::header::HeaderMap,
        ) -> Result<reqwest::RequestBuilder, LlmError> {
            *self.seen.lock().unwrap() = Some(headers.clone());
            Err(LlmError::InvalidParameter("abort".into()))
        }
    }

    #[tokio::test]
    async fn merges_request_headers_into_base_nonstream() {
        // Custom spec with base headers
        #[derive(Clone, Copy)]
        struct TestSpecWithHeaders;
        impl crate::core::ProviderSpec for TestSpecWithHeaders {
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
                let mut h = reqwest::header::HeaderMap::new();
                h.insert(
                    reqwest::header::HeaderName::from_static("x-base"),
                    reqwest::header::HeaderValue::from_static("base"),
                );
                Ok(h)
            }
            fn chat_url(
                &self,
                _stream: bool,
                _req: &crate::types::ChatRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> String {
                "http://127.0.0.1/never".to_string()
            }
            fn choose_chat_transformers(
                &self,
                _req: &crate::types::ChatRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> crate::core::ChatTransformers {
                crate::core::ChatTransformers {
                    request: Arc::new(EchoRequestTransformer),
                    response: Arc::new(NoopResponseTransformer),
                    stream: None,
                    json: None,
                }
            }
            fn chat_before_send(
                &self,
                _req: &crate::types::ChatRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> Option<crate::execution::executors::BeforeSendHook> {
                None
            }
        }

        let http = reqwest::Client::new();
        let request_transformer = Arc::new(EchoRequestTransformer);
        let response_transformer = Arc::new(NoopResponseTransformer);
        let seen = Arc::new(std::sync::Mutex::new(None));
        let interceptor = CaptureHeadersInterceptor { seen: seen.clone() };
        let provider_context = crate::core::ProviderContext::new(
            "test",
            "http://127.0.0.1",
            None,
            std::collections::HashMap::new(),
        );
        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: None,
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new()
                .with_stream_disable_compression(true)
                .with_interceptors(vec![Arc::new(interceptor)]),
            middlewares: vec![],
            provider_spec: Arc::new(TestSpecWithHeaders),
            provider_context,
        };

        let mut req = crate::types::ChatRequest::new(vec![]);
        // per-request header
        let mut hc = crate::types::HttpConfig::default();
        hc.headers.insert("X-Req".to_string(), "req".to_string());
        req.http_config = Some(hc);
        let _ = exec.execute(req).await;
        let captured = seen.lock().unwrap().clone().expect("headers captured");
        assert_eq!(
            captured
                .get("x-base")
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "base"
        );
        assert_eq!(
            captured
                .get("x-req")
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "req"
        );
    }

    #[tokio::test]
    async fn merges_request_headers_into_base_stream() {
        // Dummy stream transformer
        struct DummyTx;
        impl crate::execution::transformers::stream::StreamChunkTransformer for DummyTx {
            fn provider_id(&self) -> &str {
                "test"
            }
            fn convert_event(
                &self,
                _event: eventsource_stream::Event,
            ) -> crate::execution::transformers::stream::StreamEventFuture<'_> {
                Box::pin(async { vec![] })
            }
        }
        // Custom spec with base headers
        #[derive(Clone, Copy)]
        struct TestSpecWithHeaders;
        impl crate::core::ProviderSpec for TestSpecWithHeaders {
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
                let mut h = reqwest::header::HeaderMap::new();
                h.insert(
                    reqwest::header::HeaderName::from_static("x-base"),
                    reqwest::header::HeaderValue::from_static("base"),
                );
                Ok(h)
            }
            fn chat_url(
                &self,
                _stream: bool,
                _req: &crate::types::ChatRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> String {
                "http://127.0.0.1/never".to_string()
            }
            fn choose_chat_transformers(
                &self,
                _req: &crate::types::ChatRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> crate::core::ChatTransformers {
                crate::core::ChatTransformers {
                    request: Arc::new(EchoRequestTransformer),
                    response: Arc::new(NoopResponseTransformer),
                    stream: None,
                    json: None,
                }
            }
            fn chat_before_send(
                &self,
                _req: &crate::types::ChatRequest,
                _ctx: &crate::core::ProviderContext,
            ) -> Option<crate::execution::executors::BeforeSendHook> {
                None
            }
        }

        let http = reqwest::Client::new();
        let request_transformer = Arc::new(EchoRequestTransformer);
        let response_transformer = Arc::new(NoopResponseTransformer);
        let seen = Arc::new(std::sync::Mutex::new(None));
        let interceptor = CaptureHeadersInterceptor { seen: seen.clone() };
        let provider_context = crate::core::ProviderContext::new(
            "test",
            "http://127.0.0.1",
            None,
            std::collections::HashMap::new(),
        );
        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: Some(Arc::new(DummyTx)),
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new()
                .with_stream_disable_compression(true)
                .with_interceptors(vec![Arc::new(interceptor)]),
            middlewares: vec![],
            provider_spec: Arc::new(TestSpecWithHeaders),
            provider_context,
        };
        let mut req = crate::types::ChatRequest::new(vec![]);
        let mut hc = crate::types::HttpConfig::default();
        hc.headers.insert("X-Req".to_string(), "req".to_string());
        req.http_config = Some(hc);
        let _ = exec.execute_stream(req).await;
        let captured = seen.lock().unwrap().clone().expect("headers captured");
        assert_eq!(
            captured
                .get("x-base")
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "base"
        );
        assert_eq!(
            captured
                .get("x-req")
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "req"
        );
    }
}

#[async_trait::async_trait]
impl ChatExecutor for HttpChatExecutor {
    async fn execute(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        // Initialize telemetry if enabled
        let trace_id = uuid::Uuid::new_v4().to_string();
        let span_id = uuid::Uuid::new_v4().to_string();
        let start_time = std::time::SystemTime::now();
        let telemetry_config = req.telemetry.clone();

        crate::execution::telemetry::chat::span_start(
            telemetry_config.as_ref(),
            &trace_id,
            &span_id,
            &self.provider_id,
            &req.common_params.model,
            false,
        )
        .await;

        // Apply model-level parameter transforms
        let req = apply_transform_chain(&self.middlewares, req);
        // Try pre-generate short-circuit
        if let Some(decision) = try_pre_generate(&self.middlewares, &req) {
            // Emit telemetry span end for short-circuit
            crate::execution::telemetry::chat::span_end_ok(
                telemetry_config.as_ref(),
                &trace_id,
                &span_id,
                true,
                None,
            )
            .await;
            return decision;
        }

        // Prepare owned dependencies for the async base closure
        let provider_id = self.provider_id.clone();
        let provider_id_for_telemetry = provider_id.clone(); // Clone for telemetry use later
        let client = self.http_client.clone();
        let request_tx = self.request_transformer.clone();
        let response_tx = self.response_transformer.clone();
        let interceptors = self.policy.interceptors.clone();
        let before_send = self.policy.before_send.clone();
        let middlewares = self.middlewares.clone();
        // Pre-compute URL (provider/base-level). Request-level headers are merged later per-request.
        let url = self
            .provider_spec
            .chat_url(false, &req, &self.provider_context);
        let provider_spec = self.provider_spec.clone();
        let provider_context = self.provider_context.clone();
        let retry_options = self.policy.retry_options.clone();

        // Base async generator (no post_generate here)
        let base: Arc<GenerateAsyncFn> = Arc::new(move |req_in: ChatRequest| {
            let url = url.clone();
            let client = client.clone();
            let request_tx = request_tx.clone();
            let response_tx = response_tx.clone();
            let interceptors = interceptors.clone();
            let before_send = before_send.clone();
            let provider_id = provider_id.clone();
            let provider_spec = provider_spec.clone();
            let provider_context = provider_context.clone();
            let middlewares = middlewares.clone();
            let retry_options = retry_options.clone();
            Box::pin({
                async move {
                    let json_body =
                        build_chat_body(&request_tx, &middlewares, &before_send, &req_in)?;

                    let config = crate::execution::executors::common::HttpExecutionConfig {
                        provider_id: provider_id.clone(),
                        http_client: client.clone(),
                        provider_spec: provider_spec.clone(),
                        provider_context: provider_context.clone(),
                        interceptors: interceptors.clone(),
                        retry_options: retry_options.clone(),
                    };
                    let per_request_headers = req_in.http_config.as_ref().map(|hc| &hc.headers);
                    let result = crate::execution::executors::common::execute_json_request(
                        &config,
                        &url,
                        crate::execution::executors::common::HttpBody::Json(json_body),
                        per_request_headers,
                        false,
                    )
                    .await?;
                    let resp = response_tx.transform_chat_response(&result.json)?;
                    Ok(resp)
                }
            })
        });

        // Build around-style async wrappers in order (first registered becomes outermost)
        let wrapped = self
            .middlewares
            .iter()
            .rev()
            .fold(base, |next, mw| mw.wrap_generate_async(next));

        // Execute wrapped pipeline
        let result = wrapped(req.clone()).await;

        // Emit telemetry events
        match &result {
            Ok(response) => {
                crate::execution::telemetry::chat::span_end_ok(
                    telemetry_config.as_ref(),
                    &trace_id,
                    &span_id,
                    false,
                    response.finish_reason.as_ref(),
                )
                .await;
                crate::execution::telemetry::chat::generation(
                    telemetry_config.as_ref(),
                    &trace_id,
                    &provider_id_for_telemetry,
                    &req.common_params.model,
                    &req,
                    response,
                    start_time,
                )
                .await;
            }
            Err(error) => {
                crate::execution::telemetry::chat::span_end_err(
                    telemetry_config.as_ref(),
                    &trace_id,
                    &span_id,
                    error,
                )
                .await;
            }
        }

        // Apply post-generate processors
        let resp = result?;
        apply_post_generate_chain(&self.middlewares, &req, resp)
    }

    async fn execute_stream(&self, req: ChatRequest) -> Result<ChatStream, LlmError> {
        // Initialize telemetry if enabled
        let trace_id = uuid::Uuid::new_v4().to_string();
        let span_id = uuid::Uuid::new_v4().to_string();
        let _start_time = std::time::SystemTime::now();
        let telemetry_config = req.telemetry.clone();

        crate::execution::telemetry::chat::span_start_stream(
            telemetry_config.as_ref(),
            &trace_id,
            &span_id,
            &self.provider_id,
            &req.common_params.model,
        )
        .await;

        let sse_tx_opt = self.stream_transformer.clone();
        let json_tx_opt = self.json_stream_converter.clone();
        if sse_tx_opt.is_none() && json_tx_opt.is_none() {
            return Err(LlmError::UnsupportedOperation(
                "Streaming not supported by this executor".into(),
            ));
        }
        // Apply model-level parameter transforms
        let req = apply_transform_chain(&self.middlewares, req);
        // Try pre-stream short-circuit
        if let Some(decision) = try_pre_stream(&self.middlewares, &req) {
            // Emit telemetry span end for short-circuit
            crate::execution::telemetry::chat::span_end_ok_stream(
                telemetry_config.as_ref(),
                &trace_id,
                &span_id,
                true,
                false,
            )
            .await;
            return decision;
        }

        // Prepare owned dependencies for the async base closure
        let provider_id = self.provider_id.clone();
        let provider_id_for_telemetry = provider_id.clone(); // Clone for telemetry use later
        let http = self.http_client.clone();
        let request_tx = self.request_transformer.clone();
        let sse_tx = sse_tx_opt.clone();
        let json_tx = json_tx_opt.clone();
        let interceptors = self.policy.interceptors.clone();
        let before_send = self.policy.before_send.clone();
        let url = self
            .provider_spec
            .chat_url(true, &req, &self.provider_context);
        let headers_base = self.provider_spec.build_headers(&self.provider_context)?;
        let disable_compression = self.policy.stream_disable_compression;
        let middlewares = self.middlewares.clone();
        let provider_spec = self.provider_spec.clone();
        let provider_context = self.provider_context.clone();
        let retry_options = self.policy.retry_options.clone();

        // Base async stream builder
        let base: Arc<StreamAsyncFn> = Arc::new(move |req_in: ChatRequest| {
            let provider_id = provider_id.clone();
            let http = http.clone();
            let request_tx = request_tx.clone();
            let sse_tx = sse_tx.clone();
            let json_tx = json_tx.clone();
            let interceptors = interceptors.clone();
            let before_send = before_send.clone();
            let url = url.clone();
            let headers_base = headers_base.clone();
            let middlewares = middlewares.clone();
            let _provider_spec = provider_spec.clone();
            let _provider_context = provider_context.clone();
            let retry_options = retry_options.clone();
            Box::pin(async move {
                let transformed =
                    build_chat_body(&request_tx, &middlewares, &before_send, &req_in)?;

                // Build and send streaming via helpers below (SSE or JSON)

                // Converters are module-scoped; call unified helpers to build the stream
                if let Some(stream_tx) = sse_tx {
                    create_sse_stream_with_middlewares(
                        provider_id.clone(),
                        url.clone(),
                        http.clone(),
                        headers_base.clone(),
                        transformed.clone(),
                        stream_tx.clone(),
                        interceptors.clone(),
                        middlewares.clone(),
                        req_in.clone(),
                        disable_compression,
                        retry_options.clone(),
                    )
                    .await
                } else if let Some(jsonc) = json_tx {
                    create_json_stream_with_middlewares(
                        provider_id.clone(),
                        url.clone(),
                        http.clone(),
                        headers_base.clone(),
                        transformed.clone(),
                        jsonc.clone(),
                        interceptors.clone(),
                        middlewares.clone(),
                        req_in.clone(),
                        disable_compression,
                    )
                    .await
                } else {
                    Err(LlmError::UnsupportedOperation(
                        "No stream transformer".into(),
                    ))
                }
            })
        });

        // Wrap with around-style async middlewares in order
        let wrapped = self
            .middlewares
            .iter()
            .rev()
            .fold(base, |next, mw| mw.wrap_stream_async(next));

        let result = wrapped(req.clone()).await;

        // Emit telemetry span end event and wrap stream with telemetry
        if let Some(ref telemetry) = telemetry_config {
            if telemetry.enabled {
                match result {
                    Ok(stream) => {
                        crate::execution::telemetry::chat::span_end_ok_stream(
                            telemetry_config.as_ref(),
                            &trace_id,
                            &span_id,
                            false,
                            true,
                        )
                        .await;

                        // Wrap the stream with telemetry tracking
                        let wrapped_stream = crate::streaming::wrap_stream_with_telemetry(
                            stream,
                            std::sync::Arc::new(telemetry.clone()),
                            trace_id.clone(),
                            provider_id_for_telemetry.clone(),
                            req.common_params.model.clone(),
                            req.messages.clone(),
                        );

                        Ok(wrapped_stream)
                    }
                    Err(error) => {
                        crate::execution::telemetry::chat::span_end_err_stream(
                            telemetry_config.as_ref(),
                            &trace_id,
                            &span_id,
                            &error,
                        )
                        .await;

                        Err(error)
                    }
                }
            } else {
                result
            }
        } else {
            result
        }
    }
}

/// Builder for creating HttpChatExecutor instances
///
/// This builder simplifies the creation of HttpChatExecutor by providing
/// a fluent API and reducing code duplication across providers.
///
/// # Example
/// ```rust,ignore
/// let executor = ChatExecutorBuilder::new("openai", http_client)
///     .with_spec(spec)
///     .with_context(ctx)
///     .with_transformers(req_tx, resp_tx, Some(stream_tx))
///     .with_interceptors(interceptors)
///     .with_middlewares(middlewares)
///     .build();
/// ```
pub struct ChatExecutorBuilder {
    provider_id: String,
    http_client: reqwest::Client,
    spec: Option<Arc<dyn crate::core::ProviderSpec>>,
    context: Option<crate::core::ProviderContext>,
    request_transformer: Option<Arc<dyn RequestTransformer>>,
    response_transformer: Option<Arc<dyn ResponseTransformer>>,
    stream_transformer: Option<Arc<dyn StreamChunkTransformer>>,
    json_stream_converter: Option<Arc<dyn crate::streaming::JsonEventConverter>>,
    policy: crate::execution::ExecutionPolicy,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl ChatExecutorBuilder {
    /// Create a new builder with required fields
    pub fn new(provider_id: impl Into<String>, http_client: reqwest::Client) -> Self {
        Self {
            provider_id: provider_id.into(),
            http_client,
            spec: None,
            context: None,
            request_transformer: None,
            response_transformer: None,
            stream_transformer: None,
            json_stream_converter: None,
            policy: crate::execution::ExecutionPolicy::new(),
            middlewares: Vec::new(),
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
        stream: Option<Arc<dyn StreamChunkTransformer>>,
    ) -> Self {
        self.request_transformer = Some(request);
        self.response_transformer = Some(response);
        self.stream_transformer = stream;
        self
    }

    /// Set transformers from a ChatTransformers bundle
    pub fn with_transformer_bundle(mut self, bundle: crate::core::ChatTransformers) -> Self {
        self.request_transformer = Some(bundle.request);
        self.response_transformer = Some(bundle.response);
        self.stream_transformer = bundle.stream;
        self.json_stream_converter = bundle.json;
        self
    }

    /// Set the JSON stream converter
    pub fn with_json_converter(
        mut self,
        converter: Arc<dyn crate::streaming::JsonEventConverter>,
    ) -> Self {
        self.json_stream_converter = Some(converter);
        self
    }

    /// Set whether to disable compression for streaming
    pub fn with_stream_disable_compression(mut self, disable: bool) -> Self {
        self.policy.stream_disable_compression = disable;
        self
    }

    /// Set HTTP interceptors
    pub fn with_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.policy.interceptors = interceptors;
        self
    }

    /// Set model-level middlewares
    pub fn with_middlewares(mut self, middlewares: Vec<Arc<dyn LanguageModelMiddleware>>) -> Self {
        self.middlewares = middlewares;
        self
    }

    /// Set the before_send hook
    pub fn with_before_send(mut self, hook: crate::execution::executors::BeforeSendHook) -> Self {
        self.policy.before_send = Some(hook);
        self
    }

    /// Set retry options
    pub fn with_retry_options(mut self, retry_options: crate::retry_api::RetryOptions) -> Self {
        self.policy.retry_options = Some(retry_options);
        self
    }

    /// Build the HttpChatExecutor
    ///
    /// # Panics
    /// Panics if required fields (spec, context, transformers) are not set
    pub fn build(self) -> Arc<HttpChatExecutor> {
        Arc::new(HttpChatExecutor {
            provider_id: self.provider_id,
            http_client: self.http_client,
            request_transformer: self
                .request_transformer
                .expect("request_transformer is required"),
            response_transformer: self
                .response_transformer
                .expect("response_transformer is required"),
            stream_transformer: self.stream_transformer,
            json_stream_converter: self.json_stream_converter,
            policy: self.policy,
            middlewares: self.middlewares,
            provider_spec: self.spec.expect("provider_spec is required"),
            provider_context: self.context.expect("provider_context is required"),
        })
    }
}
