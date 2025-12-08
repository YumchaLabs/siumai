use super::*;
use futures::StreamExt;
use reqwest::header::HeaderMap;
use std::sync::Arc;

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
    fn build_headers(&self, _ctx: &crate::core::ProviderContext) -> Result<HeaderMap, LlmError> {
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
        ) -> std::sync::Arc<crate::execution::middleware::language_model::StreamAsyncFn> {
            std::sync::Arc::new(move |req: crate::types::ChatRequest| {
                let next = next.clone();
                Box::pin(async move {
                    let s = next(req).await?;
                    let mapped = s.map(|res| {
                        res.map(|ev| match ev {
                            crate::types::ChatStreamEvent::StreamEnd { mut response } => {
                                if let Some(t) = response.content_text() {
                                    response.content =
                                        crate::types::MessageContent::Text(format!("{}-outer", t));
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
        ) -> std::sync::Arc<crate::execution::middleware::language_model::StreamAsyncFn> {
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
