//! Chat executor traits
//!
//! Defines the abstraction that will drive chat operations using transformers
//! and HTTP. For now this is an interface stub for the refactor.

use crate::error::LlmError;
use crate::middleware::language_model::{
    GenerateAsyncFn, LanguageModelMiddleware, StreamAsyncFn, apply_post_generate_chain,
    apply_stream_event_chain, apply_transform_chain, try_pre_generate, try_pre_stream,
};
use crate::stream::ChatStream;
use crate::telemetry::{
    self,
    events::{GenerationEvent, SpanEvent, TelemetryEvent},
};
use crate::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::types::{ChatRequest, ChatResponse};
use crate::utils::http_interceptor::{HttpInterceptor, HttpRequestContext};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use eventsource_stream::Event;
use reqwest::header::HeaderMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

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
    pub json_stream_converter: Option<Arc<dyn crate::utils::streaming::JsonEventConverter>>,
    /// Whether to disable compression for streaming requests
    pub stream_disable_compression: bool,
    /// Optional list of HTTP interceptors (order preserved)
    pub interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares (transform ChatRequest before mapping)
    pub middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    // Strategy hooks
    pub build_url: Box<dyn Fn(bool) -> String + Send + Sync>,
    pub build_headers: Arc<dyn Fn() -> Result<HeaderMap, LlmError> + Send + Sync>,
    /// Optional external parameter transformer (plugin-like), applied to JSON body
    pub before_send: Option<crate::executors::BeforeSendHook>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    struct EchoRequestTransformer;
    impl crate::transformers::request::RequestTransformer for EchoRequestTransformer {
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
    impl crate::transformers::response::ResponseTransformer for NoopResponseTransformer {
        fn provider_id(&self) -> &str {
            "test"
        }
    }

    #[tokio::test]
    async fn applies_model_middlewares_before_mapping() {
        struct AppendSuffix;
        impl crate::middleware::language_model::LanguageModelMiddleware for AppendSuffix {
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
        let hook: crate::executors::BeforeSendHook = Arc::new(move |body: &serde_json::Value| {
            let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("");
            *seen_clone.lock().unwrap() = Some(model.to_string());
            Err(crate::error::LlmError::InvalidParameter("abort".into()))
        });

        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: true,
            interceptors: vec![],
            middlewares: vec![Arc::new(AppendSuffix)],
            build_url: Box::new(|_| "http://127.0.0.1/never".to_string()),
            build_headers: Arc::new(|| Ok(reqwest::header::HeaderMap::new())),
            before_send: Some(hook),
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
        impl crate::transformers::stream::StreamChunkTransformer for DummyStreamTx {
            fn provider_id(&self) -> &str {
                "test"
            }
            fn convert_event(
                &self,
                _event: eventsource_stream::Event,
            ) -> crate::transformers::stream::StreamEventFuture<'_> {
                Box::pin(async { Vec::new() })
            }
        }

        // Outer appends suffix to final StreamEnd
        struct Outer;
        impl crate::middleware::language_model::LanguageModelMiddleware for Outer {
            fn wrap_stream_async(
                &self,
                next: std::sync::Arc<crate::middleware::language_model::StreamAsyncFn>,
            ) -> std::sync::Arc<crate::middleware::language_model::StreamAsyncFn> {
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
                        Ok(Box::pin(mapped) as crate::stream::ChatStream)
                    })
                })
            }
        }

        // Inner short-circuits and returns a synthetic stream
        struct Inner;
        impl crate::middleware::language_model::LanguageModelMiddleware for Inner {
            fn wrap_stream_async(
                &self,
                _next: std::sync::Arc<crate::middleware::language_model::StreamAsyncFn>,
            ) -> std::sync::Arc<crate::middleware::language_model::StreamAsyncFn> {
                std::sync::Arc::new(|_req: crate::types::ChatRequest| {
                    Box::pin(async move {
                        let one = futures::stream::once(async move {
                            Ok(crate::types::ChatStreamEvent::StreamEnd {
                                response: crate::types::ChatResponse::new(
                                    crate::types::MessageContent::Text("inner".into()),
                                ),
                            })
                        });
                        Ok(Box::pin(one) as crate::stream::ChatStream)
                    })
                })
            }
        }

        let http = reqwest::Client::new();
        let request_transformer = std::sync::Arc::new(EchoRequestTransformer);
        let response_transformer = std::sync::Arc::new(NoopResponseTransformer);

        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: Some(std::sync::Arc::new(DummyStreamTx)),
            json_stream_converter: None,
            stream_disable_compression: true,
            interceptors: vec![],
            middlewares: vec![std::sync::Arc::new(Outer), std::sync::Arc::new(Inner)],
            build_url: Box::new(|_| "http://127.0.0.1/never".to_string()),
            build_headers: Arc::new(|| Ok(reqwest::header::HeaderMap::new())),
            before_send: None,
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
        use crate::middleware::language_model::GenerateAsyncFn;
        use futures::future::BoxFuture;
        use std::sync::Arc;

        struct Outer;
        impl crate::middleware::language_model::LanguageModelMiddleware for Outer {
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
        impl crate::middleware::language_model::LanguageModelMiddleware for Inner {
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

        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: true,
            interceptors: vec![],
            middlewares: vec![Arc::new(Outer), Arc::new(Inner)],
            build_url: Box::new(|_| "http://127.0.0.1/never".to_string()),
            build_headers: Arc::new(|| Ok(reqwest::header::HeaderMap::new())),
            before_send: None,
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
        impl crate::middleware::language_model::LanguageModelMiddleware for PreMw {
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
        let hook: crate::executors::BeforeSendHook = Arc::new(move |_body: &serde_json::Value| {
            Err(crate::error::LlmError::InvalidParameter(
                "should not be called".into(),
            ))
        });

        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: true,
            interceptors: vec![],
            middlewares: vec![Arc::new(PreMw)],
            build_url: Box::new(|_| "http://127.0.0.1/never".to_string()),
            build_headers: Arc::new(|| Ok(reqwest::header::HeaderMap::new())),
            before_send: Some(hook),
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
        impl crate::middleware::language_model::LanguageModelMiddleware for PreMwStream {
            fn pre_stream(
                &self,
                _req: &crate::types::ChatRequest,
            ) -> Option<Result<crate::stream::ChatStream, crate::error::LlmError>> {
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
        impl crate::transformers::stream::StreamChunkTransformer for DummyTx {
            fn provider_id(&self) -> &str {
                "test"
            }
            fn convert_event(
                &self,
                _event: eventsource_stream::Event,
            ) -> crate::transformers::stream::StreamEventFuture<'_> {
                Box::pin(async { vec![] })
            }
        }

        let http = reqwest::Client::new();
        let request_transformer = Arc::new(EchoRequestTransformer);
        let response_transformer = Arc::new(NoopResponseTransformer);

        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: Some(Arc::new(DummyTx)),
            json_stream_converter: None,
            stream_disable_compression: true,
            interceptors: vec![],
            middlewares: vec![Arc::new(PreMwStream)],
            build_url: Box::new(|_| "http://127.0.0.1/never".to_string()),
            build_headers: std::sync::Arc::new(|| Ok(reqwest::header::HeaderMap::new())),
            before_send: None,
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
    impl crate::utils::http_interceptor::HttpInterceptor for CaptureHeadersInterceptor {
        fn on_before_send(
            &self,
            _ctx: &crate::utils::http_interceptor::HttpRequestContext,
            rb: reqwest::RequestBuilder,
            _body: &serde_json::Value,
            headers: &reqwest::header::HeaderMap,
        ) -> Result<reqwest::RequestBuilder, LlmError> {
            *self.seen.lock().unwrap() = Some(headers.clone());
            Err(LlmError::InvalidParameter("abort".into()))
        }
    }

    #[tokio::test]
    async fn merges_request_headers_into_base_nonstream() {
        let http = reqwest::Client::new();
        let request_transformer = Arc::new(EchoRequestTransformer);
        let response_transformer = Arc::new(NoopResponseTransformer);
        // base headers
        let headers_builder = || {
            let mut h = reqwest::header::HeaderMap::new();
            h.insert(
                reqwest::header::HeaderName::from_static("x-base"),
                reqwest::header::HeaderValue::from_static("base"),
            );
            Ok(h)
        };
        let seen = Arc::new(std::sync::Mutex::new(None));
        let interceptor = CaptureHeadersInterceptor { seen: seen.clone() };
        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: true,
            interceptors: vec![Arc::new(interceptor)],
            middlewares: vec![],
            build_url: Box::new(|_| "http://127.0.0.1/never".to_string()),
            build_headers: Arc::new(headers_builder),
            before_send: None,
        };

        let mut req = crate::types::ChatRequest::new(vec![]);
        // per-request header
        let mut hc = crate::types::HttpConfig::default();
        hc.headers
            .insert("X-Req".to_string(), "req".to_string());
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
        impl crate::transformers::stream::StreamChunkTransformer for DummyTx {
            fn provider_id(&self) -> &str {
                "test"
            }
            fn convert_event(
                &self,
                _event: eventsource_stream::Event,
            ) -> crate::transformers::stream::StreamEventFuture<'_> {
                Box::pin(async { vec![] })
            }
        }
        let http = reqwest::Client::new();
        let request_transformer = Arc::new(EchoRequestTransformer);
        let response_transformer = Arc::new(NoopResponseTransformer);
        let seen = Arc::new(std::sync::Mutex::new(None));
        let interceptor = CaptureHeadersInterceptor { seen: seen.clone() };
        let headers_builder = || {
            let mut h = reqwest::header::HeaderMap::new();
            h.insert(
                reqwest::header::HeaderName::from_static("x-base"),
                reqwest::header::HeaderValue::from_static("base"),
            );
            Ok(h)
        };
        let exec = HttpChatExecutor {
            provider_id: "test".into(),
            http_client: http,
            request_transformer,
            response_transformer,
            stream_transformer: Some(Arc::new(DummyTx)),
            json_stream_converter: None,
            stream_disable_compression: true,
            interceptors: vec![Arc::new(interceptor)],
            middlewares: vec![],
            build_url: Box::new(|_| "http://127.0.0.1/never".to_string()),
            build_headers: Arc::new(headers_builder),
            before_send: None,
        };
        let mut req = crate::types::ChatRequest::new(vec![]);
        let mut hc = crate::types::HttpConfig::default();
        hc.headers
            .insert("X-Req".to_string(), "req".to_string());
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

        if let Some(ref telemetry) = telemetry_config {
            if telemetry.enabled {
                let span = SpanEvent::start(
                    span_id.clone(),
                    None,
                    trace_id.clone(),
                    "ai.executor.chat.execute".to_string(),
                )
                .with_attribute("provider_id", self.provider_id.clone())
                .with_attribute("model", req.common_params.model.clone())
                .with_attribute("stream", "false");

                telemetry::emit(TelemetryEvent::SpanStart(span)).await;
            }
        }

        // Apply model-level parameter transforms
        let req = apply_transform_chain(&self.middlewares, req);
        // Try pre-generate short-circuit
        if let Some(decision) = try_pre_generate(&self.middlewares, &req) {
            // Emit telemetry span end for short-circuit
            if let Some(ref telemetry) = telemetry_config {
                if telemetry.enabled {
                    let span = SpanEvent::start(
                        span_id.clone(),
                        None,
                        trace_id.clone(),
                        "ai.executor.chat.execute".to_string(),
                    )
                    .end_ok()
                    .with_attribute("short_circuit", "true");

                    telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
                }
            }
            return decision;
        }

        // Prepare owned dependencies for the async base closure
        let provider_id = self.provider_id.clone();
        let provider_id_for_telemetry = provider_id.clone(); // Clone for telemetry use later
        let client = self.http_client.clone();
        let request_tx = self.request_transformer.clone();
        let response_tx = self.response_transformer.clone();
        let interceptors = self.interceptors.clone();
        let before_send = self.before_send.clone();
        // Pre-compute URL and initial headers (provider/base-level). Request-level headers are merged later per-request.
        let url = (self.build_url)(false);
        let build_headers = self.build_headers.clone();
        let headers_initial = (build_headers)()?;
        let headers_for_interceptors_initial = headers_initial.clone();

        // Base async generator (no post_generate here)
        let base: Arc<GenerateAsyncFn> = Arc::new(move |req_in: ChatRequest| {
            let url = url.clone();
            let headers_initial = headers_initial.clone();
            let headers_for_interceptors_initial = headers_for_interceptors_initial.clone();
            let client = client.clone();
            let request_tx = request_tx.clone();
            let response_tx = response_tx.clone();
            let interceptors = interceptors.clone();
            let before_send = before_send.clone();
            let provider_id = provider_id.clone();
            Box::pin({
                let build_headers2 = build_headers.clone();
                async move {
                    let body = request_tx.transform_chat(&req_in)?;
                    let json_body = if let Some(cb) = &before_send {
                        cb(&body)?
                    } else {
                        body
                    };

                    // Build request and run interceptors
                    // Merge per-request http_config.headers on top of provider/base headers
                    let merged_headers = {
                        let mut h = headers_initial.clone();
                        if let Some(hc) = &req_in.http_config {
                            for (k, v) in &hc.headers {
                                if let (Ok(name), Ok(val)) = (
                                    reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                                    reqwest::header::HeaderValue::from_str(v),
                                ) {
                                    h.insert(name, val);
                                }
                            }
                        }
                        h
                    };
                    let mut rb = client.post(url.clone()).headers(merged_headers);
                    let ctx = HttpRequestContext {
                        provider_id: provider_id.clone(),
                        url: url.clone(),
                        stream: false,
                    };
                    // Build effective headers for interceptor visibility
                    let cloned_headers = rb
                        .try_clone()
                        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                        .unwrap_or_default();
                    for it in &interceptors {
                        rb = it.on_before_send(&ctx, rb, &json_body, &cloned_headers)?;
                    }
                    let resp = rb
                        .json(&json_body)
                        .send()
                        .await
                        .map_err(|e| LlmError::HttpError(e.to_string()))?;
                    let resp = if !resp.status().is_success() {
                        let status = resp.status();
                        if status.as_u16() == 401 {
                            // Rebuild headers and retry once
                            let headers_retry = (build_headers2)()?;
                            let headers_retry = {
                                let mut h = headers_retry;
                                if let Some(hc) = &req_in.http_config {
                                    for (k, v) in &hc.headers {
                                        if let (Ok(name), Ok(val)) = (
                                            reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                                            reqwest::header::HeaderValue::from_str(v),
                                        ) {
                                            h.insert(name, val);
                                        }
                                    }
                                }
                                h
                            };
                            let headers_for_interceptors_retry = headers_retry.clone();
                            let mut rb2 = client.post(url.clone()).headers(headers_retry);
                            let ctx = HttpRequestContext {
                                provider_id: provider_id.clone(),
                                url: url.clone(),
                                stream: false,
                            };
                            for it in &interceptors {
                                rb2 = it.on_before_send(
                                    &ctx,
                                    rb2,
                                    &json_body,
                                    &headers_for_interceptors_retry,
                                )?;
                            }
                            rb2.json(&json_body)
                                .send()
                                .await
                                .map_err(|e| LlmError::HttpError(e.to_string()))?
                        } else {
                            let headers = resp.headers().clone();
                            let text = resp.text().await.unwrap_or_default();
                            for it in &interceptors {
                                it.on_error(
                                    &ctx,
                                    &crate::retry_api::classify_http_error(
                                        &provider_id,
                                        status.as_u16(),
                                        &text,
                                        &headers,
                                        None,
                                    ),
                                );
                            }
                            return Err(crate::retry_api::classify_http_error(
                                &provider_id,
                                status.as_u16(),
                                &text,
                                &headers,
                                None,
                            ));
                        }
                    } else {
                        resp
                    };

                    for it in &interceptors {
                        it.on_response(&ctx, &resp)?;
                    }
                    let text = resp
                        .text()
                        .await
                        .map_err(|e| LlmError::HttpError(e.to_string()))?;
                    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                        LlmError::ParseError(format!("Failed to parse response JSON: {e}"))
                    })?;
                    let resp = response_tx.transform_chat_response(&json)?;
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
        if let Some(ref telemetry) = telemetry_config {
            if telemetry.enabled {
                match &result {
                    Ok(response) => {
                        // Emit span end event
                        let duration = std::time::SystemTime::now().duration_since(start_time).ok();
                        let span = SpanEvent::start(
                            span_id.clone(),
                            None,
                            trace_id.clone(),
                            "ai.executor.chat.execute".to_string(),
                        )
                        .end_ok()
                        .with_attribute("finish_reason", format!("{:?}", response.finish_reason));

                        telemetry::emit(TelemetryEvent::SpanEnd(span)).await;

                        // Emit generation event
                        let mut gen_event = GenerationEvent::new(
                            uuid::Uuid::new_v4().to_string(),
                            trace_id.clone(),
                            provider_id_for_telemetry.clone(),
                            req.common_params.model.clone(),
                        );

                        if telemetry.record_inputs {
                            gen_event = gen_event.with_input(req.messages.clone());
                        }

                        if telemetry.record_outputs {
                            gen_event = gen_event.with_output(response.clone());
                        }

                        if telemetry.record_usage {
                            if let Some(usage) = &response.usage {
                                gen_event = gen_event.with_usage(usage.clone());
                            }
                        }

                        if let Some(reason) = &response.finish_reason {
                            gen_event = gen_event.with_finish_reason(reason.clone());
                        }

                        if let Some(dur) = duration {
                            gen_event = gen_event.with_duration(dur);
                        }

                        telemetry::emit(TelemetryEvent::Generation(gen_event)).await;
                    }
                    Err(error) => {
                        // Emit error span
                        let span = SpanEvent::start(
                            span_id.clone(),
                            None,
                            trace_id.clone(),
                            "ai.executor.chat.execute".to_string(),
                        )
                        .end_error(error.to_string());

                        telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
                    }
                }
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
        let start_time = std::time::SystemTime::now();
        let telemetry_config = req.telemetry.clone();

        if let Some(ref telemetry) = telemetry_config {
            if telemetry.enabled {
                let span = SpanEvent::start(
                    span_id.clone(),
                    None,
                    trace_id.clone(),
                    "ai.executor.chat.execute_stream".to_string(),
                )
                .with_attribute("provider_id", self.provider_id.clone())
                .with_attribute("model", req.common_params.model.clone())
                .with_attribute("stream", "true");

                telemetry::emit(TelemetryEvent::SpanStart(span)).await;
            }
        }

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
            if let Some(ref telemetry) = telemetry_config {
                if telemetry.enabled {
                    let span = SpanEvent::start(
                        span_id.clone(),
                        None,
                        trace_id.clone(),
                        "ai.executor.chat.execute_stream".to_string(),
                    )
                    .end_ok()
                    .with_attribute("short_circuit", "true");

                    telemetry::emit(TelemetryEvent::SpanEnd(span)).await;
                }
            }
            return decision;
        }

        // Prepare owned dependencies for the async base closure
        let provider_id = self.provider_id.clone();
        let provider_id_for_telemetry = provider_id.clone(); // Clone for telemetry use later
        let http = self.http_client.clone();
        let request_tx = self.request_transformer.clone();
        let sse_tx = sse_tx_opt.clone();
        let json_tx = json_tx_opt.clone();
        let interceptors = self.interceptors.clone();
        let before_send = self.before_send.clone();
        let url = (self.build_url)(true);
        let headers_base = (self.build_headers)()?;
        let disable_compression = self.stream_disable_compression;
        let middlewares = self.middlewares.clone();

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
            Box::pin(async move {
                let body = request_tx.transform_chat(&req_in)?;
                let transformed = if let Some(cb) = &before_send {
                    cb(&body)?
                } else {
                    body
                };

                // Build SSE request (used only when SSE transformer present)
                let build_request = {
                    let http = http.clone();
                    let headers_base = headers_base.clone();
                    let url_for_retry = url.clone();
                    let transformed_for_retry = transformed.clone();
                    let interceptors = interceptors.clone();
                    let provider_id = provider_id.clone();
                    // Extract per-request headers once to avoid moving req_in into the closure
                    let req_headers_extra = req_in
                        .http_config
                        .as_ref()
                        .map(|hc| hc.headers.clone());
                    move || -> Result<reqwest::RequestBuilder, LlmError> {
                        // Merge per-request headers into base headers
                        let headers_effective = {
                            let mut h = headers_base.clone();
                            if let Some(ref headers_map) = req_headers_extra {
                                for (k, v) in headers_map.iter() {
                                    if let (Ok(name), Ok(val)) = (
                                        reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                                        reqwest::header::HeaderValue::from_str(v),
                                    ) {
                                        h.insert(name, val);
                                    }
                                }
                            }
                            h
                        };
                        let mut rb = http
                            .post(url_for_retry.clone())
                            .headers(headers_effective)
                            .header(reqwest::header::ACCEPT, "text/event-stream")
                            .header(reqwest::header::CACHE_CONTROL, "no-cache")
                            .header(reqwest::header::CONNECTION, "keep-alive");
                        let mut body_for_send = transformed_for_retry.clone();
                        if provider_id.starts_with("openai") {
                            body_for_send["stream"] = serde_json::Value::Bool(true);
                            if body_for_send.get("stream_options").is_none() {
                                body_for_send["stream_options"] =
                                    serde_json::json!({"include_usage": true});
                            } else if let Some(obj) =
                                body_for_send["stream_options"].as_object_mut()
                            {
                                obj.entry("include_usage")
                                    .or_insert(serde_json::Value::Bool(true));
                            }
                        }
                        rb = rb.json(&body_for_send);
                        if disable_compression {
                            rb = rb.header(reqwest::header::ACCEPT_ENCODING, "identity");
                        }
                        // Interceptors on streaming request
                        let ctx = HttpRequestContext {
                            provider_id: provider_id.clone(),
                            url: url_for_retry.clone(),
                            stream: true,
                        };
                        // Build headers again for interceptor visibility
                        let cloned_headers = rb
                            .try_clone()
                            .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                            .unwrap_or_default();
                        for it in &interceptors {
                            rb = it.on_before_send(&ctx, rb, &body_for_send, &cloned_headers)?;
                        }
                        Ok(rb)
                    }
                };

                // Adapters and converters
                #[derive(Clone)]
                struct TransformerConverter(Arc<dyn StreamChunkTransformer>);
                impl SseEventConverter for TransformerConverter {
                    fn convert_event(
                        &self,
                        event: Event,
                    ) -> Pin<
                        Box<
                            dyn Future<
                                    Output = Vec<Result<crate::stream::ChatStreamEvent, LlmError>>,
                                > + Send
                                + Sync
                                + '_,
                        >,
                    > {
                        self.0.convert_event(event)
                    }
                    fn handle_stream_end(
                        &self,
                    ) -> Option<Result<crate::stream::ChatStreamEvent, LlmError>>
                    {
                        self.0.handle_stream_end()
                    }
                }

                #[derive(Clone)]
                struct MiddlewareConverter<C> {
                    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
                    req: crate::types::ChatRequest,
                    convert: C,
                }
                impl<C> SseEventConverter for MiddlewareConverter<C>
                where
                    C: SseEventConverter + Clone + Send + Sync + 'static,
                {
                    fn convert_event(
                        &self,
                        event: eventsource_stream::Event,
                    ) -> Pin<
                        Box<
                            dyn Future<
                                    Output = Vec<Result<crate::stream::ChatStreamEvent, LlmError>>,
                                > + Send
                                + Sync
                                + '_,
                        >,
                    > {
                        let mws = self.middlewares.clone();
                        let req = self.req.clone();
                        let inner = self.convert.clone();
                        Box::pin(async move {
                            let raw = inner.convert_event(event).await;
                            let mut out = Vec::new();
                            for item in raw.into_iter() {
                                match item {
                                    Ok(ev) => match apply_stream_event_chain(&mws, &req, ev) {
                                        Ok(list) => out.extend(list.into_iter().map(Ok)),
                                        Err(e) => out.push(Err(e)),
                                    },
                                    Err(e) => out.push(Err(e)),
                                }
                            }
                            out
                        })
                    }
                    fn handle_stream_end(
                        &self,
                    ) -> Option<Result<crate::stream::ChatStreamEvent, LlmError>>
                    {
                        // Ensure end event also flows through middlewares
                        match self.convert.handle_stream_end() {
                            Some(Ok(ev)) => {
                                match apply_stream_event_chain(&self.middlewares, &self.req, ev)
                                    .map(|mut v| v.pop())
                                {
                                    Ok(Some(last)) => Some(Ok(last)),
                                    Ok(None) => None, // filtered out by middleware
                                    Err(e) => Some(Err(e)),
                                }
                            }
                            other => other,
                        }
                    }
                }

                #[derive(Clone)]
                struct InterceptingConverter<C> {
                    interceptors: Vec<Arc<dyn HttpInterceptor>>,
                    ctx: HttpRequestContext,
                    convert: C,
                }
                impl<C> SseEventConverter for InterceptingConverter<C>
                where
                    C: SseEventConverter + Clone + Send + Sync + 'static,
                {
                    fn convert_event(
                        &self,
                        event: eventsource_stream::Event,
                    ) -> Pin<
                        Box<
                            dyn Future<
                                    Output = Vec<Result<crate::stream::ChatStreamEvent, LlmError>>,
                                > + Send
                                + Sync
                                + '_,
                        >,
                    > {
                        let interceptors = self.interceptors.clone();
                        let ctx = self.ctx.clone();
                        let inner = self.convert.clone();
                        Box::pin(async move {
                            for it in &interceptors {
                                if let Err(e) = it.on_sse_event(&ctx, &event) {
                                    return vec![Err(e)];
                                }
                            }
                            inner.convert_event(event).await
                        })
                    }
                    fn handle_stream_end(
                        &self,
                    ) -> Option<Result<crate::stream::ChatStreamEvent, LlmError>>
                    {
                        self.convert.handle_stream_end()
                    }
                }

                if let Some(stream_tx) = sse_tx {
                    let converter = TransformerConverter(stream_tx.clone());
                    let mw_wrapped = MiddlewareConverter {
                        middlewares: middlewares.clone(),
                        req: req_in.clone(),
                        convert: converter,
                    };
                    let intercepting = InterceptingConverter {
                        interceptors: interceptors.clone(),
                        ctx: HttpRequestContext {
                            provider_id: provider_id.clone(),
                            url: url.clone(),
                            stream: true,
                        },
                        convert: mw_wrapped,
                    };
                    StreamFactory::create_eventsource_stream_with_retry(
                        &provider_id,
                        build_request,
                        intercepting,
                    )
                    .await
                } else if let Some(jsonc) = json_tx {
                    // JSON streaming path: build request without SSE headers
                    let mut rb = http.post(url.clone()).headers(headers_base.clone());
                    let body_for_send = transformed.clone();
                    rb = rb.json(&body_for_send);
                    if disable_compression {
                        rb = rb.header(reqwest::header::ACCEPT_ENCODING, "identity");
                    }
                    // Interceptors
                    let ctx = HttpRequestContext {
                        provider_id: provider_id.clone(),
                        url: url.clone(),
                        stream: true,
                    };
                    let cloned_headers = rb
                        .try_clone()
                        .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                        .unwrap_or_default();
                    for it in &interceptors {
                        rb = it.on_before_send(&ctx, rb, &body_for_send, &cloned_headers)?;
                    }
                    let response = rb
                        .send()
                        .await
                        .map_err(|e| LlmError::HttpError(format!("Failed to send request: {e}")))?;
                    if !response.status().is_success() {
                        let status = response.status();
                        let error_text = response.text().await.unwrap_or_default();
                        return Err(LlmError::HttpError(format!(
                            "HTTP error {}: {}",
                            status.as_u16(),
                            error_text
                        )));
                    }
                    // Wrap JSON converter with middlewares
                #[derive(Clone)]
                struct MiddlewareJsonConverter {
                    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
                    req: crate::types::ChatRequest,
                    convert: Arc<dyn crate::utils::streaming::JsonEventConverter>,
                }
                impl crate::utils::streaming::JsonEventConverter for MiddlewareJsonConverter {
                    fn convert_json<'a>(
                        &'a self,
                        json_data: &'a str,
                    ) -> Pin<
                        Box<
                            dyn Future<
                                    Output = Vec<Result<crate::stream::ChatStreamEvent, LlmError>>,
                                > + Send
                                + Sync
                                + 'a,
                        >,
                    > {
                        let mws = self.middlewares.clone();
                        let req = self.req.clone();
                        let inner = self.convert.clone();
                        Box::pin(async move {
                            let raw = inner.convert_json(json_data).await;
                            let mut out = Vec::new();
                            for item in raw.into_iter() {
                                match item {
                                    Ok(ev) => match apply_stream_event_chain(&mws, &req, ev) {
                                        Ok(list) => out.extend(list.into_iter().map(Ok)),
                                        Err(e) => out.push(Err(e)),
                                    },
                                    Err(e) => out.push(Err(e)),
                                }
                            }
                            out
                        })
                    }
                }
                    let mw = MiddlewareJsonConverter { middlewares: middlewares.clone(), req: req_in.clone(), convert: jsonc.clone() };
                    crate::utils::streaming::StreamFactory::create_json_stream(response, mw).await
                } else {
                    Err(LlmError::UnsupportedOperation("No stream transformer".into()))
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
                        let span = SpanEvent::start(
                            span_id.clone(),
                            None,
                            trace_id.clone(),
                            "ai.executor.chat.execute_stream".to_string(),
                        )
                        .end_ok()
                        .with_attribute("stream_created", "true");

                        telemetry::emit(TelemetryEvent::SpanEnd(span)).await;

                        // Wrap the stream with telemetry tracking
                        let wrapped_stream = crate::stream::wrap_stream_with_telemetry(
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
                        let span = SpanEvent::start(
                            span_id.clone(),
                            None,
                            trace_id.clone(),
                            "ai.executor.chat.execute_stream".to_string(),
                        )
                        .end_error(error.to_string());

                        telemetry::emit(TelemetryEvent::SpanEnd(span)).await;

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
