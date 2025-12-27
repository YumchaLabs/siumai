//! Factory-level SSE injection tests
//!
//! Validates StreamFactory's behavior to inject a synthetic ContentDelta when:
//! - No content deltas were seen during the stream, and
//! - The converter provides a final StreamEnd carrying non-empty content on [DONE].

use futures_util::StreamExt;
use siumai::error::LlmError;
use siumai::execution::http::{HttpRequestContext, generate_request_id};
use siumai::streaming::ChatStreamEvent;
use siumai::streaming::{SseEventConverter, StreamFactory};

fn make_ctx(provider_id: &str, url: &str) -> HttpRequestContext {
    HttpRequestContext {
        request_id: generate_request_id(),
        provider_id: provider_id.to_string(),
        url: url.to_string(),
        stream: true,
    }
}

#[derive(Clone)]
struct EndOnlyConverter;

impl SseEventConverter for EndOnlyConverter {
    fn convert_event(
        &self,
        _event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<
                        Result<siumai::streaming::ChatStreamEvent, siumai::error::LlmError>,
                    >,
                > + Send
                + Sync
                + '_,
        >,
    > {
        Box::pin(async move { vec![] })
    }

    fn handle_stream_end(
        &self,
    ) -> Option<Result<siumai::streaming::ChatStreamEvent, siumai::error::LlmError>> {
        let mut response = siumai::types::ChatResponse::new(siumai::types::MessageContent::Text(
            "INJECT".to_string(),
        ));
        response.finish_reason = Some(siumai::types::FinishReason::Stop);
        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

#[derive(Clone)]
struct DeltaThenEndConverter;

impl SseEventConverter for DeltaThenEndConverter {
    fn convert_event(
        &self,
        _event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<
                        Result<siumai::streaming::ChatStreamEvent, siumai::error::LlmError>,
                    >,
                > + Send
                + Sync
                + '_,
        >,
    > {
        Box::pin(async move {
            vec![Ok(ChatStreamEvent::ContentDelta {
                delta: "DELTA".to_string(),
                index: None,
            })]
        })
    }

    fn handle_stream_end(
        &self,
    ) -> Option<Result<siumai::streaming::ChatStreamEvent, siumai::error::LlmError>> {
        let mut response = siumai::types::ChatResponse::new(siumai::types::MessageContent::Text(
            "END".to_string(),
        ));
        response.finish_reason = Some(siumai::types::FinishReason::Stop);
        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

#[tokio::test]
async fn factory_injects_contentdelta_when_end_with_text_and_no_deltas() {
    use axum::body::{Body, Bytes};
    use axum::http::header;
    use axum::response::Response;
    use axum::{Router, routing::get};

    // Simple SSE server that only emits [DONE]
    let app = Router::new().route(
        "/sse",
        get(|| async move {
            let chunks = vec![Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: [DONE]\n\n",
            ))];
            let body = Body::from_stream(futures_util::stream::iter(chunks));
            let mut resp = Response::new(body);
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("text/event-stream"),
            );
            resp
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{}/sse", addr);
    let client = reqwest::Client::new();
    let url_clone = url.clone();
    let build_request =
        move || -> Result<reqwest::RequestBuilder, LlmError> { Ok(client.get(url_clone.clone())) };

    let converter = EndOnlyConverter;
    let stream = StreamFactory::create_eventsource_stream_with_retry(
        "factory-test",
        &url,
        true,
        build_request,
        converter,
        &[], // No interceptors in test
        make_ctx("factory-test", &url),
    )
    .await
    .expect("stream created");

    // Collect events
    let events: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|e| e.unwrap())
        .collect();

    // Expect a synthetic ContentDelta with 'INJECT' followed by StreamEnd
    assert!(
        matches!(events.first(), Some(ChatStreamEvent::ContentDelta { delta, .. }) if delta == "INJECT")
    );
    assert!(matches!(
        events.get(1),
        Some(ChatStreamEvent::StreamEnd { .. })
    ));

    drop(server); // Stop server
}

#[tokio::test]
async fn factory_does_not_inject_when_delta_already_seen() {
    use axum::body::{Body, Bytes};
    use axum::http::header;
    use axum::response::Response;
    use axum::{Router, routing::get};

    // SSE emits one non-DONE chunk and then [DONE]
    let app = Router::new().route(
        "/sse",
        get(|| async move {
            let chunks = vec![
                Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: {\"any\":true}\n\n")),
                Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\n")),
            ];
            let body = Body::from_stream(futures_util::stream::iter(chunks));
            let mut resp = Response::new(body);
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("text/event-stream"),
            );
            resp
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{}/sse", addr);
    let client = reqwest::Client::new();
    let url_clone = url.clone();
    let build_request =
        move || -> Result<reqwest::RequestBuilder, LlmError> { Ok(client.get(url_clone.clone())) };

    let converter = DeltaThenEndConverter;
    let stream = StreamFactory::create_eventsource_stream_with_retry(
        "factory-test",
        &url,
        true,
        build_request,
        converter,
        &[], // No interceptors in test
        make_ctx("factory-test", &url),
    )
    .await
    .expect("stream created");

    let events: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|e| e.unwrap())
        .collect();

    // Expect only one ContentDelta (DELTA) followed by StreamEnd, without synthetic injection from END
    assert!(
        matches!(events.first(), Some(ChatStreamEvent::ContentDelta { delta, .. }) if delta == "DELTA")
    );
    assert!(matches!(
        events.get(1),
        Some(ChatStreamEvent::StreamEnd { .. })
    ));
    assert_eq!(
        events.len(),
        2,
        "should not inject extra ContentDelta when delta already seen"
    );

    drop(server);
}

#[tokio::test]
async fn factory_no_stream_end_when_no_done_and_converter_end_only() {
    use axum::body::{Body, Bytes};
    use axum::http::header;
    use axum::response::Response;
    use axum::{Router, routing::get};

    // SSE emits only one non-DONE chunk and closes.
    let app = Router::new().route(
        "/sse",
        get(|| async move {
            let chunks = vec![Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"any\":true}\n\n",
            ))];
            let body = Body::from_stream(futures_util::stream::iter(chunks));
            let mut resp = Response::new(body);
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("text/event-stream"),
            );
            resp
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{}/sse", addr);
    let client = reqwest::Client::new();
    let url_clone = url.clone();
    let build_request =
        move || -> Result<reqwest::RequestBuilder, LlmError> { Ok(client.get(url_clone.clone())) };

    let converter = EndOnlyConverter; // emits StreamEnd only on [DONE]
    let stream = StreamFactory::create_eventsource_stream_with_retry(
        "factory-test",
        &url,
        true,
        build_request,
        converter,
        &[], // No interceptors in test
        make_ctx("factory-test", &url),
    )
    .await
    .expect("stream created");

    let events: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|e| e.unwrap())
        .collect();

    // No [DONE], and converter only produces end on [DONE] → no events
    assert!(
        events.is_empty(),
        "no events expected without [DONE] and no deltas"
    );

    drop(server);
}

#[derive(Clone)]
struct DeltaOnlyConverter;
impl SseEventConverter for DeltaOnlyConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Vec<
                        Result<siumai::streaming::ChatStreamEvent, siumai::error::LlmError>,
                    >,
                > + Send
                + Sync
                + '_,
        >,
    > {
        Box::pin(async move {
            if event.data.trim().is_empty() || event.data.trim() == "[DONE]" {
                return vec![];
            }
            vec![Ok(ChatStreamEvent::ContentDelta {
                delta: event.data,
                index: None,
            })]
        })
    }
}

#[tokio::test]
async fn factory_disconnect_mid_stream_after_delta_no_end() {
    use axum::body::{Body, Bytes};
    use axum::http::header;
    use axum::response::Response;
    use axum::{Router, routing::get};

    // SSE emits one JSON chunk then closes connection (no [DONE])
    let app = Router::new().route(
        "/sse",
        get(|| async move {
            let chunks = vec![Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"hello\":true}\n\n",
            ))];
            let body = Body::from_stream(futures_util::stream::iter(chunks));
            let mut resp = Response::new(body);
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("text/event-stream"),
            );
            resp
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{}/sse", addr);
    let client = reqwest::Client::new();
    let url_clone = url.clone();
    let build_request =
        move || -> Result<reqwest::RequestBuilder, LlmError> { Ok(client.get(url_clone.clone())) };

    let converter = DeltaOnlyConverter;
    let stream = StreamFactory::create_eventsource_stream_with_retry(
        "factory-test",
        &url,
        true,
        build_request,
        converter,
        &[], // No interceptors in test
        make_ctx("factory-test", &url),
    )
    .await
    .expect("stream created");

    let events: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|e| e.unwrap())
        .collect();
    assert!(matches!(
        events.as_slice(),
        [ChatStreamEvent::ContentDelta { .. }]
    ));

    drop(server);
}

#[tokio::test]
async fn factory_retries_on_401_then_succeeds() {
    use axum::body::{Body, Bytes};
    use axum::http::{StatusCode, header};
    use axum::response::Response;
    use axum::{Router, extract::State, routing::get};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Clone)]
    struct AppState {
        hits: Arc<AtomicUsize>,
    }

    async fn handler(State(state): State<AppState>) -> Response {
        let n = state.hits.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .body(Body::from("unauthorized"))
                .unwrap()
        } else {
            let chunks = vec![
                Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: {\"ok\":true}\n\n")),
                Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\n")),
            ];
            let body = Body::from_stream(futures_util::stream::iter(chunks));
            let mut resp = Response::new(body);
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("text/event-stream"),
            );
            resp
        }
    }

    let state = AppState {
        hits: Arc::new(AtomicUsize::new(0)),
    };
    let app = Router::new().route("/sse", get(handler)).with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{}/sse", addr);
    let client = reqwest::Client::new();
    let url_clone = url.clone();
    let build_request =
        move || -> Result<reqwest::RequestBuilder, LlmError> { Ok(client.get(url_clone.clone())) };

    let converter = DeltaOnlyConverter;
    let stream = StreamFactory::create_eventsource_stream_with_retry(
        "factory-test",
        &url,
        true,
        build_request,
        converter,
        &[], // No interceptors in test
        make_ctx("factory-test", &url),
    )
    .await
    .expect("stream created");
    let events: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|e| e.unwrap())
        .collect();
    assert!(
        events.iter().any(
            |e| matches!(e, ChatStreamEvent::ContentDelta { delta, .. } if delta == "{\"ok\":true}")
        ),
        "expected ContentDelta for ok payload"
    );

    drop(server);
}

#[tokio::test]
async fn factory_http_429_classified_as_rate_limit() {
    use axum::body::Body;
    use axum::http::StatusCode;
    use axum::response::Response;
    use axum::{Router, routing::get};

    let app = Router::new().route(
        "/sse",
        get(|| async move {
            Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .body(Body::from("Too many requests"))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{}/sse", addr);
    let client = reqwest::Client::new();
    let url_clone = url.clone();
    let build_request =
        move || -> Result<reqwest::RequestBuilder, LlmError> { Ok(client.get(url_clone.clone())) };

    let converter = DeltaOnlyConverter;
    let res = StreamFactory::create_eventsource_stream_with_retry(
        "factory-test",
        &url,
        true,
        build_request,
        converter,
        &[], // No interceptors in test
        make_ctx("factory-test", &url),
    )
    .await;

    match res {
        Err(LlmError::RateLimitError(msg)) => assert!(msg.contains("provider=factory-test")),
        Ok(_) => panic!("expected RateLimitError, got Ok"),
        Err(e) => panic!("expected RateLimitError, got: {e:?}"),
    }

    drop(server);
}

#[tokio::test]
async fn factory_http_503_classified_as_api_error() {
    use axum::body::Body;
    use axum::http::StatusCode;
    use axum::response::Response;
    use axum::{Router, routing::get};

    let app = Router::new().route(
        "/sse",
        get(|| async move {
            Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Body::from("backend down"))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{}/sse", addr);
    let client = reqwest::Client::new();
    let url_clone = url.clone();
    let build_request =
        move || -> Result<reqwest::RequestBuilder, LlmError> { Ok(client.get(url_clone.clone())) };

    let converter = DeltaOnlyConverter;
    let res = StreamFactory::create_eventsource_stream_with_retry(
        "factory-test",
        &url,
        true,
        build_request,
        converter,
        &[], // No interceptors in test
        make_ctx("factory-test", &url),
    )
    .await;

    match res {
        Err(LlmError::ApiError { code, message, .. }) => {
            assert_eq!(code, 503);
            assert!(message.contains("server error") || message.contains("backend down"));
        }
        Ok(_) => panic!("expected ApiError 503, got Ok"),
        Err(e) => panic!("expected ApiError 503, got: {e:?}"),
    }

    drop(server);
}

#[tokio::test]
async fn factory_http_502_classified_as_api_error() {
    use axum::body::Body;
    use axum::http::StatusCode;
    use axum::response::Response;
    use axum::{Router, routing::get};

    let app = Router::new().route(
        "/sse",
        get(|| async move {
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from("bad gateway"))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{}/sse", addr);
    let client = reqwest::Client::new();
    let url_clone = url.clone();
    let build_request =
        move || -> Result<reqwest::RequestBuilder, LlmError> { Ok(client.get(url_clone.clone())) };

    let converter = DeltaOnlyConverter;
    let res = StreamFactory::create_eventsource_stream_with_retry(
        "factory-test",
        &url,
        true,
        build_request,
        converter,
        &[], // No interceptors in test
        make_ctx("factory-test", &url),
    )
    .await;

    match res {
        Err(LlmError::ApiError { code, .. }) => {
            assert_eq!(code, 502);
        }
        Ok(_) => panic!("expected ApiError 502, got Ok"),
        Err(e) => panic!("expected ApiError 502, got: {e:?}"),
    }

    drop(server);
}

#[tokio::test]
async fn factory_http_504_classified_as_api_error() {
    use axum::body::Body;
    use axum::http::StatusCode;
    use axum::response::Response;
    use axum::{Router, routing::get};

    let app = Router::new().route(
        "/sse",
        get(|| async move {
            Response::builder()
                .status(StatusCode::GATEWAY_TIMEOUT)
                .body(Body::from("gateway timeout"))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{}/sse", addr);
    let client = reqwest::Client::new();
    let url_clone = url.clone();
    let build_request =
        move || -> Result<reqwest::RequestBuilder, LlmError> { Ok(client.get(url_clone.clone())) };

    let converter = DeltaOnlyConverter;
    let res = StreamFactory::create_eventsource_stream_with_retry(
        "factory-test",
        &url,
        true,
        build_request,
        converter,
        &[], // No interceptors in test
        make_ctx("factory-test", &url),
    )
    .await;

    match res {
        Err(LlmError::ApiError { code, .. }) => {
            assert_eq!(code, 504);
        }
        Ok(_) => panic!("expected ApiError 504, got Ok"),
        Err(e) => panic!("expected ApiError 504, got: {e:?}"),
    }

    drop(server);
}
