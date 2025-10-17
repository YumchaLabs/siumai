//! Verify OpenAI streaming request headers and body fields at call-site
//!
//! This asserts that when using the OpenAI client in streaming mode:
//! - Request headers include `Accept: text/event-stream`
//! - Request headers include `Accept-Encoding: identity` (disable compression)
//! - JSON body includes `stream: true` and `stream_options.include_usage: true`

use axum::{routing::post, Router};
use axum::body;
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use futures_util::StreamExt;
use siumai::prelude::*;
use siumai::stream::ChatStreamEvent;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Default, Clone, Debug)]
struct SeenState {
    accept_is_sse: bool,
    accept_encoding_identity: bool,
    body_stream_true: bool,
    body_include_usage_true: bool,
}

async fn handler(req: Request, state: Arc<Mutex<SeenState>>) -> Response {
    let (parts, body) = req.into_parts();
    let headers: &HeaderMap = &parts.headers;

    let mut seen = SeenState::default();

    // Check headers (case-insensitive)
    if let Some(v) = headers.get("accept") {
        seen.accept_is_sse = v == HeaderValue::from_static("text/event-stream");
    }
    if let Some(v) = headers.get("accept-encoding") {
        seen.accept_encoding_identity = v == HeaderValue::from_static("identity");
    }

    // Read body and check JSON flags
    let body_bytes = body::to_bytes(body, 64 * 1024).await.unwrap_or_default();
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
        seen.body_stream_true = json.get("stream").and_then(|v| v.as_bool()) == Some(true);
        seen.body_include_usage_true = json
            .get("stream_options")
            .and_then(|v| v.get("include_usage"))
            .and_then(|v| v.as_bool())
            == Some(true);
    }

    // Persist markers
    {
        let mut guard = state.lock().await;
        *guard = seen;
    }

    // Minimal SSE payload (text/event-stream)
    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: [DONE]\n\n",
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from(sse_body))
        .unwrap()
}

#[tokio::test]
async fn openai_streaming_request_includes_sse_headers_and_stream_options() {
    // Start local server
    let state = Arc::new(Mutex::new(SeenState::default()));
    let app = {
        let state = state.clone();
        Router::new().route(
            "/v1/chat/completions",
            post(move |req| handler(req, state.clone())),
        )
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Build OpenAI client pointing to local server
    let base = format!("http://{}:{}", addr.ip(), addr.port());
    let client = Provider::openai()
        .api_key("test-key")
        .base_url(format!("{}/v1", base))
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("client");

    // Issue streaming request and consume stream quickly
    let messages = vec![system!("You are a test."), user!("Hi")];
    let mut stream = client
        .chat_stream(messages, None)
        .await
        .expect("stream ok");

    // Drain until end
    while let Some(ev) = stream.next().await {
        match ev.expect("event ok") {
            ChatStreamEvent::StreamEnd { .. } => break,
            _ => {}
        }
    }

    // Assert recorded request had expected headers/body fields
    let seen = state.lock().await.clone();
    assert!(seen.accept_is_sse, "Accept header should be text/event-stream");
    assert!(
        seen.accept_encoding_identity,
        "Accept-Encoding should be identity to disable compression"
    );
    assert!(seen.body_stream_true, "Body.stream should be true");
    assert!(
        seen.body_include_usage_true,
        "Body.stream_options.include_usage should be true"
    );

    // Shutdown server
    drop(server);
}

#[tokio::test]
async fn openai_streaming_respects_disable_compression_flag() {
    // Start local server
    let state = Arc::new(Mutex::new(SeenState::default()));
    let app = {
        let state = state.clone();
        Router::new().route(
            "/v1/chat/completions",
            post(move |req| handler(req, state.clone())),
        )
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Build OpenAI client pointing to local server, but disable compression opt-out
    let base = format!("http://{}:{}", addr.ip(), addr.port());
    let client = Provider::openai()
        .api_key("test-key")
        .base_url(format!("{}/v1", base))
        .http_stream_disable_compression(false)
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("client");

    // Issue streaming request and consume stream quickly
    let messages = vec![system!("You are a test."), user!("Hi")];
    let mut stream = client
        .chat_stream(messages, None)
        .await
        .expect("stream ok");

    while let Some(ev) = stream.next().await {
        match ev.expect("event ok") {
            ChatStreamEvent::StreamEnd { .. } => break,
            _ => {}
        }
    }

    // Assert Accept-Encoding did not force identity
    let seen = state.lock().await.clone();
    assert!(seen.accept_is_sse, "Accept header should be text/event-stream");
    assert!(
        !seen.accept_encoding_identity,
        "Accept-Encoding identity should be absent/disabled when flag is false"
    );

    drop(server);
}
