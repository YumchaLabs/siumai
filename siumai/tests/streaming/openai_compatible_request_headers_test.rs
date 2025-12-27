//! Verify OpenAI-compatible (e.g., DeepSeek) streaming request headers

use axum::body;
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use axum::{Router, routing::post};
use futures_util::StreamExt;
use siumai::prelude::*;
use siumai::streaming::ChatStreamEvent;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Default, Clone, Debug)]
struct Seen {
    accept_is_sse: bool,
    accept_encoding_identity: bool,
}

async fn handler(req: Request, state: Arc<Mutex<Seen>>) -> Response {
    let (parts, body_in) = req.into_parts();
    let headers: &HeaderMap = &parts.headers;
    let mut seen = Seen::default();
    if let Some(v) = headers.get("accept") {
        seen.accept_is_sse = v == HeaderValue::from_static("text/event-stream");
    }
    if let Some(v) = headers.get("accept-encoding") {
        seen.accept_encoding_identity = v == HeaderValue::from_static("identity");
    }
    let _ = body::to_bytes(body_in, 64 * 1024).await.unwrap_or_default();
    {
        let mut g = state.lock().await;
        *g = seen;
    }
    let sse = "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n"
        .to_string() + "data: [DONE]\n\n";
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from(sse))
        .unwrap()
}

#[tokio::test]
async fn compat_streaming_headers_default_identity() {
    let state = Arc::new(Mutex::new(Seen::default()));
    let app = {
        let s = state.clone();
        Router::new().route(
            "/v1/chat/completions",
            post(move |req| handler(req, s.clone())),
        )
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let base = format!("http://{}:{}", addr.ip(), addr.port());
    let client = Siumai::builder()
        .deepseek()
        .api_key("test")
        .base_url(format!("{}/v1", base))
        .model("deepseek-chat")
        .build()
        .await
        .expect("client");
    let mut stream = client
        .chat_stream(vec![user!("hi")], None)
        .await
        .expect("stream ok");
    while let Some(ev) = stream.next().await {
        if matches!(ev.unwrap(), ChatStreamEvent::StreamEnd { .. }) {
            break;
        }
    }
    let seen = state.lock().await.clone();
    assert!(seen.accept_is_sse);
    assert!(seen.accept_encoding_identity);
    drop(server);
}

#[tokio::test]
async fn compat_streaming_headers_disable_identity() {
    let state = Arc::new(Mutex::new(Seen::default()));
    let app = {
        let s = state.clone();
        Router::new().route(
            "/v1/chat/completions",
            post(move |req| handler(req, s.clone())),
        )
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let base = format!("http://{}:{}", addr.ip(), addr.port());
    let client = Siumai::builder()
        .deepseek()
        .api_key("test")
        .base_url(format!("{}/v1", base))
        .http_stream_disable_compression(false)
        .model("deepseek-chat")
        .build()
        .await
        .expect("client");
    let mut stream = client
        .chat_stream(vec![user!("hi")], None)
        .await
        .expect("stream ok");
    while let Some(ev) = stream.next().await {
        if matches!(ev.unwrap(), ChatStreamEvent::StreamEnd { .. }) {
            break;
        }
    }
    let seen = state.lock().await.clone();
    assert!(seen.accept_is_sse);
    assert!(!seen.accept_encoding_identity);
    drop(server);
}
