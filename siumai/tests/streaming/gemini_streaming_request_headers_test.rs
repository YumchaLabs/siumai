//! Verify Gemini streaming request headers at call-site

use axum::{routing::post, Router};
use axum::body;
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use futures_util::StreamExt;
use siumai::prelude::*;
use siumai::streaming::ChatStreamEvent;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Default, Clone, Debug)]
struct Seen { accept_is_sse: bool, accept_encoding_identity: bool }

async fn handler(req: Request, state: Arc<Mutex<Seen>>) -> Response {
    let (parts, body_in) = req.into_parts();
    let headers: &HeaderMap = &parts.headers;
    let mut seen = Seen::default();
    if let Some(v) = headers.get("accept") { seen.accept_is_sse = v == HeaderValue::from_static("text/event-stream"); }
    if let Some(v) = headers.get("accept-encoding") { seen.accept_encoding_identity = v == HeaderValue::from_static("identity"); }
    let _ = body::to_bytes(body_in, 64 * 1024).await.unwrap_or_default();
    { let mut g = state.lock().await; *g = seen; }
    let sse = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n\n".to_string() + "data: [DONE]\n\n";
    Response::builder().status(StatusCode::OK).header("content-type","text/event-stream").body(axum::body::Body::from(sse)).unwrap()
}

#[tokio::test]
async fn gemini_streaming_headers_default_identity() {
    let state = Arc::new(Mutex::new(Seen::default()));
    let app = { let s=state.clone(); Router::new().route("/models/*rest", post(move |req| handler(req, s.clone()))) };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let base = format!("http://{}:{}/v1beta", addr.ip(), addr.port());
    let client = Provider::gemini().api_key("").base_url(base).model("gemini-1.5-flash").build().await.expect("client");
    let mut stream = client.chat_stream(vec![user!("hi")], None).await.expect("stream ok");
    while let Some(ev) = stream.next().await { if matches!(ev.unwrap(), ChatStreamEvent::StreamEnd{..}) { break; } }
    let seen = state.lock().await.clone();
    assert!(seen.accept_is_sse);
    assert!(seen.accept_encoding_identity);
    drop(server);
}

#[tokio::test]
async fn gemini_streaming_headers_disable_identity() {
    let state = Arc::new(Mutex::new(Seen::default()));
    let app = { let s=state.clone(); Router::new().route("/models/*rest", post(move |req| handler(req, s.clone()))) };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let base = format!("http://{}:{}/v1beta", addr.ip(), addr.port());
    let client = Provider::gemini().api_key("").base_url(base).http_stream_disable_compression(false).model("gemini-1.5-flash").build().await.expect("client");
    let mut stream = client.chat_stream(vec![user!("hi")], None).await.expect("stream ok");
    while let Some(ev) = stream.next().await { if matches!(ev.unwrap(), ChatStreamEvent::StreamEnd{..}) { break; } }
    let seen = state.lock().await.clone();
    assert!(seen.accept_is_sse);
    assert!(!seen.accept_encoding_identity);
    drop(server);
}

