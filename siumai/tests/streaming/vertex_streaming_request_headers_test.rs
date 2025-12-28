//! Verify Vertex Anthropic streaming request headers at call-site

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
    let sse =
        "data: {\"type\":\"message_start\",\"message\":{}}\n\n".to_string() + "data: [DONE]\n\n";
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from(sse))
        .unwrap()
}

#[tokio::test]
async fn vertex_anthropic_streaming_headers_default_identity() {
    let state = Arc::new(Mutex::new(Seen::default()));
    let app = {
        let s = state.clone();
        Router::new().route("/models/{*rest}", post(move |req| handler(req, s.clone())))
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Build VertexAnthropicClient directly
    let base = format!("http://{}:{}", addr.ip(), addr.port());
    let cfg = siumai::providers::anthropic_vertex::client::VertexAnthropicConfig {
        base_url: base,
        model: "claude-3-5-haiku-20241022".to_string(),
        http_config: HttpConfig::default(),
    };
    let http = reqwest::Client::new();
    let client = siumai::providers::anthropic_vertex::client::VertexAnthropicClient::new(cfg, http);

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
async fn vertex_anthropic_streaming_headers_disable_identity() {
    let state = Arc::new(Mutex::new(Seen::default()));
    let app = {
        let s = state.clone();
        Router::new().route("/models/{*rest}", post(move |req| handler(req, s.clone())))
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let base = format!("http://{}:{}", addr.ip(), addr.port());
    let http_cfg = HttpConfig {
        stream_disable_compression: false,
        ..Default::default()
    };
    let cfg = siumai::providers::anthropic_vertex::client::VertexAnthropicConfig {
        base_url: base,
        model: "claude-3-5-haiku-20241022".to_string(),
        http_config: http_cfg,
    };
    let http = reqwest::Client::new();
    let client = siumai::providers::anthropic_vertex::client::VertexAnthropicClient::new(cfg, http);

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
