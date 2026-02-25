use axum::http::{HeaderValue, StatusCode};
use axum::response::Response;
use axum::{Router, routing::post};
use futures::StreamExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

#[tokio::test]
async fn http_connect_or_send_timeout_then_retry_success() {
    // Shared attempt counter in server state
    let attempts = Arc::new(AtomicU32::new(0));
    let attempts_clone = attempts.clone();

    // Define handler for POST /chat/completions
    let app = Router::new().route(
        "/chat/completions",
        post(move || {
            let attempts = attempts_clone.clone();
            async move {
                let n = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if n == 1 {
                    // Simulate a slow/timeout response (no headers for a while)
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(axum::body::Body::from("fail"))
                        .unwrap()
                } else {
                    // Return a minimal SSE body with one content delta and [DONE]
                    let sse_body = format!(
                        "{}{}",
                        "data: {\"id\":\"chatcmpl-test\",\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
                        "data: [DONE]\n\n"
                    );
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", HeaderValue::from_static("text/event-stream"))
                        .body(axum::body::Body::from(sse_body))
                        .unwrap()
                }
            }
        }),
    );

    // Bind to an ephemeral local port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Build client that targets our test server (OpenAI native path)
    use siumai::prelude::*;
    let base_url = format!("http://{}:{}", addr.ip(), addr.port());
    let client = Siumai::builder()
        .openai_chat()
        .api_key("sk-test")
        .base_url(base_url)
        .model("gpt-4o-mini")
        // Set a short timeout to trigger the first-attempt timeout
        .http_timeout(Duration::from_millis(50))
        .with_retry(RetryOptions::policy_default().with_max_attempts(3))
        .build()
        .await
        .expect("build");

    // Start streaming - first attempt will time out, second should succeed
    let stream = client
        .chat_stream(vec![ChatMessage::user("hi").build()], None)
        .await
        .expect("stream after retry");

    let events: Vec<_> = stream.collect().await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::ContentDelta { .. }))),
        "expected at least one ContentDelta"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. }))),
        "expected StreamEnd"
    );

    // Shutdown server
    drop(server_task);
}
