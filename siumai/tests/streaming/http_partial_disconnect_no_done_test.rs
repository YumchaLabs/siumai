use axum::body::Bytes;
use axum::http::HeaderValue;
use axum::response::Response;
use axum::{Router, routing::post};
use futures::StreamExt;
use std::time::Duration;

#[tokio::test]
async fn http_partial_disconnect_without_done_yields_no_stream_end() {
    // SSE route that sends one content delta then closes without [DONE]
    let app = Router::new().route(
        "/chat/completions",
        post(|| async move {
            // Minimal OpenAI-compatible chunk with content delta
            let chunk = format!(
                "data: {}\n\n",
                r#"{"id":"chatcmpl-test","model":"gpt-4","created":1731234567,"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#
            );
            let stream = async_stream::stream! {
                yield Ok::<Bytes, std::io::Error>(Bytes::from(chunk));
                // No [DONE]; end of body to simulate abrupt disconnect
                tokio::time::sleep(Duration::from_millis(10)).await;
            };
            Response::builder()
                .header("content-type", HeaderValue::from_static("text/event-stream"))
                .body(axum::body::Body::from_stream(stream))
                .unwrap()
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Build client pointing to the mock server
    use siumai::prelude::*;
    let base_url = format!("http://{}:{}", addr.ip(), addr.port());
    let client = Siumai::builder()
        .openai_chat()
        .api_key("sk-test")
        .base_url(base_url)
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("build");

    let stream = client
        .chat_stream(vec![ChatMessage::user("hi").build()], None)
        .await
        .expect("stream start");

    let events: Vec<_> = stream.collect().await;
    // Should have at least one delta
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::ContentDelta { .. })))
    );
    // No StreamEnd because server closed without [DONE]
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::StreamEnd { .. })))
    );

    drop(server_task);
}
