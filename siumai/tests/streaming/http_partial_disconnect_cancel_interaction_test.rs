use axum::body::Bytes;
use axum::http::HeaderValue;
use axum::response::Response;
use axum::{Router, routing::post};
use futures::StreamExt;
use std::time::Duration;

#[tokio::test]
async fn partial_disconnect_then_immediate_cancel() {
    // SSE route that sends two deltas then closes without [DONE]
    let app = Router::new().route(
        "/chat/completions",
        post(|| async move {
            let chunk1 = format!(
                "data: {}\n\n",
                r#"{"id":"chatcmpl-test","model":"gpt-4","created":1731234567,"choices":[{"index":0,"delta":{"content":"A"}}]}"#
            );
            let chunk2 = format!(
                "data: {}\n\n",
                r#"{"choices":[{"index":0,"delta":{"content":"B"}}]}"#
            );
            let stream = async_stream::stream! {
                yield Ok::<Bytes, std::io::Error>(Bytes::from(chunk1));
                tokio::time::sleep(Duration::from_millis(20)).await;
                yield Ok::<Bytes, std::io::Error>(Bytes::from(chunk2));
                // No [DONE]; body ends here to simulate abrupt disconnect shortly after
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

    let handle = client
        .chat_stream_with_cancel(vec![ChatMessage::user("hi").build()], None)
        .await
        .expect("start stream");

    let ChatStreamHandle { mut stream, cancel } = handle;

    // Receive first content delta ("A") then cancel immediately
    let mut first_delta: Option<String> = None;
    while let Some(ev) = stream.next().await {
        if let ChatStreamEvent::ContentDelta { delta, .. } = ev.expect("ok") {
            first_delta = Some(delta);
            break;
        }
    }
    assert_eq!(first_delta.as_deref(), Some("A"));

    cancel.cancel();

    // Collect remaining events — should be empty due to cancellation, even if server had more
    let rest: Vec<_> = stream.collect().await;
    assert!(rest.is_empty());

    drop(server_task);
}
