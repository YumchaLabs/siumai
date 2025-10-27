use axum::{routing::post, Router};
use axum::http::HeaderValue;
use axum::response::Response;
use bytes::Bytes;
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
    let server_task = tokio::spawn(axum::serve(listener, app));

    use siumai::prelude::*;
    let base_url = format!("http://{}:{}", addr.ip(), addr.port());
    let client = Siumai::builder()
        .openai()
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

    futures::pin_mut!(handle.stream);

    // Receive first delta ("A") then cancel immediately
    let first = handle.stream.next().await.expect("first").expect("ok");
    match first {
        ChatStreamEvent::ContentDelta { ref delta, .. } => assert_eq!(delta, "A"),
        other => panic!("expected delta, got: {other:?}"),
    }

    handle.cancel.cancel();

    // Collect remaining events — should be empty due to cancellation, even if server had more
    let rest: Vec<_> = handle.stream.collect().await;
    assert!(rest.is_empty());

    drop(server_task);
}

