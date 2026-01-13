//! Factory-level test: non-SSE fallback should still produce ContentDelta + StreamEnd.
//! We simulate a server that returns `application/json` (no SSE) on a streaming
//! endpoint and verify that the StreamFactory fallback kicks in.

use axum::{Router, routing::post};
use futures::StreamExt;
use siumai::prelude::*;
use std::net::SocketAddr;
use tokio::task::JoinHandle;

async fn spawn_json_server() -> (SocketAddr, JoinHandle<()>) {
    async fn chat_handler() -> axum::response::Response {
        // Minimal JSON body compatible with OpenAI Responses transformer.
        let body = serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "object": "response",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            { "type": "output_text", "text": "Hello", "annotations": [], "logprobs": [] }
                        ]
                    }
                ],
                "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
            }
        });
        axum::response::Response::builder()
            .status(200)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                serde_json::to_string(&body).unwrap(),
            ))
            .unwrap()
    }

    let app = Router::new().route("/v1/responses", post(chat_handler));
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, handle)
}

#[tokio::test]
async fn non_sse_json_fallback_yields_delta_and_end() {
    let (addr, _jh) = spawn_json_server().await;
    let base = format!("http://{}:{}{}", addr.ip(), addr.port(), "");
    let base_v1 = format!("{}/v1", base);

    // Build OpenAI client pointing to local server. We will call chat_stream,
    // but the server returns JSON instead of SSE. The StreamFactory should
    // fallback and still yield content + end.
    let client = Provider::openai()
        .api_key("test-key")
        .base_url(base_v1.clone())
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("build client");

    let msgs = vec![user!("hi")];
    let mut stream = client.chat_stream(msgs, None).await.expect("chat_stream");

    let mut deltas = Vec::new();
    let mut saw_end = false;
    while let Some(ev) = stream.next().await {
        match ev.expect("event ok") {
            ChatStreamEvent::ContentDelta { delta, .. } => deltas.push(delta),
            ChatStreamEvent::StreamEnd { .. } => {
                saw_end = true;
                break;
            }
            _ => {}
        }
    }

    assert!(
        !deltas.is_empty(),
        "should have content deltas on JSON fallback"
    );
    assert!(saw_end, "should emit StreamEnd on JSON fallback");
    let text = deltas.join("");
    assert_eq!(text, "Hello");
}
