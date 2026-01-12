#![cfg(feature = "anthropic")]

use async_trait::async_trait;
use reqwest::header::HeaderMap;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamResponse,
};
use siumai::prelude::unified::{ChatCapability, ChatMessage, LlmError, Siumai};
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct RecordingTransport {
    calls: Arc<Mutex<Vec<HttpTransportRequest>>>,
}

impl RecordingTransport {
    fn calls(&self) -> Vec<HttpTransportRequest> {
        self.calls.lock().expect("lock").clone()
    }
}

#[async_trait]
impl HttpTransport for RecordingTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        self.calls.lock().expect("lock").push(request);

        let json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [{ "type": "text", "text": "Hello from transport" }],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 3, "output_tokens": 5 }
        });

        Ok(HttpTransportResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: serde_json::to_vec(&json).expect("json bytes"),
        })
    }

    async fn execute_stream(
        &self,
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "test transport only supports json".to_string(),
        ))
    }
}

#[tokio::test]
async fn anthropic_chat_uses_custom_transport_and_passes_request_content() {
    let transport = RecordingTransport::default();
    let transport_arc: Arc<dyn HttpTransport> = Arc::new(transport.clone());

    let client = Siumai::builder()
        .anthropic()
        .api_key("test-key")
        .base_url("https://example.invalid")
        .model("claude-3-5-haiku-20241022")
        .max_tokens(16)
        .fetch(transport_arc)
        .build()
        .await
        .expect("build ok");

    let resp = client
        .chat(vec![ChatMessage::user("hi").build()])
        .await
        .expect("chat ok");
    assert_eq!(
        resp.content_text().unwrap_or_default(),
        "Hello from transport"
    );

    let calls = transport.calls();
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.url, "https://example.invalid/v1/messages");
    assert_eq!(
        call.headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
        "test-key"
    );
    assert_eq!(
        call.headers
            .get("anthropic-version")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
        "2023-06-01"
    );
    assert_eq!(
        call.body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "claude-3-5-haiku-20241022"
    );
    assert_eq!(
        call.body
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or_default(),
        16
    );
    assert_eq!(
        call.body
            .get("messages")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "hi"
    );
}
