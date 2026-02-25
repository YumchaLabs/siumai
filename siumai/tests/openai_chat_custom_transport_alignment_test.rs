#![cfg(feature = "openai")]

use async_trait::async_trait;
use reqwest::header::HeaderMap;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamResponse,
};
use siumai::prelude::unified::{ChatCapability, ChatMessage, ChatRequest, LlmError, Siumai};
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
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": { "role": "assistant", "content": "Hello from transport" },
                    "finish_reason": "stop"
                }
            ],
            "usage": { "prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4 }
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
async fn openai_chat_uses_custom_transport_and_passes_request_content() {
    let transport = RecordingTransport::default();
    let transport_arc: Arc<dyn HttpTransport> = Arc::new(transport.clone());

    let client = Siumai::builder()
        .openai_chat()
        .api_key("sk-test")
        .base_url("https://example.invalid/v1")
        .model("gpt-4o-mini")
        .fetch(transport_arc)
        .build()
        .await
        .expect("build ok");

    let req = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .model("gpt-4o-mini")
        .build();

    let resp = client.chat_request(req).await.expect("chat ok");
    assert_eq!(
        resp.content_text().unwrap_or_default(),
        "Hello from transport"
    );

    let calls = transport.calls();
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.url, "https://example.invalid/v1/chat/completions");
    assert_eq!(
        call.headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
        "Bearer sk-test"
    );
    assert_eq!(
        call.body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "gpt-4o-mini"
    );
    assert_eq!(
        call.body
            .get("messages")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "hi"
    );
}
