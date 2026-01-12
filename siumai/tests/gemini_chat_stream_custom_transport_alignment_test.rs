#![cfg(feature = "google")]

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
    HttpTransportStreamResponse,
};
use siumai::prelude::unified::{
    ChatCapability, ChatMessage, ChatRequest, ChatStreamEvent, LlmError, Siumai,
};
use std::sync::{Arc, Mutex};

const GEMINI_SSE: &str = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello \"}]}}]}\n\n\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"world\"}]}}]}\n\n\
data: {\"candidates\":[{\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":5,\"totalTokenCount\":8}}\n\n";

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
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "test transport only supports streaming".to_string(),
        ))
    }

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        self.calls.lock().expect("lock").push(request);

        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );

        Ok(HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_bytes(GEMINI_SSE.as_bytes().to_vec()),
        })
    }
}

#[tokio::test]
async fn gemini_chat_stream_uses_custom_transport_and_passes_request_content() {
    let transport = RecordingTransport::default();
    let transport_arc: Arc<dyn HttpTransport> = Arc::new(transport.clone());

    let client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url("https://example.invalid/v1beta")
        .model("gemini-2.5-pro")
        .fetch(transport_arc)
        .build()
        .await
        .expect("build ok");

    let req = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .model("gemini-2.5-pro")
        .stream(true)
        .build();

    let mut stream = client.chat_stream_request(req).await.expect("stream ok");

    let mut deltas = Vec::new();
    let mut saw_end = false;
    while let Some(item) = stream.next().await {
        let evt = item.expect("event ok");
        match evt {
            ChatStreamEvent::ContentDelta { delta, .. } => deltas.push(delta),
            ChatStreamEvent::StreamEnd { .. } => {
                saw_end = true;
                break;
            }
            _ => {}
        }
    }

    assert_eq!(deltas.join(""), "Hello world");
    assert!(saw_end, "expected StreamEnd event");

    let calls = transport.calls();
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(
        call.url,
        "https://example.invalid/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
    );
    assert_eq!(
        call.headers
            .get("x-goog-api-key")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
        "test-key"
    );
    assert_eq!(
        call.headers
            .get("accept")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
        "text/event-stream"
    );
    assert_eq!(
        call.body
            .get("contents")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("parts"))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "hi"
    );
}
