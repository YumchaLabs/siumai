#![cfg(feature = "google-vertex")]

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::experimental::execution::executors::chat::{ChatExecutor, ChatExecutorBuilder};
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
    HttpTransportStreamResponse,
};
use siumai::prelude::unified::{ChatMessage, ChatRequest, ChatStreamEvent, LlmError};
use std::collections::HashMap;
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
async fn vertex_chat_stream_uses_custom_transport_and_passes_request_content() {
    let transport = RecordingTransport::default();
    let transport_arc: Arc<dyn HttpTransport> = Arc::new(transport.clone());

    let base_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google";
    let mut extra_headers = HashMap::new();
    extra_headers.insert("Authorization".to_string(), "Bearer token".to_string());
    let ctx = ProviderContext::new("vertex", base_url.to_string(), None, extra_headers);
    let spec = Arc::new(
        siumai::experimental::providers::google_vertex::standards::vertex_generative_ai::VertexGenerativeAiStandard::new()
            .create_spec("vertex"),
    );

    let req = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .model("gemini-2.5-pro")
        .stream(true)
        .build();

    let bundle = spec.choose_chat_transformers(&req, &ctx);
    let exec = ChatExecutorBuilder::new("vertex", reqwest::Client::new())
        .with_spec(spec)
        .with_context(ctx)
        .with_transformer_bundle(bundle)
        .with_transport(transport_arc)
        .build();

    let mut stream = exec.execute_stream(req).await.expect("stream ok");

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
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
    );

    let ua = call
        .headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    let expected = format!("siumai/google-vertex/{}", env!("CARGO_PKG_VERSION"));
    assert!(
        ua.contains(&expected),
        "expected user-agent to contain {expected}, got {ua}"
    );

    assert_eq!(
        call.headers
            .get("accept")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
        "text/event-stream"
    );
    assert_eq!(
        call.headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
        "application/json"
    );

    assert_eq!(
        {
            let mut v = call.body.clone();
            if let Some(obj) = v.as_object_mut()
                && let Some(gc) = obj.get("generationConfig")
                && gc.as_object().is_some_and(|o| o.is_empty())
            {
                obj.remove("generationConfig");
            }
            v
        },
        serde_json::json!({
            "model": "gemini-2.5-pro",
            "contents": [
                {
                    "role": "user",
                    "parts": [{ "text": "hi" }]
                }
            ]
        })
    );
}
