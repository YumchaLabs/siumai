#![cfg(any(
    feature = "openai",
    feature = "google",
    feature = "anthropic",
    feature = "google-vertex"
))]
//! Interceptor-based request assertions on the unified Siumai interface.
//!
//! This test ensures interceptors attached via `Siumai::builder()` are applied
//! to provider requests and the expected headers/body are present for streaming.

use std::sync::{Arc, Mutex};

use reqwest::header::{ACCEPT, ACCEPT_ENCODING, HeaderMap};
use siumai::experimental::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use siumai::prelude::unified::{ChatMessage, LlmError};
use siumai::provider::SiumaiBuilder;

#[derive(Clone, Debug)]
struct Captured {
    ctx: HttpRequestContext,
    headers: HeaderMap,
    body: serde_json::Value,
}

#[derive(Clone)]
struct TestInterceptor {
    slot: Arc<Mutex<Option<Captured>>>,
}

impl TestInterceptor {
    fn new() -> (Self, Arc<Mutex<Option<Captured>>>) {
        let slot = Arc::new(Mutex::new(None));
        (Self { slot: slot.clone() }, slot)
    }
}

impl HttpInterceptor for TestInterceptor {
    fn on_before_send(
        &self,
        ctx: &HttpRequestContext,
        _builder: reqwest::RequestBuilder,
        body: &serde_json::Value,
        headers: &HeaderMap,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        let mut guard = self.slot.lock().expect("failed to lock capture slot");
        *guard = Some(Captured {
            ctx: ctx.clone(),
            headers: headers.clone(),
            body: body.clone(),
        });
        drop(guard);
        Err(LlmError::UnsupportedOperation(
            "interceptor_short_circuit".to_string(),
        ))
    }
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn siumai_openai_streaming_headers_and_body() {
    let (it, captured) = TestInterceptor::new();
    let client = SiumaiBuilder::new()
        .openai()
        .api_key("sk-test")
        .model("gpt-4o-mini")
        .with_http_interceptor(Arc::new(it))
        .build()
        .await
        .expect("siumai openai build");

    let msgs = vec![ChatMessage::user("hi").build()];
    let res = client.chat_stream(msgs, None).await;
    assert!(res.is_err());
    let data = captured.lock().unwrap().clone().expect("captured");
    assert!(data.ctx.stream);
    let accept = data
        .headers
        .get(ACCEPT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(accept, "text/event-stream");
    let enc = data
        .headers
        .get(ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(enc.is_empty() || enc == "identity");
    let stream_flag = data
        .body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(stream_flag);
    let include_usage = data.body["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    assert!(include_usage);
}

#[cfg(feature = "google")]
#[tokio::test]
async fn siumai_gemini_streaming_headers() {
    let (it, captured) = TestInterceptor::new();
    let client = SiumaiBuilder::new()
        .gemini()
        .api_key("gcloud-test")
        .model("gemini-1.5-flash")
        .with_http_interceptor(Arc::new(it))
        .build()
        .await
        .expect("siumai gemini build");
    let msgs = vec![ChatMessage::user("hi").build()];
    let res = client.chat_stream(msgs, None).await;
    assert!(res.is_err());
    let data = captured.lock().unwrap().clone().expect("captured");
    assert!(data.ctx.stream);
    let accept = data
        .headers
        .get(ACCEPT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(accept, "text/event-stream");
    let enc = data
        .headers
        .get(ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(enc, "identity");
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn siumai_anthropic_streaming_headers() {
    let (it, captured) = TestInterceptor::new();
    let client = SiumaiBuilder::new()
        .anthropic()
        .api_key("anthropic-test")
        .model("claude-3-5-sonnet-20241022")
        .with_http_interceptor(Arc::new(it))
        .build()
        .await
        .expect("siumai anthropic build");
    let msgs = vec![ChatMessage::user("hi").build()];
    let res = client.chat_stream(msgs, None).await;
    assert!(res.is_err());
    let data = captured.lock().unwrap().clone().expect("captured");
    assert!(data.ctx.stream);
    let accept = data
        .headers
        .get(ACCEPT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(accept, "text/event-stream");
    let enc = data
        .headers
        .get(ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(enc.is_empty() || enc == "identity");
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn siumai_anthropic_vertex_streaming_headers() {
    let (it, captured) = TestInterceptor::new();
    let client = SiumaiBuilder::new()
        .provider_id("anthropic-vertex")
        .model("claude-3-5-sonnet-20241022")
        .http_header("authorization", "Bearer test-token")
        .base_url("https://us-central1-aiplatform.googleapis.com/v1beta/projects/demo/locations/us-central1/publishers/anthropic")
        .with_http_interceptor(Arc::new(it))
        .build()
        .await
        .expect("siumai anthropic-vertex build");
    let msgs = vec![ChatMessage::user("hi").build()];
    let res = client.chat_stream(msgs, None).await;
    assert!(res.is_err());
    let data = captured.lock().unwrap().clone().expect("captured");
    assert!(data.ctx.stream);
    let accept = data
        .headers
        .get(ACCEPT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(accept, "text/event-stream");
    let enc = data
        .headers
        .get(ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(enc.is_empty() || enc == "identity");
}
