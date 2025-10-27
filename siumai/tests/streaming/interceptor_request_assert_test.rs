//! Interceptor-based request assertions
//!
//! These tests install a custom HTTP interceptor that captures the request
//! headers and JSON body right before sending, then short-circuits the request
//! with an error so no network call is made. This allows verifying provider
//! request construction (headers/body) without external dependencies.

use std::sync::{Arc, Mutex};

use reqwest::header::{ACCEPT, ACCEPT_ENCODING, HeaderMap};
use siumai::builder::LlmBuilder;
use siumai::error::LlmError;
use siumai::types::ChatMessage;
use siumai::traits::ChatCapability;
use siumai::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};

#[derive(Clone, Debug)]
struct Captured {
    ctx: HttpRequestContext,
    headers: HeaderMap,
    body: serde_json::Value,
}

#[derive(Clone)]
struct TestInterceptor {
    slot: Arc<Mutex<Option<Captured>>>,
    /// If true, short-circuit the request on first on_before_send call
    abort: bool,
}

impl TestInterceptor {
    fn new() -> (Self, Arc<Mutex<Option<Captured>>>) {
        let slot = Arc::new(Mutex::new(None));
        (
            Self {
                slot: slot.clone(),
                abort: true,
            },
            slot,
        )
    }
}

impl HttpInterceptor for TestInterceptor {
    fn on_before_send(
        &self,
        ctx: &HttpRequestContext,
        builder: reqwest::RequestBuilder,
        body: &serde_json::Value,
        headers: &HeaderMap,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        let mut guard = self
            .slot
            .lock()
            .expect("failed to lock capture slot in interceptor");
        *guard = Some(Captured {
            ctx: ctx.clone(),
            headers: headers.clone(),
            body: body.clone(),
        });
        drop(guard);
        if self.abort {
            return Err(LlmError::UnsupportedOperation(
                "interceptor_short_circuit".to_string(),
            ));
        }
        Ok(builder)
    }
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_streaming_request_headers_and_body_inserted() {
    let (it, captured) = TestInterceptor::new();
    let client = LlmBuilder::new()
        .with_http_interceptor(Arc::new(it))
        .openai()
        .api_key("sk-test")
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("openai builder should build with test key");

    let msgs = vec![ChatMessage::user("hello").build()];
    let result = client.chat_stream(msgs, None).await;
    assert!(result.is_err(), "interceptor must short-circuit the request");

    let data = captured
        .lock()
        .unwrap()
        .clone()
        .expect("interceptor should have captured data");
    assert!(data.ctx.stream, "should be streaming context");
    // Headers
    let accept = data
        .headers
        .get(ACCEPT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(accept, "text/event-stream", "Accept must be SSE for streaming");
    let enc = data
        .headers
        .get(ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    // Allow empty or identity in short-circuit environment
    assert!(enc.is_empty() || enc == "identity", "Accept-Encoding should be identity or empty in tests");
    // Body
    let stream_flag = data.body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    assert!(stream_flag, "OpenAI body must include stream=true");
    let include_usage = data.body["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    assert!(include_usage, "OpenAI stream_options.include_usage must be true");
}

#[cfg(feature = "google")]
#[tokio::test]
async fn gemini_streaming_request_headers() {
    let (it, captured) = TestInterceptor::new();
    let client = LlmBuilder::new()
        .with_http_interceptor(Arc::new(it))
        .gemini()
        .api_key("gcloud-test")
        .model("gemini-1.5-flash")
        .build()
        .await
        .expect("gemini builder should build with test key");
    let msgs = vec![ChatMessage::user("hello").build()];
    let result = client.chat_stream(msgs, None).await;
    assert!(result.is_err());
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

#[cfg(feature = "groq")]
#[tokio::test]
async fn groq_streaming_request_headers() {
    let (it, captured) = TestInterceptor::new();
    let client = LlmBuilder::new()
        .with_http_interceptor(Arc::new(it))
        .groq()
        .api_key("groq-test")
        .model("llama-3.3-70b-versatile")
        .build()
        .await
        .expect("groq builder should build with test key");
    let msgs = vec![ChatMessage::user("hello").build()];
    let result = client.chat_stream(msgs, None).await;
    assert!(result.is_err());
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

#[cfg(feature = "xai")]
#[tokio::test]
async fn xai_streaming_request_headers() {
    let (it, captured) = TestInterceptor::new();
    let client = LlmBuilder::new()
        .with_http_interceptor(Arc::new(it))
        .xai()
        .api_key("xai-test")
        .model("grok-3-latest")
        .build()
        .await
        .expect("xai builder should build with test key");
    let msgs = vec![ChatMessage::user("hello").build()];
    let result = client.chat_stream(msgs, None).await;
    assert!(result.is_err());
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

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_compatible_deepseek_streaming_request_headers() {
    let (it, captured) = TestInterceptor::new();
    let client = LlmBuilder::new()
        .with_http_interceptor(Arc::new(it))
        .deepseek()
        .api_key("deepseek-test")
        // Use default model for deepseek if available; otherwise set explicitly
        .model("deepseek-chat")
        .build()
        .await
        .expect("deepseek builder should build with test key");
    let msgs = vec![ChatMessage::user("hello").build()];
    let result = client.chat_stream(msgs, None).await;
    assert!(result.is_err());
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
