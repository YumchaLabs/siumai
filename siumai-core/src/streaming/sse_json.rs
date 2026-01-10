//! SSE JSON streaming helpers
//!
//! This module provides small utilities for providers that emit JSON objects
//! as SSE `data:` payloads (one JSON object per SSE message).
//!
//! It is intentionally protocol-agnostic: providers can parse the returned JSON
//! into their own event enums.

use crate::error::LlmError;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::streaming::SseStreamExt;
use futures_util::Stream;
use futures_util::StreamExt;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SseJsonStreamConfig {
    /// Label used in error messages (e.g. "openai speech" / "gemini").
    pub label: String,
    /// SSE `data` payloads that indicate end-of-stream and should be ignored.
    pub done_markers: Vec<String>,
}

impl SseJsonStreamConfig {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            done_markers: vec!["[DONE]".to_string()],
        }
    }
}

pub type JsonSseStream =
    Pin<Box<dyn Stream<Item = Result<serde_json::Value, LlmError>> + Send + Sync>>;

/// Convert a bytes stream into a JSON stream by parsing SSE `data:` payloads.
///
/// - Calls `HttpInterceptor::on_sse_event` for every SSE event.
/// - Ignores empty payloads and configurable done markers.
/// - Parses JSON strictly (`serde_json::from_str`).
pub fn stream_sse_json_values<S, B>(
    byte_stream: S,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    ctx: HttpRequestContext,
    cfg: SseJsonStreamConfig,
) -> JsonSseStream
where
    S: Stream<Item = Result<B, LlmError>> + Send + Sync + Unpin + 'static,
    B: AsRef<[u8]> + Send + Sync + 'static,
{
    let done_markers = cfg.done_markers;
    let label = cfg.label;

    let out = async_stream::stream! {
        let mut sse_stream = byte_stream.into_sse_stream();

        while let Some(item) = sse_stream.next().await {
            let event = match item {
                Ok(ev) => ev,
                Err(e) => {
                    yield Err(LlmError::StreamError(format!("SSE stream error ({label}): {e}")));
                    return;
                }
            };

            for it in &interceptors {
                if let Err(e) = it.on_sse_event(&ctx, &event) {
                    yield Err(e);
                    return;
                }
            }

            let data = event.data.trim();
            if data.is_empty() {
                continue;
            }
            if done_markers.iter().any(|m| m == data) {
                continue;
            }

            let payload: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(e) => {
                    yield Err(LlmError::ParseError(format!(
                        "Failed to parse SSE JSON ({label}): {e}"
                    )));
                    return;
                }
            };

            yield Ok(payload);
        }
    };

    Box::pin(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct CountInterceptor(Arc<Mutex<usize>>);
    impl HttpInterceptor for CountInterceptor {
        fn on_sse_event(
            &self,
            _ctx: &HttpRequestContext,
            _event: &eventsource_stream::Event,
        ) -> Result<(), LlmError> {
            *self.0.lock().unwrap() += 1;
            Ok(())
        }
    }

    fn ctx() -> HttpRequestContext {
        HttpRequestContext {
            request_id: "test".to_string(),
            provider_id: "test".to_string(),
            url: "http://example.invalid".to_string(),
            stream: true,
        }
    }

    #[tokio::test]
    async fn parses_json_events_and_calls_interceptors() {
        let data: Vec<Result<&[u8], LlmError>> = vec![
            Ok(b": keep-alive\n\n".as_slice()),
            Ok(b"data: {\"a\":1}\n\n".as_slice()),
            Ok(b"data: [DONE]\n\n".as_slice()),
            Ok(b"data: {\"b\":2}\n\n".as_slice()),
        ];

        let seen = Arc::new(Mutex::new(0usize));
        let it: Arc<dyn HttpInterceptor> = Arc::new(CountInterceptor(seen.clone()));

        let mut stream = stream_sse_json_values(
            futures_util::stream::iter(data),
            vec![it],
            ctx(),
            SseJsonStreamConfig::new("test"),
        );

        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item.expect("json"));
        }

        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["a"], 1);
        assert_eq!(out[1]["b"], 2);

        // The interceptor should see both data events (keep-alive comment may or may not surface).
        assert!(*seen.lock().unwrap() >= 2);
    }

    #[tokio::test]
    async fn returns_parse_error_on_invalid_json() {
        let data: Vec<Result<&[u8], LlmError>> = vec![Ok(b"data: {not-json}\n\n".as_slice())];
        let mut stream = stream_sse_json_values(
            futures_util::stream::iter(data),
            vec![],
            ctx(),
            SseJsonStreamConfig::new("test"),
        );

        let err = stream.next().await.expect("one").expect_err("err");
        match err {
            LlmError::ParseError(msg) => assert!(msg.contains("Failed to parse SSE JSON")),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
