//! Stream Factory
//!
//! Factory for creating provider-specific streaming implementations.
//! Handles SSE and JSON-based streaming with proper UTF-8 handling and error classification.

use crate::error::LlmError;
use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use crate::streaming::{
    ChatStream, ChatStreamEvent, JsonEventConverter, SseEventConverter, SseStreamExt,
};
use futures_util::StreamExt;
use futures_util::TryStreamExt;
use std::sync::Arc;

/// Stream Factory
///
/// Provides utilities for creating SSE and JSON streams with proper UTF-8 handling,
/// error classification, and automatic retry logic.
pub struct StreamFactory;

impl StreamFactory {
    /// Convert an HTTP response into a ChatStream, using SSE when available,
    /// and falling back to a single JSON body conversion when not SSE.
    async fn stream_from_response_with_sse_fallback<C>(
        _provider_id: &str,
        response: reqwest::Response,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        C: SseEventConverter + Clone + Send + 'static,
    {
        // If server didn't return SSE, fall back to single JSON body conversion
        let is_sse = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.to_ascii_lowercase().contains("text/event-stream"))
            .unwrap_or(false);

        if !is_sse {
            let text = response
                .text()
                .await
                .map_err(|e| LlmError::HttpError(format!("Failed to read body: {e}")))?;
            let evt = eventsource_stream::Event {
                event: "message".to_string(),
                data: text,
                id: "0".to_string(),
                retry: None,
            };
            let mut events = converter.clone().convert_event(evt).await;
            let saw_content = events
                .iter()
                .any(|ev| matches!(ev, Ok(ChatStreamEvent::ContentDelta { .. })));
            if let Some(end) = converter.handle_stream_end() {
                if let Ok(ChatStreamEvent::StreamEnd { response }) = &end {
                    if !saw_content
                        && let Some(text) = response.content_text()
                        && !text.is_empty()
                    {
                        events.push(Ok(ChatStreamEvent::ContentDelta {
                            delta: text.to_string(),
                            index: None,
                        }));
                    }
                    events.push(end);
                } else {
                    events.push(end);
                }
            }
            let stream = futures::stream::iter(events);
            return Ok(Box::pin(stream));
        }

        // Convert to byte stream and then to SSE
        let byte_stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        let sse_stream = byte_stream.into_sse_stream();

        // Track whether any ContentDelta was emitted
        let saw_content = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let chat_stream = sse_stream
            .then(move |event_result| {
                let converter = converter.clone();
                let saw_content = saw_content.clone();
                async move {
                    match event_result {
                        Ok(event) => {
                            if event.data.trim() == "[DONE]" {
                                if let Some(end) = converter.handle_stream_end() {
                                    // Inject synthetic ContentDelta if no deltas were seen
                                    if let Ok(ChatStreamEvent::StreamEnd { response }) = &end
                                        && !saw_content.load(std::sync::atomic::Ordering::Relaxed)
                                        && let Some(text) = response.content_text()
                                        && !text.is_empty()
                                    {
                                        return vec![
                                            Ok(ChatStreamEvent::ContentDelta {
                                                delta: text.to_string(),
                                                index: None,
                                            }),
                                            end,
                                        ];
                                    }
                                    return vec![end];
                                }
                                return vec![];
                            }
                            if event.data.trim().is_empty() {
                                return vec![];
                            }
                            let events = converter.convert_event(event).await;
                            // Mark if any ContentDelta is present
                            let has_content = events
                                .iter()
                                .any(|ev| matches!(ev, Ok(ChatStreamEvent::ContentDelta { .. })));
                            if has_content {
                                saw_content.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            events
                        }
                        Err(e) => {
                            vec![Err(LlmError::StreamError(format!(
                                "SSE parsing error: {e}"
                            )))]
                        }
                    }
                }
            })
            .flat_map(futures::stream::iter);
        Ok(Box::pin(chat_stream))
    }

    /// Create a chat stream with optional 401 retry and error classification.
    ///
    /// The `build_request` closure must construct a fresh RequestBuilder each call
    /// with up-to-date headers (e.g., refreshed Bearer token). On non-401 errors,
    /// this method classifies the error using `retry_api::classify_http_error`.
    ///
    /// # Arguments
    /// * `provider_id` - Provider identifier for error classification
    /// * `url` - Request URL for interceptor context
    /// * `retry_401` - Whether to retry 401 errors with rebuilt request
    /// * `build_request` - Closure that builds a fresh request (called on retry)
    /// * `converter` - SSE event converter for this provider
    /// * `interceptors` - HTTP interceptors to call on retry/error
    ///
    /// # Returns
    /// A ChatStream that yields ChatStreamEvents
    pub async fn create_eventsource_stream_with_retry<B, C>(
        provider_id: &str,
        _url: &str,
        retry_401: bool,
        build_request: B,
        converter: C,
        interceptors: &[Arc<dyn HttpInterceptor>],
        ctx: HttpRequestContext,
    ) -> Result<ChatStream, LlmError>
    where
        B: Fn() -> Result<reqwest::RequestBuilder, LlmError>,
        C: SseEventConverter + Clone + Send + 'static,
    {
        Self::create_eventsource_stream_with_retry_options(
            provider_id,
            _url,
            retry_401,
            build_request,
            converter,
            interceptors,
            ctx,
            None,
        )
        .await
    }

    /// Create a chat stream with retry options (transport-level retries).
    ///
    /// This extends `create_eventsource_stream_with_retry` by optionally retrying
    /// request send failures (timeouts / connection failures) for streaming handshake.
    /// It keeps the existing semantics for HTTP status codes: only 401 is retried here.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_eventsource_stream_with_retry_options<B, C>(
        provider_id: &str,
        _url: &str,
        retry_401: bool,
        build_request: B,
        converter: C,
        interceptors: &[Arc<dyn HttpInterceptor>],
        ctx: HttpRequestContext,
        retry_options: Option<crate::retry_api::RetryOptions>,
    ) -> Result<ChatStream, LlmError>
    where
        B: Fn() -> Result<reqwest::RequestBuilder, LlmError>,
        C: SseEventConverter + Clone + Send + 'static,
    {
        fn map_send_error(e: reqwest::Error) -> LlmError {
            if e.is_timeout() {
                return LlmError::TimeoutError(format!("Request timed out: {e}"));
            }
            if e.is_connect() {
                return LlmError::ConnectionError(format!("Connection error: {e}"));
            }
            LlmError::HttpError(format!("Failed to send request: {e}"))
        }

        fn max_attempts(opts: &crate::retry_api::RetryOptions) -> u32 {
            match opts.backend {
                crate::retry_api::RetryBackend::Policy => {
                    opts.policy.as_ref().map(|p| p.max_attempts).unwrap_or(1)
                }
                // Keep backoff-based behavior unchanged for streaming handshake.
                crate::retry_api::RetryBackend::Backoff => 1,
            }
        }

        let retry_cfg = retry_options.as_ref();
        let allow_transport_retry = retry_cfg.map(|o| o.idempotent).unwrap_or(false);
        let attempts: usize = retry_cfg.map(max_attempts).unwrap_or(1).max(1) as usize;

        for attempt in 1..=attempts {
            // First attempt (or subsequent retries)
            let response = match build_request()?.send().await {
                Ok(r) => r,
                Err(e) => {
                    let err = map_send_error(e);

                    let should_retry = allow_transport_retry
                        && attempt < attempts
                        && matches!(
                            err,
                            LlmError::TimeoutError(_)
                                | LlmError::ConnectionError(_)
                                | LlmError::HttpError(_)
                        );

                    if should_retry {
                        for interceptor in interceptors {
                            interceptor.on_retry(&ctx, &err, attempt);
                        }
                        continue;
                    }

                    for interceptor in interceptors {
                        interceptor.on_error(&ctx, &err);
                    }
                    return Err(err);
                }
            };

            let response = if !response.status().is_success() {
                let status = response.status();
                if status.as_u16() == 401 && retry_401 {
                    // Notify interceptors of retry
                    let retry_error = LlmError::HttpError("401 Unauthorized".to_string());
                    for interceptor in interceptors {
                        interceptor.on_retry(&ctx, &retry_error, attempt);
                    }
                    // Retry once with rebuilt headers/request
                    build_request()?.send().await.map_err(map_send_error)?
                } else {
                    let headers = response.headers().clone();
                    let text = response.text().await.unwrap_or_default();
                    let error = crate::retry_api::classify_http_error(
                        provider_id,
                        status.as_u16(),
                        &text,
                        &headers,
                        None,
                    );
                    for interceptor in interceptors {
                        interceptor.on_error(&ctx, &error);
                    }
                    return Err(error);
                }
            } else {
                response
            };

            return Self::stream_from_response_with_sse_fallback(provider_id, response, converter)
                .await;
        }

        Err(LlmError::InternalError(
            "Streaming handshake retry exhausted without error".to_string(),
        ))
    }

    /// Create a chat stream for JSON-based streaming (provider emits JSON fragments)
    ///
    /// We route the byte stream through line-delimited JSON parsing for consistent UTF-8 handling.
    /// Each line is treated as a JSON object for conversion.
    ///
    /// # Arguments
    /// * `response` - HTTP response containing JSON stream
    /// * `converter` - JSON event converter for this provider
    ///
    /// # Returns
    /// A ChatStream that yields ChatStreamEvents
    pub async fn create_json_stream<C>(
        response: reqwest::Response,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        C: JsonEventConverter + Clone + 'static,
    {
        use tokio_util::codec::{FramedRead, LinesCodec};
        use tokio_util::io::StreamReader;

        // Convert the byte stream to an AsyncRead via StreamReader
        let byte_stream = response
            .bytes_stream()
            .map_err(|e| std::io::Error::other(format!("Stream error: {e}")));
        let reader = StreamReader::new(byte_stream);
        let lines = FramedRead::new(reader, LinesCodec::new());

        // Clone converter for the end-of-stream handler
        let end_converter = converter.clone();

        let chat_stream = lines
            .map(|res| match res {
                Ok(line) => Ok(line),
                Err(e) => Err(LlmError::ParseError(format!("JSON line error: {e}"))),
            })
            .then(move |line_res| {
                let converter = converter.clone();
                async move {
                    match line_res {
                        Ok(line) => {
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                return vec![];
                            }
                            converter.convert_json(trimmed).await
                        }
                        Err(e) => vec![Err(e)],
                    }
                }
            })
            .flat_map(futures::stream::iter)
            // Chain the end-of-stream event
            .chain(futures::stream::iter(
                end_converter
                    .handle_stream_end()
                    .into_iter()
                    .collect::<Vec<_>>(),
            ));

        Ok(Box::pin(chat_stream))
    }

    /// Create a chat stream using eventsource-stream
    ///
    /// This method creates an SSE stream using the eventsource-stream crate,
    /// which handles UTF-8 boundaries, line buffering, and SSE parsing automatically.
    ///
    /// # Arguments
    /// * `request_builder` - Request builder to send
    /// * `converter` - SSE event converter for this provider
    ///
    /// # Returns
    /// A ChatStream that yields ChatStreamEvents
    pub async fn create_eventsource_stream<C>(
        request_builder: reqwest::RequestBuilder,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        C: SseEventConverter + Clone + Send + 'static,
    {
        // Reuse fallback converter to handle SSE/JSON consistently

        // Send the request and get the response
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to send request: {e}")))?;

        // Check for HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());
            return Err(LlmError::HttpError(format!("HTTP {status}: {text}")));
        }

        // Reuse SSE/JSON fallback converter for consistency
        Self::stream_from_response_with_sse_fallback("", response, converter).await
    }
}
