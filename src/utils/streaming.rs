//! Common Streaming Utilities
//!
//! Utilities for handling streaming responses across providers, including
//! UTF-8 safe processing and unified SSE handling using `eventsource-stream`.

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::utils::sse_stream::SseStreamExt;
use eventsource_stream::Event;
use futures_util::StreamExt;
use futures_util::TryStreamExt;
use std::future::Future;
use std::pin::Pin;

/// Type alias for SSE event conversion future - now supports multiple events
type SseEventFuture<'a> =
    Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>;

/// Type alias for JSON event conversion future - now supports multiple events
type JsonEventFuture<'a> =
    Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>;

/// Trait for converting provider-specific SSE events to ChatStreamEvent
///
/// This trait now supports multi-event emission, allowing a single provider event
/// to generate multiple ChatStreamEvents (e.g., StreamStart + ContentDelta).
pub trait SseEventConverter: Send + Sync {
    /// Convert an SSE event to zero or more ChatStreamEvents
    fn convert_event(&self, event: Event) -> SseEventFuture<'_>;

    /// Handle the end of stream
    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        None
    }
}

/// Trait for converting JSON data to ChatStreamEvent (for providers like Gemini)
///
/// This trait now supports multi-event emission for JSON-based streaming.
pub trait JsonEventConverter: Send + Sync {
    /// Convert JSON data to zero or more ChatStreamEvents
    fn convert_json<'a>(&'a self, json_data: &'a str) -> JsonEventFuture<'a>;
}

/// Stream factory for creating provider-specific streams
///
/// This factory provides utilities for creating SSE and JSON streams
/// using eventsource-stream for proper UTF-8 handling.
pub struct StreamFactory;

impl StreamFactory {
    /// Create a chat stream with one-shot 401 retry and error classification.
    ///
    /// The `build_request` closure must construct a fresh RequestBuilder each call
    /// with up-to-date headers (e.g., refreshed Bearer token). On non-401 errors,
    /// this method classifies the error using `retry_api::classify_http_error`.
    pub async fn create_eventsource_stream_with_retry<B, C>(
        provider_id: &str,
        build_request: B,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        B: Fn() -> Result<reqwest::RequestBuilder, LlmError>,
        C: SseEventConverter + Clone + Send + 'static,
    {
        // First attempt
        let response = build_request()?
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to send request: {e}")))?;
        let response = if !response.status().is_success() {
            let status = response.status();
            if status.as_u16() == 401 {
                // Retry once with rebuilt headers/request
                build_request()?
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(format!("Failed to send request: {e}")))?
            } else {
                let headers = response.headers().clone();
                let text = response.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    provider_id,
                    status.as_u16(),
                    &text,
                    &headers,
                    None,
                ));
            }
        } else {
            response
        };

        // Convert to byte stream and then to SSE
        let byte_stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        let sse_stream = byte_stream.into_sse_stream();
        // Track whether any ContentDelta was emitted; used to inject a fallback delta on StreamEnd
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
                                    // If the converter provides a final response and no deltas were seen,
                                    // inject a synthetic ContentDelta before StreamEnd so downstream
                                    // consumers that accumulate deltas still see content.
                                    if let Ok(ChatStreamEvent::StreamEnd { response }) = &end {
                                        if !saw_content.load(std::sync::atomic::Ordering::Relaxed) {
                                            if let Some(text) = response.content_text() {
                                                if !text.is_empty() {
                                                    return vec![
                                                        Ok(ChatStreamEvent::ContentDelta {
                                                            delta: text.to_string(),
                                                            index: None,
                                                        }),
                                                        end,
                                                    ];
                                                }
                                            }
                                        }
                                    }
                                    return vec![end];
                                }
                                return vec![];
                            }
                            if event.data.trim().is_empty() {
                                return vec![];
                            }
                            let mut events = converter.convert_event(event).await;
                            // Mark if any ContentDelta is present
                            let has_content = events.iter().any(|ev| matches!(ev, Ok(ChatStreamEvent::ContentDelta { .. })));
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
    /// Create a chat stream for JSON-based streaming (provider emits JSON fragments)
    ///
    /// We route the byte stream through the SSE parser for consistent UTF-8 handling
    /// and treat each `event.data` as a raw JSON string for conversion. Note: Providers
    /// emitting plain line-delimited JSON may not be fully compatible with this method.
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
        let byte_stream = response.bytes_stream().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("Stream error: {e}"))
        });
        let reader = StreamReader::new(byte_stream);
        let lines = FramedRead::new(reader, LinesCodec::new());

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
            .flat_map(futures::stream::iter);

        Ok(Box::pin(chat_stream))
    }

    /// Create a chat stream using eventsource-stream
    ///
    /// This method creates an SSE stream using the eventsource-stream crate,
    /// which handles UTF-8 boundaries, line buffering, and SSE parsing automatically.
    pub async fn create_eventsource_stream<C>(
        request_builder: reqwest::RequestBuilder,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        C: SseEventConverter + Clone + Send + 'static,
    {
        use crate::utils::sse_stream::SseStreamExt;

        // Send the request and get the response
        let response = request_builder
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to send request: {e}")))?;

        // Check if the response is successful
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpError(format!(
                "HTTP error {}: {}",
                status.as_u16(),
                error_text
            )));
        }

        // Convert response to byte stream
        let byte_stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        // Use eventsource-stream to parse SSE
        let sse_stream = byte_stream.into_sse_stream();

        // Convert SSE events to ChatStreamEvents - now supports multiple events per conversion
        let chat_stream = sse_stream
            .then(move |event_result| {
                let converter = converter.clone();
                async move {
                    match event_result {
                        Ok(event) => {
                            // Handle special [DONE] event
                            if event.data.trim() == "[DONE]" {
                                if let Some(end_event) = converter.handle_stream_end() {
                                    return vec![end_event];
                                } else {
                                    return vec![];
                                }
                            }

                            // Skip empty events
                            if event.data.trim().is_empty() {
                                return vec![];
                            }

                            // Convert using provider-specific logic - now returns multiple events
                            converter.convert_event(event).await
                        }
                        Err(e) => {
                            let error =
                                Err(LlmError::StreamError(format!("SSE parsing error: {e}")));
                            vec![error]
                        }
                    }
                }
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(chat_stream))
    }
}

/// Helper utilities for efficient event building
pub struct EventBuilder {
    events: Vec<ChatStreamEvent>,
}

impl EventBuilder {
    /// Create a new event builder
    pub fn new() -> Self {
        Self {
            events: Vec::with_capacity(2), // Most conversions produce 1-2 events
        }
    }

    /// Create a new event builder with specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
        }
    }

    /// Add a StreamStart event
    pub fn add_stream_start(mut self, metadata: crate::types::ResponseMetadata) -> Self {
        self.events.push(ChatStreamEvent::StreamStart { metadata });
        self
    }

    /// Add a ContentDelta event (only if delta is not empty)
    pub fn add_content_delta(mut self, delta: String, index: Option<usize>) -> Self {
        if !delta.is_empty() {
            self.events
                .push(ChatStreamEvent::ContentDelta { delta, index });
        }
        self
    }

    /// Add a ToolCallDelta event
    pub fn add_tool_call_delta(
        mut self,
        id: String,
        function_name: Option<String>,
        arguments_delta: Option<String>,
        index: Option<usize>,
    ) -> Self {
        self.events.push(ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            index,
        });
        self
    }

    /// Add a ThinkingDelta event (only if delta is not empty)
    pub fn add_thinking_delta(mut self, delta: String) -> Self {
        if !delta.is_empty() {
            self.events.push(ChatStreamEvent::ThinkingDelta { delta });
        }
        self
    }

    /// Add a UsageUpdate event
    pub fn add_usage_update(mut self, usage: crate::types::Usage) -> Self {
        self.events.push(ChatStreamEvent::UsageUpdate { usage });
        self
    }

    /// Add a StreamEnd event
    pub fn add_stream_end(mut self, response: crate::types::ChatResponse) -> Self {
        self.events.push(ChatStreamEvent::StreamEnd { response });
        self
    }

    /// Build the events vector
    pub fn build(self) -> Vec<ChatStreamEvent> {
        self.events
    }

    /// Build the events vector wrapped in Results
    pub fn build_results(self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        self.events.into_iter().map(Ok).collect()
    }
}

impl Default for EventBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Note: legacy helper macros for single-event conversion were removed to avoid
// signature drift. Converters should implement the traits directly and emit
// zero or more events per provider chunk using the multi-event signatures.
