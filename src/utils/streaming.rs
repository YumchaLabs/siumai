//! Common Streaming Utilities
//!
//! This module provides common utilities for handling streaming responses
//! across different providers, including line buffering, UTF-8 handling,
//! and unified SSE processing using eventsource-stream.

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::utils::sse_stream::SseStreamExt;
use eventsource_stream::Event;
use futures_util::StreamExt;
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
    /// Create a chat stream for JSON-based streaming (like Gemini)
    ///
    /// Some providers use JSON streaming instead of SSE. This method handles
    /// JSON object parsing across chunk boundaries using UTF-8 safe processing.
    pub async fn create_json_stream<C>(
        response: reqwest::Response,
        converter: C,
    ) -> Result<ChatStream, LlmError>
    where
        C: JsonEventConverter + Clone + 'static,
    {
        let byte_stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));

        // Use eventsource-stream for UTF-8 handling, then parse as JSON
        let sse_stream = byte_stream.into_sse_stream();

        let chat_stream = sse_stream
            .then(move |event_result| {
                let converter = converter.clone();
                async move {
                    match event_result {
                        Ok(event) => {
                            // For JSON streaming, we treat the data as raw JSON
                            if event.data.trim().is_empty() {
                                return vec![];
                            }

                            converter.convert_json(&event.data).await
                        }
                        Err(e) => {
                            let error =
                                Err(LlmError::ParseError(format!("JSON parsing error: {e}")));
                            vec![error]
                        }
                    }
                }
            })
            .flat_map(futures::stream::iter);

        // Explicitly type the boxed stream to help the compiler
        let boxed_stream: ChatStream = Box::pin(chat_stream);
        Ok(boxed_stream)
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

/// Helper macro to create SSE event converters
#[macro_export]
macro_rules! impl_sse_converter {
    ($converter_type:ty, $event_type:ty, $convert_fn:ident) => {
        impl $crate::utils::streaming::SseEventConverter for $converter_type {
            fn convert_event(
                &self,
                event: eventsource_stream::Event,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Option<
                                Result<$crate::stream::ChatStreamEvent, $crate::error::LlmError>,
                            >,
                        > + Send
                        + Sync
                        + '_,
                >,
            > {
                Box::pin(async move {
                    match serde_json::from_str::<$event_type>(&event.data) {
                        Ok(parsed_event) => Some(Ok(self.$convert_fn(parsed_event))),
                        Err(e) => Some(Err($crate::error::LlmError::ParseError(format!(
                            "Failed to parse event: {e}"
                        )))),
                    }
                })
            }
        }
    };
}

/// Helper macro to create JSON event converters
#[macro_export]
macro_rules! impl_json_converter {
    ($converter_type:ty, $event_type:ty, $convert_fn:ident) => {
        impl $crate::utils::streaming::JsonEventConverter for $converter_type {
            fn convert_json<'a>(
                &'a self,
                json_data: &'a str,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Option<
                                Result<$crate::stream::ChatStreamEvent, $crate::error::LlmError>,
                            >,
                        > + Send
                        + Sync
                        + 'a,
                >,
            > {
                Box::pin(async move {
                    match serde_json::from_str::<$event_type>(json_data) {
                        Ok(parsed_event) => Some(Ok(self.$convert_fn(parsed_event))),
                        Err(e) => Some(Err($crate::error::LlmError::ParseError(format!(
                            "Failed to parse JSON: {e}"
                        )))),
                    }
                })
            }
        }
    };
}
