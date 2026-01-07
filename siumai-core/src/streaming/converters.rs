//! Stream Event Converters
//!
//! Traits for converting provider-specific streaming events to unified ChatStreamEvent.

use crate::error::LlmError;
use crate::types::ChatStreamEvent;
use eventsource_stream::Event;
use std::future::Future;
use std::pin::Pin;

/// Type alias for SSE event conversion future - supports multiple events
pub type SseEventFuture<'a> =
    Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>;

/// Type alias for JSON event conversion future - supports multiple events
pub type JsonEventFuture<'a> =
    Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>;

/// Trait for converting provider-specific SSE events to ChatStreamEvent
///
/// This trait supports multi-event emission, allowing a single provider event
/// to generate multiple ChatStreamEvents (e.g., StreamStart + ContentDelta).
///
/// # Example
/// ```rust,ignore
/// struct MyConverter;
///
/// impl SseEventConverter for MyConverter {
///     fn convert_event(&self, event: Event) -> SseEventFuture<'_> {
///         Box::pin(async move {
///             // Parse event.data and return ChatStreamEvents
///             vec![Ok(ChatStreamEvent::ContentDelta {
///                 delta: event.data,
///                 index: None,
///             })]
///         })
///     }
/// }
/// ```
pub trait SseEventConverter: Send + Sync {
    /// Convert an SSE event to zero or more ChatStreamEvents
    fn convert_event(&self, event: Event) -> SseEventFuture<'_>;

    /// Whether the StreamFactory should call `handle_stream_end` when the SSE
    /// connection closes without an explicit `[DONE]` marker.
    ///
    /// Default is `false` to preserve the existing semantics: an unexpected
    /// disconnect should not synthesize a StreamEnd.
    fn finalize_on_disconnect(&self) -> bool {
        false
    }

    /// Handle the end of stream and emit zero or more final events.
    ///
    /// By default this wraps `handle_stream_end()` (0 or 1 event). Converters
    /// that need multiple end events (e.g., a custom `finish` event followed by
    /// `StreamEnd`) should override this method.
    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        self.handle_stream_end().into_iter().collect()
    }

    /// Handle the end of stream
    ///
    /// Called when the stream ends (e.g., [DONE] event).
    /// Return Some(event) to emit a final event, or None to end silently.
    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        None
    }
}

/// Trait for converting JSON data to ChatStreamEvent (for providers like Gemini, Ollama)
///
/// This trait supports multi-event emission for JSON-based streaming.
/// Used for providers that emit line-delimited JSON instead of SSE.
///
/// # Example
/// ```rust,ignore
/// struct MyJsonConverter;
///
/// impl JsonEventConverter for MyJsonConverter {
///     fn convert_json<'a>(&'a self, json_data: &'a str) -> JsonEventFuture<'a> {
///         Box::pin(async move {
///             // Parse JSON and return ChatStreamEvents
///             vec![Ok(ChatStreamEvent::ContentDelta {
///                 delta: json_data.to_string(),
///                 index: None,
///             })]
///         })
///     }
/// }
/// ```
pub trait JsonEventConverter: Send + Sync {
    /// Convert JSON data to zero or more ChatStreamEvents
    fn convert_json<'a>(&'a self, json_data: &'a str) -> JsonEventFuture<'a>;

    /// Handle the end of stream
    ///
    /// Called when the JSON stream ends (e.g., connection closed).
    /// Return Some(event) to emit a final event, or None to end silently.
    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        None
    }
}
