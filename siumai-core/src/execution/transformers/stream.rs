//! Stream chunk transformation traits
//!
//! Converts provider-specific SSE/streaming chunks into unified ChatStreamEvent sequences.
//! This is similar to Cherry Studio's ResponseChunkTransformer.

use crate::error::LlmError;
use crate::streaming::ChatStreamEvent;
use eventsource_stream::Event;
use std::future::Future;
use std::pin::Pin;

/// Type alias for stream event conversion futures to keep signatures readable
pub type StreamEventFuture<'a> =
    Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>;

/// Convert provider SSE events to ChatStreamEvents
pub trait StreamChunkTransformer: Send + Sync {
    /// Provider identifier
    fn provider_id(&self) -> &str;

    /// Convert a single SSE event into zero or more ChatStreamEvents
    fn convert_event(&self, _event: Event) -> StreamEventFuture<'_>;

    /// Optional end-of-stream event (e.g., [DONE] for OpenAI-compatible)
    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        None
    }

    /// Handle the end of stream and emit zero or more final events.
    ///
    /// By default this wraps `handle_stream_end()` (0 or 1 event). Transformers
    /// that need multiple end events should override this method.
    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        self.handle_stream_end().into_iter().collect()
    }

    /// Whether the StreamFactory should call `handle_stream_end` when the SSE
    /// connection closes without an explicit `[DONE]` marker.
    ///
    /// Default is `false` to preserve the existing semantics: an unexpected
    /// disconnect should not synthesize a StreamEnd.
    fn finalize_on_disconnect(&self) -> bool {
        false
    }
}
