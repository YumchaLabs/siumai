//! Stream chunk transformation traits (Phase 0 scaffolding)
//!
//! Converts provider-specific SSE/streaming chunks into unified ChatStreamEvent sequences.
//! This is similar to Cherry Studio's ResponseChunkTransformer.

use crate::error::LlmError;
use crate::stream::ChatStreamEvent;
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
}
