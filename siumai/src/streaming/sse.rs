//! Server-Sent Events (SSE) Stream Utilities
//!
//! This module provides utilities for handling SSE streams with proper UTF-8 handling
//! and line buffering using the eventsource-stream crate.
//!
//! This is the unified SSE processing solution for all providers in siumai.

use eventsource_stream::{Event, Eventsource};
use futures_util::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

// Re-export Event for convenience
pub use eventsource_stream::Event as SseEvent;

/// A wrapper around eventsource-stream that provides additional functionality
/// for LLM streaming responses
pub struct SseStream<S> {
    inner: eventsource_stream::EventStream<S>,
}

impl<S> SseStream<S> {
    /// Create a new SSE stream from a byte stream
    pub fn new(stream: S) -> Self
    where
        S: Stream + Eventsource,
    {
        Self {
            inner: stream.eventsource(),
        }
    }

    /// Set the last event ID for reconnection purposes
    pub fn set_last_event_id(&mut self, id: impl Into<String>) {
        self.inner.set_last_event_id(id);
    }

    /// Get the last event ID
    pub fn last_event_id(&self) -> &str {
        self.inner.last_event_id()
    }
}

impl<S, B, E> Stream for SseStream<S>
where
    S: Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
{
    type Item = Result<Event, eventsource_stream::EventStreamError<E>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// Extension trait to easily convert byte streams to SSE streams
pub trait SseStreamExt: Sized {
    /// Convert this byte stream into an SSE stream
    fn into_sse_stream(self) -> SseStream<Self>;
}

impl<S, B, E> SseStreamExt for S
where
    S: Stream<Item = Result<B, E>>,
    B: AsRef<[u8]>,
{
    fn into_sse_stream(self) -> SseStream<Self> {
        SseStream::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn test_sse_stream_basic() {
        let data: Vec<Result<&[u8], std::io::Error>> = vec![
            Ok(b"data: hello\n\n".as_slice()),
            Ok(b"data: world\n\n".as_slice()),
        ];

        let stream = futures_util::stream::iter(data);
        let mut sse_stream = stream.into_sse_stream();

        let event1 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "hello");

        let event2 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "world");
    }

    #[tokio::test]
    async fn test_sse_stream_utf8_handling() {
        // Test UTF-8 characters in SSE data
        let chinese_text = "你好世界"; // "Hello World" in Chinese

        let sse_data = format!("data: {chinese_text}\n\n");
        let full_data: Vec<Result<&[u8], std::io::Error>> = vec![Ok(sse_data.as_bytes())];

        let stream = futures_util::stream::iter(full_data);
        let mut sse_stream = stream.into_sse_stream();

        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.data, chinese_text);
    }
}
