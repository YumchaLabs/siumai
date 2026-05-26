use super::OpenAiResponsesEventConverter;
use crate::streaming::ChatStreamEvent;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Buffers terminal stream events until the Responses SSE connection is actually terminal.
///
/// OpenAI Responses can emit multiple `response.created` / `response.completed` pairs in a single
/// SSE connection when provider-hosted tools run. The converter should only release the latest
/// terminal `Finish` / `StreamEnd` sequence once the transport reports the end of the stream.
#[derive(Debug, Clone, Default)]
pub(super) struct TerminalEventBuffer {
    events: Arc<Mutex<VecDeque<ChatStreamEvent>>>,
}

impl TerminalEventBuffer {
    pub(super) fn clear(&self) {
        if let Ok(mut q) = self.events.lock() {
            q.clear();
        }
    }

    pub(super) fn replace<I>(&self, events: I)
    where
        I: IntoIterator<Item = ChatStreamEvent>,
    {
        if let Ok(mut q) = self.events.lock() {
            q.clear();
            q.extend(events);
        }
    }

    pub(super) fn pop(&self) -> Option<ChatStreamEvent> {
        self.events.lock().ok().and_then(|mut q| q.pop_front())
    }

    pub(super) fn drain(&self) -> Vec<ChatStreamEvent> {
        let Ok(mut q) = self.events.lock() else {
            return Vec::new();
        };
        q.drain(..).collect()
    }
}

impl OpenAiResponsesEventConverter {
    pub(super) fn clear_pending_stream_end_events(&self) {
        self.pending_stream_end_events.clear();
    }

    pub(super) fn replace_pending_stream_end_events(&self, events: Vec<ChatStreamEvent>) {
        self.pending_stream_end_events.replace(events);
    }

    pub(super) fn pop_pending_stream_end_event(&self) -> Option<ChatStreamEvent> {
        self.pending_stream_end_events.pop()
    }

    pub(super) fn drain_pending_stream_end_events(&self) -> Vec<ChatStreamEvent> {
        self.pending_stream_end_events.drain()
    }
}
