use super::*;

impl OpenAiResponsesEventConverter {
    pub(super) fn clear_pending_stream_end_events(&self) {
        if let Ok(mut q) = self.pending_stream_end_events.lock() {
            q.clear();
        }
    }

    pub(super) fn replace_pending_stream_end_events(
        &self,
        events: Vec<crate::streaming::ChatStreamEvent>,
    ) {
        if let Ok(mut q) = self.pending_stream_end_events.lock() {
            q.clear();
            q.extend(events);
        }
    }

    pub(super) fn pop_pending_stream_end_event(&self) -> Option<crate::streaming::ChatStreamEvent> {
        self.pending_stream_end_events
            .lock()
            .ok()
            .and_then(|mut q| q.pop_front())
    }
}
