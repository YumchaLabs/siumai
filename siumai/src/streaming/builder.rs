//! Event Builder
//!
//! Helper utilities for efficiently building ChatStreamEvent sequences.

use crate::error::LlmError;
use crate::types::{ChatResponse, ChatStreamEvent, ResponseMetadata, Usage};

/// Event Builder
///
/// Helper for efficiently building sequences of ChatStreamEvents.
/// Commonly used in provider-specific converters to emit multiple events
/// from a single provider event (e.g., StreamStart + ContentDelta).
///
/// # Example
/// ```rust,ignore
/// use siumai::streaming::EventBuilder;
///
/// let events = EventBuilder::new()
///     .add_stream_start(metadata)
///     .add_content_delta("Hello".to_string(), None)
///     .build();
/// ```
pub struct EventBuilder {
    events: Vec<ChatStreamEvent>,
}

impl Default for EventBuilder {
    fn default() -> Self {
        Self::new()
    }
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
    pub fn add_stream_start(mut self, metadata: ResponseMetadata) -> Self {
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
    pub fn add_usage_update(mut self, usage: Usage) -> Self {
        self.events.push(ChatStreamEvent::UsageUpdate { usage });
        self
    }

    /// Add a StreamEnd event
    pub fn add_stream_end(mut self, response: ChatResponse) -> Self {
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
