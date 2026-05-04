//! Event Builder
//!
//! Helper utilities for efficiently building ChatStreamEvent sequences.

use crate::error::LlmError;
use crate::types::{
    ChatResponse, ChatStreamEvent, ChatStreamPart, ChatStreamReplay, ResponseMetadata,
};

/// Event Builder
///
/// Helper for efficiently building sequences of ChatStreamEvents.
/// Commonly used in provider-specific converters to emit multiple events
/// from a single provider event (e.g., StreamStart + TextDelta part).
///
/// # Example
/// ```rust,ignore
/// use siumai::streaming::EventBuilder;
///
/// let events = EventBuilder::new()
///     .add_stream_start(metadata)
///     .add_text_delta("0", "Hello")
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

    /// Add a typed text delta part (only if delta is not empty).
    pub fn add_text_delta(mut self, id: impl Into<String>, delta: impl Into<String>) -> Self {
        let delta = delta.into();
        if !delta.is_empty() {
            self.events.push(ChatStreamEvent::Part {
                part: ChatStreamPart::TextDelta {
                    id: id.into(),
                    delta,
                    provider_metadata: None,
                },
            });
        }
        self
    }

    /// Add a typed reasoning delta part (only if delta is not empty).
    pub fn add_reasoning_delta(mut self, id: impl Into<String>, delta: impl Into<String>) -> Self {
        let delta = delta.into();
        if !delta.is_empty() {
            self.events.push(ChatStreamEvent::Part {
                part: ChatStreamPart::ReasoningDelta {
                    id: id.into(),
                    delta,
                    provider_metadata: None,
                },
            });
        }
        self
    }

    /// Add a typed tool input start part.
    pub fn add_tool_input_start(
        mut self,
        id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        self.events.push(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputStart {
                id: id.into(),
                tool_name: tool_name.into(),
                provider_metadata: None,
                provider_executed: None,
                dynamic: None,
                title: None,
            },
        });
        self
    }

    /// Add a typed tool input delta part (only if delta is not empty).
    pub fn add_tool_input_delta(mut self, id: impl Into<String>, delta: impl Into<String>) -> Self {
        let delta = delta.into();
        if !delta.is_empty() {
            self.events.push(ChatStreamEvent::Part {
                part: ChatStreamPart::ToolInputDelta {
                    id: id.into(),
                    delta,
                    provider_metadata: None,
                },
            });
        }
        self
    }

    /// Add a typed tool input end part.
    pub fn add_tool_input_end(mut self, id: impl Into<String>) -> Self {
        self.events.push(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputEnd {
                id: id.into(),
                provider_metadata: None,
            },
        });
        self
    }

    /// Add a StreamEnd event
    pub fn add_stream_end(mut self, response: ChatResponse) -> Self {
        self.events.push(ChatStreamEvent::StreamEnd { response });
        self
    }

    /// Add a Custom event
    pub fn add_custom_event(mut self, event_type: String, data: serde_json::Value) -> Self {
        self.events
            .push(ChatStreamEvent::Custom { event_type, data });
        self
    }

    /// Add a typed stream part event.
    pub fn add_part(mut self, part: ChatStreamPart) -> Self {
        self.events.push(ChatStreamEvent::Part { part });
        self
    }

    /// Add a typed stream part event with runtime replay hints.
    pub fn add_part_with_replay(mut self, part: ChatStreamPart, replay: ChatStreamReplay) -> Self {
        self.events
            .push(ChatStreamEvent::PartWithReplay { part, replay });
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
