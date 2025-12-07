#![allow(clippy::large_enum_variant)]
//! Streaming event types for real-time responses

use super::chat::ChatResponse;
use crate::error::LlmError;
use crate::types::{ResponseMetadata, Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chat streaming event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatStreamEvent {
    /// Content delta (incremental text)
    ContentDelta {
        /// The incremental text content
        delta: String,
        /// Index of the choice (for multiple responses)
        index: Option<usize>,
    },
    /// Tool call delta
    ToolCallDelta {
        /// Tool call ID
        id: String,
        /// Function name (if this is the start of a tool call)
        function_name: Option<String>,
        /// Incremental arguments
        arguments_delta: Option<String>,
        /// Index of the choice
        index: Option<usize>,
    },
    /// Thinking/reasoning content delta (for models that support internal reasoning)
    /// This includes content from `<think>` tags, reasoning fields, and thinking modes
    ThinkingDelta {
        /// The incremental thinking/reasoning content
        delta: String,
    },
    /// Usage statistics update
    UsageUpdate {
        /// Token usage information
        usage: Usage,
    },
    /// Stream start event with metadata
    StreamStart {
        /// Response metadata
        metadata: ResponseMetadata,
    },
    /// Stream end event with final response
    StreamEnd {
        /// Final response
        response: ChatResponse,
    },
    /// Error occurred during streaming
    Error {
        /// Error message
        error: String,
    },
    /// Custom provider-specific event
    ///
    /// Allows providers to emit custom events without modifying the core enum.
    /// Users can pattern match on `event_type` to handle provider-specific features.
    ///
    /// # Example
    /// ```rust,ignore
    /// match event {
    ///     ChatStreamEvent::Custom { event_type, data } => {
    ///         match event_type.as_str() {
    ///             "openai:citation" => { /* Handle OpenAI citation */ }
    ///             "anthropic:thinking_progress" => { /* Handle thinking progress */ }
    ///             _ => { /* Ignore unknown custom events */ }
    ///         }
    ///     }
    ///     _ => { /* Handle standard events */ }
    /// }
    /// ```
    Custom {
        /// Event type identifier (e.g., "openai:function_call_progress", "anthropic:citation")
        event_type: String,
        /// Event data as JSON value
        data: serde_json::Value,
    },
}

/// Audio streaming event
#[derive(Debug, Clone)]
pub enum AudioStreamEvent {
    /// Audio data chunk
    AudioDelta {
        /// Audio data bytes
        data: Vec<u8>,
        /// Audio format
        format: String,
    },
    /// Metadata about the audio
    Metadata {
        /// Sample rate
        sample_rate: Option<u32>,
        /// Duration estimate
        duration: Option<f32>,
        /// Additional metadata
        metadata: HashMap<String, serde_json::Value>,
    },
    /// Stream finished
    Done {
        /// Total duration
        duration: Option<f32>,
        /// Final metadata
        metadata: HashMap<String, serde_json::Value>,
    },
    /// Error occurred during streaming
    Error {
        /// Error message
        error: String,
    },
}

// Stream types
use futures::Stream;
use std::pin::Pin;

/// Audio stream for streaming TTS
pub type AudioStream =
    Pin<Box<dyn Stream<Item = Result<AudioStreamEvent, LlmError>> + Send + Sync>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Test that stream types are Send + Sync for multi-threading
    #[test]
    fn test_stream_types_are_send_sync() {
        // Test that stream types can be used in Arc (requires Send + Sync)
        fn test_arc_usage() {
            let _: Option<Arc<AudioStream>> = None;
        }

        test_arc_usage();
    }
}
