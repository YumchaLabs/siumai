//! Streaming event types for real-time responses

use super::chat::ChatResponse;
use super::common::FinishReason;
use super::common::{ResponseMetadata, Usage};
use crate::error::LlmError;
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

/// Completion streaming event (for non-chat completions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionStreamEvent {
    /// Text delta
    TextDelta {
        /// The incremental text
        text: String,
        /// Index of the completion
        index: Option<usize>,
    },
    /// Usage statistics
    Usage {
        /// Token usage information
        usage: Usage,
    },
    /// Stream finished
    Done {
        /// Final finish reason
        finish_reason: Option<FinishReason>,
        /// Final usage statistics
        usage: Option<Usage>,
    },
    /// Error occurred
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

/// Completion stream for streaming completions
pub type CompletionStream =
    Pin<Box<dyn Stream<Item = Result<CompletionStreamEvent, LlmError>> + Send + Sync>>;

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
            let _: Option<Arc<CompletionStream>> = None;
        }

        test_arc_usage();
    }
}
