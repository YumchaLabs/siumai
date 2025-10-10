//! Streaming Processing Module
//!
//! Defines types and functionality for handling streaming responses from LLM providers

use futures::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;

use crate::error::LlmError;
use crate::types::*;

/// Chat Stream - Main interface for streaming responses
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + Sync>>;

// Re-export ChatStreamEvent from types module to avoid duplication
pub use crate::types::ChatStreamEvent;

/// Tool Call Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Call ID
    pub id: Option<String>,
    /// Tool type
    pub r#type: Option<ToolType>,
    /// Function call delta
    pub function: Option<FunctionCallDelta>,
}

/// Function Call Delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name delta
    pub name: Option<String>,
    /// Arguments delta
    pub arguments: Option<String>,
}

/// Stream Processor configuration
pub type OverflowHandler = Box<dyn Fn(&str, usize) + Send + Sync>;
pub struct StreamProcessorConfig {
    /// Maximum size for content buffer (in bytes)
    pub max_content_buffer_size: Option<usize>,
    /// Maximum size for thinking buffer (in bytes)  
    pub max_thinking_buffer_size: Option<usize>,
    /// Maximum number of tool calls to track
    pub max_tool_calls: Option<usize>,
    /// Handler for buffer overflow
    pub overflow_handler: Option<OverflowHandler>,
}

impl std::fmt::Debug for StreamProcessorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamProcessorConfig")
            .field("max_content_buffer_size", &self.max_content_buffer_size)
            .field("max_thinking_buffer_size", &self.max_thinking_buffer_size)
            .field("max_tool_calls", &self.max_tool_calls)
            .field(
                "has_overflow_handler",
                &self
                    .overflow_handler
                    .as_ref()
                    .map(|_| true)
                    .unwrap_or(false),
            )
            .finish()
    }
}

impl Default for StreamProcessorConfig {
    fn default() -> Self {
        Self {
            max_content_buffer_size: Some(10 * 1024 * 1024), // 10MB default
            max_thinking_buffer_size: Some(5 * 1024 * 1024), // 5MB default
            max_tool_calls: Some(100),                       // 100 tool calls max
            overflow_handler: None,
        }
    }
}

/// Stream Processor - for processing and transforming stream events
pub struct StreamProcessor {
    buffer: String,
    tool_calls: std::collections::HashMap<String, ToolCallBuilder>, // Use ID as key to handle duplicate indices
    tool_call_order: Vec<String>, // Track order of tool calls for consistent output
    thinking_buffer: String,
    current_usage: Option<Usage>,
    config: StreamProcessorConfig,
}

impl StreamProcessor {
    pub fn new() -> Self {
        Self::with_config(StreamProcessorConfig::default())
    }

    /// Create a new stream processor with custom configuration
    pub fn with_config(config: StreamProcessorConfig) -> Self {
        Self {
            buffer: String::new(),
            tool_calls: std::collections::HashMap::new(),
            tool_call_order: Vec::new(),
            thinking_buffer: String::new(),
            current_usage: None,
            config,
        }
    }

    /// Process a stream event
    pub fn process_event(&mut self, event: ChatStreamEvent) -> ProcessedEvent {
        match event {
            ChatStreamEvent::ContentDelta { delta, index } => {
                // Check buffer size limit before appending
                if let Some(max_size) = self.config.max_content_buffer_size {
                    let new_size = self.buffer.len() + delta.len();
                    if new_size > max_size {
                        // Call overflow handler if provided
                        if let Some(handler) = &self.config.overflow_handler {
                            (handler)("content_buffer", new_size);
                        }
                        // Truncate buffer to keep within limits
                        let available = max_size.saturating_sub(self.buffer.len());
                        let truncated_delta = if available > 0 {
                            delta.chars().take(available).collect()
                        } else {
                            String::new()
                        };
                        self.buffer.push_str(&truncated_delta);
                        return ProcessedEvent::ContentUpdate {
                            delta: truncated_delta,
                            accumulated: self.buffer.clone(),
                            index,
                        };
                    }
                }

                self.buffer.push_str(&delta);
                ProcessedEvent::ContentUpdate {
                    delta,
                    accumulated: self.buffer.clone(),
                    index,
                }
            }
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            } => {
                tracing::debug!("Tool call delta - ID: '{}', Index: {:?}", id, index);

                // Use tool call ID as the primary key to handle duplicate indices
                // This solves the problem where OpenAI returns multiple tool calls with the same index
                let tool_id = if !id.is_empty() {
                    id.clone()
                } else {
                    // If no ID, we need to find the most recent tool call that doesn't have an ID yet
                    // This handles the case where subsequent deltas don't include the ID
                    if let Some(last_id) = self.tool_call_order.last() {
                        last_id.clone()
                    } else {
                        // Fallback: create a temporary ID based on order
                        format!("temp_tool_call_{}", self.tool_call_order.len())
                    }
                };

                // Get or create the tool call builder
                let is_new_tool_call = !self.tool_calls.contains_key(&tool_id);

                // Check tool call limit
                if let Some(max_tool_calls) = self.config.max_tool_calls
                    && is_new_tool_call
                    && self.tool_calls.len() >= max_tool_calls
                {
                    // Too many tool calls, skip this one
                    if let Some(handler) = &self.config.overflow_handler {
                        (handler)("tool_calls", self.tool_calls.len() + 1);
                    }
                    return ProcessedEvent::ToolCallUpdate {
                        id: tool_id,
                        current_state: ToolCallBuilder::new(),
                        index,
                    };
                }

                let builder = self.tool_calls.entry(tool_id.clone()).or_insert_with(|| {
                    let mut builder = ToolCallBuilder::new();
                    if !id.is_empty() {
                        builder.id = id.clone();
                    } else {
                        builder.id = tool_id.clone();
                    }
                    builder
                });

                // Track order of tool calls for consistent output
                if is_new_tool_call && !id.is_empty() {
                    self.tool_call_order.push(tool_id.clone());
                }

                // Accumulate function name
                if let Some(name) = function_name {
                    if builder.name.is_empty() {
                        builder.name = name;
                    } else {
                        builder.name.push_str(&name);
                    }
                }

                // Accumulate arguments
                if let Some(args) = arguments_delta {
                    builder.arguments.push_str(&args);
                }

                ProcessedEvent::ToolCallUpdate {
                    id: builder.id.clone(),
                    current_state: builder.clone(),
                    index,
                }
            }
            ChatStreamEvent::ThinkingDelta { delta } => {
                // Check thinking buffer size limit
                if let Some(max_size) = self.config.max_thinking_buffer_size {
                    let new_size = self.thinking_buffer.len() + delta.len();
                    if new_size > max_size {
                        // Call overflow handler if provided
                        if let Some(handler) = &self.config.overflow_handler {
                            (handler)("thinking_buffer", new_size);
                        }
                        // Truncate buffer to keep within limits
                        let available = max_size.saturating_sub(self.thinking_buffer.len());
                        let truncated_delta = if available > 0 {
                            delta.chars().take(available).collect()
                        } else {
                            String::new()
                        };
                        self.thinking_buffer.push_str(&truncated_delta);
                        return ProcessedEvent::ThinkingUpdate {
                            delta: truncated_delta,
                            accumulated: self.thinking_buffer.clone(),
                        };
                    }
                }

                self.thinking_buffer.push_str(&delta);
                ProcessedEvent::ThinkingUpdate {
                    delta,
                    accumulated: self.thinking_buffer.clone(),
                }
            }
            ChatStreamEvent::UsageUpdate { usage } => {
                if let Some(ref mut current) = self.current_usage {
                    current.merge(&usage);
                } else {
                    self.current_usage = Some(usage.clone());
                }
                ProcessedEvent::UsageUpdate {
                    usage: self.current_usage.clone().unwrap(),
                }
            }
            ChatStreamEvent::StreamStart { metadata } => ProcessedEvent::StreamStart { metadata },
            ChatStreamEvent::StreamEnd { response } => ProcessedEvent::StreamEnd { response },

            ChatStreamEvent::Error { error } => ProcessedEvent::Error {
                error: LlmError::InternalError(error),
            },
        }
    }

    /// Build the final response
    pub fn build_final_response(&self) -> ChatResponse {
        self.build_final_response_with_finish_reason(None)
    }

    /// Build the final response with finish reason
    pub fn build_final_response_with_finish_reason(
        &self,
        finish_reason: Option<FinishReason>,
    ) -> ChatResponse {
        let mut metadata = HashMap::new();

        if !self.thinking_buffer.is_empty() {
            metadata.insert(
                "thinking".to_string(),
                serde_json::Value::String(self.thinking_buffer.clone()),
            );
        }

        let tool_calls = if !self.tool_calls.is_empty() {
            Some(
                self.tool_call_order
                    .iter()
                    .filter_map(|id| self.tool_calls.get(id))
                    .filter(|builder| !builder.name.is_empty()) // Only include tool calls with names
                    .map(|builder| builder.build())
                    .collect(),
            )
        } else {
            None
        };

        let thinking = if !self.thinking_buffer.is_empty() {
            Some(self.thinking_buffer.clone())
        } else {
            None
        };

        ChatResponse {
            id: None,
            content: MessageContent::Text(self.buffer.clone()),
            model: None,
            usage: self.current_usage.clone(),
            finish_reason,
            tool_calls,
            thinking,
            metadata,
        }
    }
}

/// Processed Event
#[derive(Debug, Clone)]
pub enum ProcessedEvent {
    ContentUpdate {
        delta: String,
        accumulated: String,
        index: Option<usize>,
    },
    ToolCallUpdate {
        id: String,
        current_state: ToolCallBuilder,
        index: Option<usize>,
    },
    ThinkingUpdate {
        delta: String,
        accumulated: String,
    },
    UsageUpdate {
        usage: Usage,
    },
    StreamStart {
        metadata: ResponseMetadata,
    },
    StreamEnd {
        response: ChatResponse,
    },
    Error {
        error: LlmError,
    },
}

/// Tool Call Builder
#[derive(Debug, Clone)]
pub struct ToolCallBuilder {
    pub id: String,
    pub r#type: Option<ToolType>,
    pub name: String,
    pub arguments: String,
}

impl Default for ToolCallBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallBuilder {
    pub const fn new() -> Self {
        Self {
            id: String::new(),
            r#type: None,
            name: String::new(),
            arguments: String::new(),
        }
    }

    pub fn build(&self) -> ToolCall {
        ToolCall {
            id: self.id.clone(),
            r#type: self
                .r#type
                .as_ref()
                .map(|t| format!("{t:?}"))
                .unwrap_or_else(|| "function".to_string()),
            function: Some(FunctionCall {
                name: self.name.clone(),
                arguments: self.arguments.clone(),
            }),
        }
    }
}

impl Default for StreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Stream Utilities
///
/// Utility functions for working with chat streams
/// Collect all stream events into a single response
///
/// This function consumes the entire stream and builds a final `ChatResponse`
pub async fn collect_stream_response(mut stream: ChatStream) -> Result<ChatResponse, LlmError> {
    use futures::StreamExt;

    let mut processor = StreamProcessor::new();
    let mut _metadata = None;

    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::StreamStart { metadata: meta } => {
                _metadata = Some(meta);
            }
            ChatStreamEvent::StreamEnd { response } => {
                return Ok(response);
            }
            ChatStreamEvent::Error { error } => {
                return Err(LlmError::InternalError(error));
            }
            event => {
                processor.process_event(event);
            }
        }
    }

    Ok(processor.build_final_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_processor() {
        let mut processor = StreamProcessor::new();

        let event = ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        };

        let processed = processor.process_event(event);

        match processed {
            ProcessedEvent::ContentUpdate {
                delta, accumulated, ..
            } => {
                assert_eq!(delta, "Hello");
                assert_eq!(accumulated, "Hello");
            }
            _ => panic!("Expected ContentUpdate"),
        }
    }

    // Test that stream types are Send + Sync for multi-threading
    #[test]
    fn test_stream_types_are_send_sync() {
        use std::sync::Arc;

        // Test that stream types can be used in Arc (requires Send + Sync)
        fn test_arc_usage() {
            let _: Option<Arc<ChatStream>> = None;
        }

        test_arc_usage();
    }

    // Test actual multi-threading with stream types
    #[tokio::test]
    async fn test_stream_multithreading() {
        use futures::stream;
        use std::sync::Arc;
        use tokio::task;

        // Create a mock stream that we can share across threads
        let mock_events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: " World".to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: crate::types::ChatResponse {
                    id: Some("test-id".to_string()),
                    content: crate::types::MessageContent::Text("Hello World".to_string()),
                    model: Some("test-model".to_string()),
                    usage: None,
                    finish_reason: Some(crate::types::FinishReason::Stop),
                    tool_calls: None,
                    thinking: None,
                    metadata: std::collections::HashMap::new(),
                },
            }),
        ];

        let stream: ChatStream = Box::pin(stream::iter(mock_events));
        let stream_arc = Arc::new(tokio::sync::Mutex::new(stream));

        // Spawn multiple tasks that could potentially access the stream
        // (In practice, streams are usually consumed by one task, but this tests Send + Sync)
        let mut handles = Vec::new();

        for i in 0..3 {
            let stream_clone = stream_arc.clone();
            let handle = task::spawn(async move {
                // This tests that the stream can be moved across thread boundaries
                let _guard = stream_clone.lock().await;
                // In a real scenario, we'd consume the stream here
                i // Return task id for verification
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }

        // Verify all tasks completed
        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, i);
        }
    }
}
