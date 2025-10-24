//! Stream Processor
//!
//! Processes and transforms streaming events, accumulating content, tool calls,
//! and thinking buffers with configurable limits and overflow handling.

use crate::error::LlmError;
use crate::types::{
    ChatResponse, ChatStreamEvent, ContentPart, FinishReason, MessageContent, ResponseMetadata,
    Usage,
};
use std::collections::HashMap;

/// Overflow handler callback type
///
/// Called when a buffer exceeds its configured limit.
/// Parameters: (buffer_name, attempted_size)
pub type OverflowHandler = Box<dyn Fn(&str, usize) + Send + Sync>;

/// Stream Processor Configuration
///
/// Controls buffer limits and overflow behavior for stream processing.
#[derive(Default)]
pub struct StreamProcessorConfig {
    /// Maximum size for content buffer (in bytes)
    pub max_content_buffer_size: Option<usize>,
    /// Maximum size for thinking buffer (in bytes)  
    pub max_thinking_buffer_size: Option<usize>,
    /// Maximum number of tool calls to track
    pub max_tool_calls: Option<usize>,
    /// Maximum accumulated size for a single tool call's arguments (in bytes)
    pub max_tool_arguments_size: Option<usize>,
    /// Handler for buffer overflow
    pub overflow_handler: Option<OverflowHandler>,
}

impl std::fmt::Debug for StreamProcessorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamProcessorConfig")
            .field("max_content_buffer_size", &self.max_content_buffer_size)
            .field("max_thinking_buffer_size", &self.max_thinking_buffer_size)
            .field("max_tool_calls", &self.max_tool_calls)
            .field("max_tool_arguments_size", &self.max_tool_arguments_size)
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

impl StreamProcessorConfig {
    /// Create default configuration with reasonable limits
    pub fn default() -> Self {
        Self {
            max_content_buffer_size: Some(10 * 1024 * 1024), // 10MB default
            max_thinking_buffer_size: Some(5 * 1024 * 1024), // 5MB default
            max_tool_calls: Some(100),                       // 100 tool calls max
            max_tool_arguments_size: None,                   // default: no truncation for args
            overflow_handler: None,
        }
    }
}

/// Stream Processor
///
/// Processes streaming events and accumulates content, tool calls, and thinking buffers.
/// Provides buffer overflow protection and incremental state tracking.
pub struct StreamProcessor {
    buffer: String,
    tool_calls: HashMap<String, ToolCallBuilder>, // Use ID as key to handle duplicate indices
    tool_call_order: Vec<String>,                 // Track order of tool calls for consistent output
    thinking_buffer: String,
    current_usage: Option<Usage>,
    config: StreamProcessorConfig,
}

impl Default for StreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamProcessor {
    /// Create a new stream processor with default configuration
    pub fn new() -> Self {
        Self::with_config(StreamProcessorConfig::default())
    }

    /// Create a new stream processor with custom configuration
    pub fn with_config(config: StreamProcessorConfig) -> Self {
        Self {
            buffer: String::new(),
            tool_calls: HashMap::new(),
            tool_call_order: Vec::new(),
            thinking_buffer: String::new(),
            current_usage: None,
            config,
        }
    }

    /// Process a stream event and return the processed result
    pub fn process_event(&mut self, event: ChatStreamEvent) -> ProcessedEvent {
        match event {
            ChatStreamEvent::ContentDelta { delta, index } => {
                self.process_content_delta(delta, index)
            }
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            } => self.process_tool_call_delta(id, function_name, arguments_delta, index),
            ChatStreamEvent::ThinkingDelta { delta } => self.process_thinking_delta(delta),
            ChatStreamEvent::UsageUpdate { usage } => self.process_usage_update(usage),
            ChatStreamEvent::StreamStart { metadata } => ProcessedEvent::StreamStart { metadata },
            ChatStreamEvent::StreamEnd { response } => ProcessedEvent::StreamEnd { response },
            ChatStreamEvent::Error { error } => ProcessedEvent::Error {
                error: LlmError::InternalError(error),
            },
            ChatStreamEvent::Custom { event_type, data } => {
                ProcessedEvent::Custom { event_type, data }
            }
        }
    }

    /// Process content delta
    fn process_content_delta(&mut self, delta: String, index: Option<usize>) -> ProcessedEvent {
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

    /// Process tool call delta
    fn process_tool_call_delta(
        &mut self,
        id: String,
        function_name: Option<String>,
        arguments_delta: Option<String>,
        index: Option<usize>,
    ) -> ProcessedEvent {
        tracing::debug!("Tool call delta - ID: '{}', Index: {:?}", id, index);

        // Use tool call ID as the primary key to handle duplicate indices
        let tool_id = if !id.is_empty() {
            id.clone()
        } else {
            // If no ID, use the most recent tool call
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
        if let Some(max_tool_calls) = self.config.max_tool_calls {
            if is_new_tool_call && self.tool_calls.len() >= max_tool_calls {
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
            if let Some(max_args) = self.config.max_tool_arguments_size {
                let new_size = builder.arguments.len() + args.len();
                if new_size > max_args {
                    if let Some(handler) = &self.config.overflow_handler {
                        (handler)("tool_arguments", new_size);
                    }
                    let available = max_args.saturating_sub(builder.arguments.len());
                    if available > 0 {
                        let truncated: String = args.chars().take(available).collect();
                        builder.arguments.push_str(&truncated);
                    }
                } else {
                    builder.arguments.push_str(&args);
                }
            } else {
                builder.arguments.push_str(&args);
            }
        }

        ProcessedEvent::ToolCallUpdate {
            id: builder.id.clone(),
            current_state: builder.clone(),
            index,
        }
    }

    /// Process thinking delta
    fn process_thinking_delta(&mut self, delta: String) -> ProcessedEvent {
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

    /// Process usage update
    fn process_usage_update(&mut self, usage: Usage) -> ProcessedEvent {
        if let Some(ref mut current) = self.current_usage {
            current.merge(&usage);
        } else {
            self.current_usage = Some(usage.clone());
        }
        ProcessedEvent::UsageUpdate {
            usage: self.current_usage.clone().unwrap(),
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
        let mut stream_metadata = HashMap::new();

        if !self.thinking_buffer.is_empty() {
            stream_metadata.insert(
                "thinking".to_string(),
                serde_json::Value::String(self.thinking_buffer.clone()),
            );
        }

        // Build content with text, tool calls, and reasoning
        let mut parts = Vec::new();

        // Add text content if present
        if !self.buffer.is_empty() {
            parts.push(ContentPart::text(&self.buffer));
        }

        // Add tool calls if present
        if !self.tool_calls.is_empty() {
            for id in &self.tool_call_order {
                if let Some(builder) = self.tool_calls.get(id) {
                    if !builder.name.is_empty() {
                        // Parse arguments string to JSON Value
                        let arguments =
                            serde_json::from_str(&builder.arguments).unwrap_or_else(|_| {
                                serde_json::Value::String(builder.arguments.clone())
                            });

                        parts.push(ContentPart::tool_call(
                            builder.id.clone(),
                            builder.name.clone(),
                            arguments,
                            None,
                        ));
                    }
                }
            }
        }

        // Add thinking/reasoning if present
        if !self.thinking_buffer.is_empty() {
            parts.push(ContentPart::reasoning(&self.thinking_buffer));
        }

        // Determine final content
        let content = if parts.len() == 1 && parts[0].is_text() {
            MessageContent::Text(self.buffer.clone())
        } else if !parts.is_empty() {
            MessageContent::MultiModal(parts)
        } else {
            MessageContent::Text(String::new())
        };

        // Convert to nested provider_metadata structure
        let provider_metadata = if !stream_metadata.is_empty() {
            let mut meta = HashMap::new();
            meta.insert("stream".to_string(), stream_metadata);
            Some(meta)
        } else {
            None
        };

        ChatResponse {
            id: None,
            content,
            model: None,
            usage: self.current_usage.clone(),
            finish_reason,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata,
        }
    }
}

/// Processed Event
///
/// Result of processing a stream event, containing accumulated state.
#[derive(Debug, Clone)]
pub enum ProcessedEvent {
    /// Content update with delta and accumulated content
    ContentUpdate {
        delta: String,
        accumulated: String,
        index: Option<usize>,
    },
    /// Tool call update with current state
    ToolCallUpdate {
        id: String,
        current_state: ToolCallBuilder,
        index: Option<usize>,
    },
    /// Thinking update with delta and accumulated thinking
    ThinkingUpdate { delta: String, accumulated: String },
    /// Usage update
    UsageUpdate { usage: Usage },
    /// Stream start event
    StreamStart { metadata: ResponseMetadata },
    /// Stream end event
    StreamEnd { response: ChatResponse },
    /// Error event
    Error { error: LlmError },
    /// Custom provider-specific event (passed through without processing)
    Custom {
        event_type: String,
        data: serde_json::Value,
    },
}

/// Tool Call Builder
///
/// Accumulates tool call information incrementally during streaming.
#[derive(Debug, Clone)]
pub struct ToolCallBuilder {
    /// Tool call ID
    pub id: String,
    /// Tool type (deprecated, kept for compatibility)
    #[allow(dead_code)]
    pub r#type: Option<String>,
    /// Function name
    pub name: String,
    /// Function arguments (JSON string)
    pub arguments: String,
}

impl Default for ToolCallBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallBuilder {
    /// Create a new empty tool call builder
    pub const fn new() -> Self {
        Self {
            id: String::new(),
            r#type: None,
            name: String::new(),
            arguments: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_arguments_respect_max_size() {
        let mut cfg = StreamProcessorConfig::default();
        cfg.max_tool_arguments_size = Some(8);
        let mut called = false;
        cfg.overflow_handler = Some(Box::new(|name, size| {
            assert_eq!(name, "tool_arguments");
            assert!(size > 8);
            called = true;
        }));
        let mut sp = StreamProcessor::with_config(cfg);
        let ev = ChatStreamEvent::ToolCallDelta {
            id: "id1".into(),
            function_name: Some("fn".into()),
            arguments_delta: Some("abcdefghijk".into()),
            index: Some(0),
        };
        let _ = sp.process_event(ev);
        // Ensure builder exists and arguments have been truncated
        let b = sp.tool_calls.get("id1").unwrap();
        assert!(b.arguments.len() <= 8);
        assert!(called);
    }
}
