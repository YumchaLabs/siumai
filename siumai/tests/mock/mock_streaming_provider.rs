//! Mock Streaming Provider for Testing
//!
//! This module provides a mock streaming provider that can simulate
//! complete streaming scenarios for testing purposes.

use async_trait::async_trait;
use futures::{stream, StreamExt};
use siumai::stream::{ChatStream, ChatStreamEvent};
use siumai::types::{ChatMessage, ChatResponse, MessageContent, ResponseMetadata, Usage};
use siumai::error::LlmError;
use std::time::Duration;

/// Mock streaming provider for testing complete event sequences
#[derive(Debug, Clone)]
pub struct MockStreamingProvider {
    /// Simulated response metadata
    pub metadata: ResponseMetadata,
    /// Simulated content to stream
    pub content: String,
    /// Whether to include thinking content
    pub include_thinking: bool,
    /// Whether to include tool calls
    pub include_tool_calls: bool,
    /// Whether to include usage information
    pub include_usage: bool,
    /// Delay between events (for timing tests)
    pub event_delay: Option<Duration>,
}

impl Default for MockStreamingProvider {
    fn default() -> Self {
        Self {
            metadata: ResponseMetadata {
                id: Some("mock-123".to_string()),
                model: Some("mock-model".to_string()),
                created: Some(chrono::Utc::now()),
                provider: "mock".to_string(),
                request_id: Some("req-456".to_string()),
            },
            content: "Hello world!".to_string(),
            include_thinking: false,
            include_tool_calls: false,
            include_usage: true,
            event_delay: None,
        }
    }
}

impl MockStreamingProvider {
    /// Create a new mock provider with custom settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the content to stream
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }

    /// Include thinking content in the stream
    pub fn with_thinking(mut self, thinking: bool) -> Self {
        self.include_thinking = thinking;
        self
    }

    /// Include tool calls in the stream
    pub fn with_tool_calls(mut self, tool_calls: bool) -> Self {
        self.include_tool_calls = tool_calls;
        self
    }

    /// Include usage information
    pub fn with_usage(mut self, usage: bool) -> Self {
        self.include_usage = usage;
        self
    }

    /// Set delay between events
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.event_delay = Some(delay);
        self
    }

    /// Generate a complete stream of events
    pub async fn create_stream(&self) -> Result<ChatStream, LlmError> {
        let mut events = Vec::new();

        // 1. Always start with StreamStart
        events.push(Ok(ChatStreamEvent::StreamStart {
            metadata: self.metadata.clone(),
        }));

        // 2. Add thinking content if requested
        if self.include_thinking {
            events.push(Ok(ChatStreamEvent::ThinkingDelta {
                delta: "Let me think about this...".to_string(),
            }));
            
            if let Some(delay) = self.event_delay {
                tokio::time::sleep(delay).await;
            }
        }

        // 3. Add tool calls if requested
        if self.include_tool_calls {
            events.push(Ok(ChatStreamEvent::ToolCallDelta {
                id: "call_123".to_string(),
                function_name: Some("get_weather".to_string()),
                arguments_delta: None,
                index: Some(0),
            }));

            events.push(Ok(ChatStreamEvent::ToolCallDelta {
                id: "call_123".to_string(),
                function_name: None,
                arguments_delta: Some("{\"location\":\"New York\"}".to_string()),
                index: Some(0),
            }));

            if let Some(delay) = self.event_delay {
                tokio::time::sleep(delay).await;
            }
        }

        // 4. Stream content as deltas
        let words: Vec<&str> = self.content.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let delta = if i == 0 {
                word.to_string()
            } else {
                format!(" {}", word)
            };

            events.push(Ok(ChatStreamEvent::ContentDelta {
                delta,
                index: Some(0),
            }));

            if let Some(delay) = self.event_delay {
                tokio::time::sleep(delay).await;
            }
        }

        // 5. Add usage information if requested
        if self.include_usage {
            events.push(Ok(ChatStreamEvent::UsageUpdate {
                usage: Usage {
                    prompt_tokens: 10,
                    completion_tokens: words.len() as u32,
                    total_tokens: 10 + words.len() as u32,
                    reasoning_tokens: if self.include_thinking { Some(5) } else { None },
                    cached_tokens: None,
                },
            }));
        }

        // 6. End with StreamEnd
        events.push(Ok(ChatStreamEvent::StreamEnd {
            response: ChatResponse {
                id: self.metadata.id.clone(),
                content: MessageContent::Text(self.content.clone()),
                model: self.metadata.model.clone(),
                usage: if self.include_usage {
                    Some(Usage {
                        prompt_tokens: 10,
                        completion_tokens: words.len() as u32,
                        total_tokens: 10 + words.len() as u32,
                        reasoning_tokens: if self.include_thinking { Some(5) } else { None },
                        cached_tokens: None,
                    })
                } else {
                    None
                },
                finish_reason: Some(siumai::types::FinishReason::Stop),
                tool_calls: None,
                thinking: if self.include_thinking {
                    Some("Let me think about this...".to_string())
                } else {
                    None
                },
                metadata: std::collections::HashMap::new(),
            },
        }));

        // Convert to stream
        let stream = stream::iter(events);
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_mock_basic_stream() {
        let provider = MockStreamingProvider::new()
            .with_content("Hello world");

        let mut stream = provider.create_stream().await.unwrap();
        let mut events = Vec::new();

        while let Some(event) = stream.next().await {
            events.push(event.unwrap());
        }

        // Should have: StreamStart + 2 ContentDeltas + UsageUpdate + StreamEnd
        assert_eq!(events.len(), 5);

        // Verify sequence
        match &events[0] {
            ChatStreamEvent::StreamStart { metadata } => {
                assert_eq!(metadata.provider, "mock");
            }
            _ => panic!("Expected StreamStart"),
        }

        match &events[1] {
            ChatStreamEvent::ContentDelta { delta, .. } => {
                assert_eq!(delta, "Hello");
            }
            _ => panic!("Expected ContentDelta"),
        }

        match &events[4] {
            ChatStreamEvent::StreamEnd { .. } => {
                // Expected
            }
            _ => panic!("Expected StreamEnd"),
        }
    }

    #[tokio::test]
    async fn test_mock_complete_stream() {
        let provider = MockStreamingProvider::new()
            .with_content("Test response")
            .with_thinking(true)
            .with_tool_calls(true)
            .with_usage(true);

        let mut stream = provider.create_stream().await.unwrap();
        let mut events = Vec::new();

        while let Some(event) = stream.next().await {
            events.push(event.unwrap());
        }

        // Should have all event types
        let has_stream_start = events.iter().any(|e| matches!(e, ChatStreamEvent::StreamStart { .. }));
        let has_thinking = events.iter().any(|e| matches!(e, ChatStreamEvent::ThinkingDelta { .. }));
        let has_tool_call = events.iter().any(|e| matches!(e, ChatStreamEvent::ToolCallDelta { .. }));
        let has_content = events.iter().any(|e| matches!(e, ChatStreamEvent::ContentDelta { .. }));
        let has_usage = events.iter().any(|e| matches!(e, ChatStreamEvent::UsageUpdate { .. }));
        let has_stream_end = events.iter().any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. }));

        assert!(has_stream_start, "Should have StreamStart");
        assert!(has_thinking, "Should have ThinkingDelta");
        assert!(has_tool_call, "Should have ToolCallDelta");
        assert!(has_content, "Should have ContentDelta");
        assert!(has_usage, "Should have UsageUpdate");
        assert!(has_stream_end, "Should have StreamEnd");
    }
}
