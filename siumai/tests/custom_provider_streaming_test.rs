//! Custom Provider Streaming Tests
//!
//! This test demonstrates how to use custom providers to test streaming functionality.
//! Custom providers are simpler than mock HTTP servers and provide full control over
//! the streaming event sequence.

use futures_util::StreamExt;
use siumai::error::LlmError;
use siumai::streaming::{ChatStream, ChatStreamEvent};
use siumai::traits::ChatCapability;
use siumai::types::{
    ChatMessage, ChatResponse, FinishReason, MessageContent, ResponseMetadata, Tool, Usage,
};
use std::sync::Arc;
use tokio::sync::Mutex;

/// A simple custom provider for testing streaming
#[derive(Clone)]
struct TestStreamingProvider {
    /// Predefined events to return
    events: Arc<Mutex<Vec<ChatStreamEvent>>>,
}

impl TestStreamingProvider {
    fn new(events: Vec<ChatStreamEvent>) -> Self {
        Self {
            events: Arc::new(Mutex::new(events)),
        }
    }

    /// Create a provider that returns a simple content stream
    fn with_content_stream() -> Self {
        let events = vec![
            ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("test-123".to_string()),
                    model: Some("test-model".to_string()),
                    created: None,
                    provider: "test".to_string(),
                    request_id: None,
                },
            },
            ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            },
            ChatStreamEvent::ContentDelta {
                delta: " ".to_string(),
                index: None,
            },
            ChatStreamEvent::ContentDelta {
                delta: "World".to_string(),
                index: None,
            },
            ChatStreamEvent::ContentDelta {
                delta: "!".to_string(),
                index: None,
            },
            ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("test-123".to_string()),
                    content: MessageContent::Text("Hello World!".to_string()),
                    model: Some("test-model".to_string()),
                    usage: Some(Usage {
                        prompt_tokens: 5,
                        completion_tokens: 10,
                        total_tokens: 15,
                        cached_tokens: None,
                        reasoning_tokens: None,
                        completion_tokens_details: None,
                        prompt_tokens_details: None,
                    }),
                    finish_reason: Some(FinishReason::Stop),
                    provider_metadata: None,
                    warnings: None,
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                },
            },
        ];
        Self::new(events)
    }

    /// Create a provider that returns a stream with tool calls
    fn with_tool_call_stream() -> Self {
        let events = vec![
            ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("test-456".to_string()),
                    model: Some("test-model".to_string()),
                    created: None,
                    provider: "test".to_string(),
                    request_id: None,
                },
            },
            ChatStreamEvent::ToolCallDelta {
                id: "call_123".to_string(),
                function_name: Some("get_weather".to_string()),
                arguments_delta: None,
                index: None,
            },
            ChatStreamEvent::ToolCallDelta {
                id: "call_123".to_string(),
                function_name: None,
                arguments_delta: Some(r#"{"location":""#.to_string()),
                index: None,
            },
            ChatStreamEvent::ToolCallDelta {
                id: "call_123".to_string(),
                function_name: None,
                arguments_delta: Some(r#"San Francisco"}"#.to_string()),
                index: None,
            },
            ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("test-456".to_string()),
                    content: MessageContent::Text("".to_string()),
                    model: Some("test-model".to_string()),
                    usage: Some(Usage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                        cached_tokens: None,
                        reasoning_tokens: None,
                        completion_tokens_details: None,
                        prompt_tokens_details: None,
                    }),
                    finish_reason: Some(FinishReason::ToolCalls),
                    provider_metadata: None,
                    warnings: None,
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                },
            },
        ];
        Self::new(events)
    }

    /// Create a provider that returns a stream with usage updates
    fn with_usage_updates() -> Self {
        let events = vec![
            ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("test-789".to_string()),
                    model: Some("test-model".to_string()),
                    created: None,
                    provider: "test".to_string(),
                    request_id: None,
                },
            },
            ChatStreamEvent::ContentDelta {
                delta: "Test".to_string(),
                index: None,
            },
            ChatStreamEvent::UsageUpdate {
                usage: Usage {
                    prompt_tokens: 5,
                    completion_tokens: 1,
                    total_tokens: 6,
                    cached_tokens: None,
                    reasoning_tokens: None,
                    completion_tokens_details: None,
                    prompt_tokens_details: None,
                },
            },
            ChatStreamEvent::ContentDelta {
                delta: " response".to_string(),
                index: None,
            },
            ChatStreamEvent::UsageUpdate {
                usage: Usage {
                    prompt_tokens: 5,
                    completion_tokens: 3,
                    total_tokens: 8,
                    cached_tokens: None,
                    reasoning_tokens: None,
                    completion_tokens_details: None,
                    prompt_tokens_details: None,
                },
            },
            ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("test-789".to_string()),
                    content: MessageContent::Text("Test response".to_string()),
                    model: Some("test-model".to_string()),
                    usage: Some(Usage {
                        prompt_tokens: 5,
                        completion_tokens: 3,
                        total_tokens: 8,
                        cached_tokens: None,
                        reasoning_tokens: None,
                        completion_tokens_details: None,
                        prompt_tokens_details: None,
                    }),
                    finish_reason: Some(FinishReason::Stop),
                    provider_metadata: None,
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                },
            },
        ];
        Self::new(events)
    }
}

#[async_trait::async_trait]
impl ChatCapability for TestStreamingProvider {
    async fn chat_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Use chat_stream for this test provider".to_string(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let events = self.events.lock().await.clone();
        let stream = futures_util::stream::iter(events.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }
}

#[tokio::test]
async fn test_custom_provider_content_streaming() {
    let provider = TestStreamingProvider::with_content_stream();

    let mut stream = provider
        .chat_stream(vec![ChatMessage::user("Hello").build()], None)
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }

    // Verify we got all expected events
    assert_eq!(events.len(), 6);

    // Verify StreamStart
    assert!(matches!(events[0], ChatStreamEvent::StreamStart { .. }));

    // Verify ContentDelta events
    let content_deltas: Vec<String> = events
        .iter()
        .filter_map(|e| {
            if let ChatStreamEvent::ContentDelta { delta, .. } = e {
                Some(delta.clone())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(content_deltas, vec!["Hello", " ", "World", "!"]);

    // Verify StreamEnd
    if let ChatStreamEvent::StreamEnd { response } = &events[5] {
        assert_eq!(response.id, Some("test-123".to_string()));
        assert_eq!(response.content.text(), Some("Hello World!"));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert!(response.usage.is_some());
        let usage = response.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 10);
        assert_eq!(usage.total_tokens, 15);
    } else {
        panic!("Expected StreamEnd event");
    }
}

#[tokio::test]
async fn test_custom_provider_tool_call_streaming() {
    let provider = TestStreamingProvider::with_tool_call_stream();

    let mut stream = provider
        .chat_stream(vec![ChatMessage::user("What's the weather?").build()], None)
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }

    // Verify we got all expected events
    assert_eq!(events.len(), 5);

    // Verify StreamStart
    assert!(matches!(events[0], ChatStreamEvent::StreamStart { .. }));

    // Verify ToolCallDelta events
    let tool_call_deltas: Vec<_> = events
        .iter()
        .filter_map(|e| {
            if let ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            } = e
            {
                Some((id.clone(), function_name.clone(), arguments_delta.clone()))
            } else {
                None
            }
        })
        .collect();

    assert_eq!(tool_call_deltas.len(), 3);
    assert_eq!(tool_call_deltas[0].0, "call_123");
    assert_eq!(tool_call_deltas[0].1, Some("get_weather".to_string()));

    // Combine arguments
    let combined_args: String = tool_call_deltas
        .iter()
        .filter_map(|(_, _, args)| args.clone())
        .collect();
    assert_eq!(combined_args, r#"{"location":"San Francisco"}"#);

    // Verify StreamEnd
    if let ChatStreamEvent::StreamEnd { response } = &events[4] {
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    } else {
        panic!("Expected StreamEnd event");
    }
}

#[tokio::test]
async fn test_custom_provider_usage_updates() {
    let provider = TestStreamingProvider::with_usage_updates();

    let mut stream = provider
        .chat_stream(vec![ChatMessage::user("Test").build()], None)
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }

    // Verify we got all expected events
    assert_eq!(events.len(), 6);

    // Verify UsageUpdate events
    let usage_updates: Vec<Usage> = events
        .iter()
        .filter_map(|e| {
            if let ChatStreamEvent::UsageUpdate { usage } = e {
                Some(usage.clone())
            } else {
                None
            }
        })
        .collect();

    assert_eq!(usage_updates.len(), 2);
    assert_eq!(usage_updates[0].total_tokens, 6);
    assert_eq!(usage_updates[1].total_tokens, 8);

    // Verify final usage in StreamEnd
    if let ChatStreamEvent::StreamEnd { response } = &events[5] {
        let usage = response.usage.as_ref().unwrap();
        assert_eq!(usage.total_tokens, 8);
    } else {
        panic!("Expected StreamEnd event");
    }
}
