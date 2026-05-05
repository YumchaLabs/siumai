//! Custom Provider Streaming Tests
//!
//! This test demonstrates how to use custom providers to test streaming functionality.
//! Custom providers are simpler than mock HTTP servers and provide full control over
//! the streaming event sequence.

use futures_util::StreamExt;
use siumai::prelude::unified::{
    ChatCapability, ChatMessage, ChatResponse, ChatStream, ChatStreamEvent, ChatStreamPart,
    FinishReason, LlmError, MessageContent, ResponseMetadata, Tool, Usage,
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
                    headers: None,
                    body: None,
                },
            },
            ChatStreamEvent::text_delta_part("0", "Hello"),
            ChatStreamEvent::text_delta_part("0", " "),
            ChatStreamEvent::text_delta_part("0", "World"),
            ChatStreamEvent::text_delta_part("0", "!"),
            ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("test-123".to_string()),
                    content: MessageContent::Text("Hello World!".to_string()),
                    model: Some("test-model".to_string()),
                    usage: Some(Usage::new(5, 10)),
                    finish_reason: Some(FinishReason::Stop),
                    raw_finish_reason: None,
                    provider_metadata: None,
                    warnings: None,
                    request: None,
                    response: None,
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
                    headers: None,
                    body: None,
                },
            },
            ChatStreamEvent::tool_input_start_part("call_123", "get_weather"),
            ChatStreamEvent::tool_input_delta_part("call_123", r#"{"location":""#),
            ChatStreamEvent::tool_input_delta_part("call_123", r#"San Francisco"}"#),
            ChatStreamEvent::tool_input_end_part("call_123"),
            ChatStreamEvent::tool_call_part(
                "call_123",
                "get_weather",
                r#"{"location":"San Francisco"}"#,
            ),
            ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("test-456".to_string()),
                    content: MessageContent::Text("".to_string()),
                    model: Some("test-model".to_string()),
                    usage: Some(Usage::new(10, 5)),
                    finish_reason: Some(FinishReason::ToolCalls),
                    raw_finish_reason: None,
                    provider_metadata: None,
                    warnings: None,
                    request: None,
                    response: None,
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
                    headers: None,
                    body: None,
                },
            },
            ChatStreamEvent::text_delta_part("0", "Test"),
            ChatStreamEvent::finish_part(Usage::new(5, 1), FinishReason::Unknown),
            ChatStreamEvent::text_delta_part("0", " response"),
            ChatStreamEvent::finish_part(Usage::new(5, 3), FinishReason::Stop),
            ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: Some("test-789".to_string()),
                    content: MessageContent::Text("Test response".to_string()),
                    model: Some("test-model".to_string()),
                    usage: Some(Usage::new(5, 3)),
                    finish_reason: Some(FinishReason::Stop),
                    raw_finish_reason: None,
                    provider_metadata: None,
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    request: None,
                    response: None,
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

    // Verify TextDelta events
    let content_deltas: Vec<String> = events
        .iter()
        .filter_map(|e| e.text_delta().map(ToString::to_string))
        .collect();
    assert_eq!(content_deltas, vec!["Hello", " ", "World", "!"]);

    // Verify StreamEnd
    if let ChatStreamEvent::StreamEnd { response } = &events[5] {
        assert_eq!(response.id, Some("test-123".to_string()));
        assert_eq!(response.content.text(), Some("Hello World!"));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert!(response.usage.is_some());
        let usage = response.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens(), Some(5));
        assert_eq!(usage.completion_tokens(), Some(10));
        assert_eq!(usage.total_tokens(), Some(15));
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
    assert_eq!(events.len(), 7);

    // Verify StreamStart
    assert!(matches!(events[0], ChatStreamEvent::StreamStart { .. }));

    // Verify typed tool input events
    let tool_input_parts: Vec<_> = events
        .iter()
        .filter_map(|e| match e.part_ref()? {
            ChatStreamPart::ToolInputStart { id, tool_name, .. } => {
                Some((id.clone(), Some(tool_name.clone()), None))
            }
            ChatStreamPart::ToolInputDelta { id, delta, .. } => {
                Some((id.clone(), None, Some(delta.clone())))
            }
            _ => None,
        })
        .collect();

    assert_eq!(tool_input_parts.len(), 3);
    assert_eq!(tool_input_parts[0].0, "call_123");
    assert_eq!(tool_input_parts[0].1, Some("get_weather".to_string()));

    // Combine arguments
    let combined_args: String = tool_input_parts
        .iter()
        .filter_map(|(_, _, args)| args.clone())
        .collect();
    assert_eq!(combined_args, r#"{"location":"San Francisco"}"#);

    // Verify StreamEnd
    if let ChatStreamEvent::StreamEnd { response } = &events[6] {
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

    // Verify Finish usage events
    let usage_updates: Vec<Usage> = events
        .iter()
        .filter_map(|e| e.finish_usage().cloned())
        .collect();

    assert_eq!(usage_updates.len(), 2);
    assert_eq!(usage_updates[0].total_tokens(), Some(6));
    assert_eq!(usage_updates[1].total_tokens(), Some(8));

    // Verify final usage in StreamEnd
    if let ChatStreamEvent::StreamEnd { response } = &events[5] {
        let usage = response.usage.as_ref().unwrap();
        assert_eq!(usage.total_tokens(), Some(8));
    } else {
        panic!("Expected StreamEnd event");
    }
}
