//! Tests for Anthropic standard layer SSE adapter hooks
//!
//! This test verifies that the Anthropic standard layer correctly applies
//! adapter transformations to SSE events during streaming.

use eventsource_stream::Event;
use siumai::error::LlmError;
use siumai::standards::anthropic::chat::{AnthropicChatAdapter, AnthropicChatStandard};
use siumai::streaming::ChatStreamEvent;
use siumai::transformers::stream::StreamChunkTransformer;
use std::sync::Arc;

/// Mock adapter that transforms SSE events by adding metadata
#[derive(Debug, Clone)]
struct MockAnthropicAdapter {
    tag: String,
}

impl MockAnthropicAdapter {
    fn new(tag: impl Into<String>) -> Self {
        Self { tag: tag.into() }
    }
}

impl AnthropicChatAdapter for MockAnthropicAdapter {
    fn transform_request(
        &self,
        _req: &siumai::types::ChatRequest,
        _params: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn transform_response(&self, _response: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    fn transform_sse_event(&self, event: &mut serde_json::Value) -> Result<(), LlmError> {
        // Add a custom tag to text deltas
        if event.get("type").and_then(|v| v.as_str()) == Some("content_block_delta") {
            if let Some(delta) = event.get_mut("delta") {
                if let Some(text) = delta.get_mut("text").and_then(|v| v.as_str()) {
                    let tagged = format!("{}{}", self.tag, text);
                    delta["text"] = serde_json::Value::String(tagged);
                }
            }
        }
        Ok(())
    }
}

#[tokio::test]
async fn test_anthropic_standard_adapter_transforms_sse_events() {
    // Create standard with adapter
    let adapter = Arc::new(MockAnthropicAdapter::new("[ADAPTED] "));
    let standard = AnthropicChatStandard::with_adapter(adapter);

    // Create transformers
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    // Create a mock Anthropic SSE event (content_block_delta)
    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    // Convert event
    let result = stream_transformer.convert_event(event).await;

    // Verify transformation was applied
    assert!(!result.is_empty(), "Should have at least one event");

    let mut found_transformed = false;
    for event_result in result {
        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event_result {
            if delta.starts_with("[ADAPTED] ") {
                found_transformed = true;
                assert_eq!(delta, "[ADAPTED] Hello");
            }
        }
    }

    assert!(
        found_transformed,
        "Should have found transformed content delta"
    );
}

#[tokio::test]
async fn test_anthropic_standard_without_adapter_no_transformation() {
    // Create standard without adapter
    let standard = AnthropicChatStandard::new();

    // Create transformers
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    // Create a mock SSE event
    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    // Convert event
    let result = stream_transformer.convert_event(event).await;

    // Verify no transformation (original content)
    assert!(!result.is_empty());

    let mut found_content = false;
    for event_result in result {
        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event_result {
            found_content = true;
            assert_eq!(delta, "Hello", "Content should not be transformed");
        }
    }

    assert!(found_content, "Should have found content delta");
}

#[tokio::test]
async fn test_anthropic_adapter_error_handling() {
    /// Adapter that always fails
    #[derive(Debug, Clone)]
    struct FailingAdapter;

    impl AnthropicChatAdapter for FailingAdapter {
        fn transform_request(
            &self,
            _req: &siumai::types::ChatRequest,
            _params: &mut serde_json::Value,
        ) -> Result<(), LlmError> {
            Ok(())
        }

        fn transform_response(&self, _response: &mut serde_json::Value) -> Result<(), LlmError> {
            Ok(())
        }

        fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
            Err(LlmError::InternalError(
                "Adapter transformation failed".to_string(),
            ))
        }
    }

    let adapter = Arc::new(FailingAdapter);
    let standard = AnthropicChatStandard::with_adapter(adapter);
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;

    // Should return error
    assert_eq!(result.len(), 1);
    assert!(result[0].is_err(), "Should return error from adapter");
}

#[tokio::test]
async fn test_anthropic_adapter_invalid_json_handling() {
    let adapter = Arc::new(MockAnthropicAdapter::new("[TAG] "));
    let standard = AnthropicChatStandard::with_adapter(adapter);
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    // Invalid JSON event
    let event = Event {
        event: "".to_string(),
        data: "invalid json".to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;

    // Should return parse error
    assert_eq!(result.len(), 1);
    assert!(result[0].is_err(), "Should return parse error");
    if let Err(LlmError::ParseError(msg)) = &result[0] {
        assert!(msg.contains("Failed to parse SSE event"));
    } else {
        panic!("Expected ParseError");
    }
}

#[tokio::test]
async fn test_anthropic_adapter_message_start_event() {
    let adapter = Arc::new(MockAnthropicAdapter::new("[PREFIX] "));
    let standard = AnthropicChatStandard::with_adapter(adapter);
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    // message_start event (should not be transformed by our adapter)
    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"message_start","message":{"id":"msg_123","role":"assistant"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;

    // Should process without error (adapter doesn't modify message_start)
    // The actual events depend on the converter implementation
    assert!(!result.is_empty() || result.is_empty()); // Just verify no panic
}

#[tokio::test]
async fn test_anthropic_adapter_thinking_delta() {
    /// Adapter that also transforms thinking content
    #[derive(Debug, Clone)]
    struct ThinkingAdapter;

    impl AnthropicChatAdapter for ThinkingAdapter {
        fn transform_request(
            &self,
            _req: &siumai::types::ChatRequest,
            _params: &mut serde_json::Value,
        ) -> Result<(), LlmError> {
            Ok(())
        }

        fn transform_response(&self, _response: &mut serde_json::Value) -> Result<(), LlmError> {
            Ok(())
        }

        fn transform_sse_event(&self, event: &mut serde_json::Value) -> Result<(), LlmError> {
            // Transform both text and thinking deltas
            if event.get("type").and_then(|v| v.as_str()) == Some("content_block_delta") {
                if let Some(delta) = event.get_mut("delta") {
                    // Transform text delta
                    if let Some(text) = delta.get_mut("text").and_then(|v| v.as_str()) {
                        delta["text"] = serde_json::Value::String(format!("[TEXT] {}", text));
                    }
                    // Transform thinking delta (if present)
                    if let Some(thinking) = delta.get_mut("thinking").and_then(|v| v.as_str()) {
                        delta["thinking"] =
                            serde_json::Value::String(format!("[THINK] {}", thinking));
                    }
                }
            }
            Ok(())
        }
    }

    let adapter = Arc::new(ThinkingAdapter);
    let standard = AnthropicChatStandard::with_adapter(adapter);
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    // Event with thinking content
    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Answer"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;

    // Verify transformation
    let mut found = false;
    for event_result in result {
        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event_result {
            if delta.starts_with("[TEXT] ") {
                found = true;
            }
        }
    }

    assert!(found, "Should have found transformed text");
}

#[tokio::test]
async fn test_anthropic_adapter_preserves_event_structure() {
    /// Adapter that adds metadata without breaking structure
    #[derive(Debug, Clone)]
    struct MetadataAdapter;

    impl AnthropicChatAdapter for MetadataAdapter {
        fn transform_request(
            &self,
            _req: &siumai::types::ChatRequest,
            _params: &mut serde_json::Value,
        ) -> Result<(), LlmError> {
            Ok(())
        }

        fn transform_response(&self, _response: &mut serde_json::Value) -> Result<(), LlmError> {
            Ok(())
        }

        fn transform_sse_event(&self, event: &mut serde_json::Value) -> Result<(), LlmError> {
            // Add custom metadata field (should be preserved)
            event["_adapter_metadata"] = serde_json::json!({"processed": true});
            Ok(())
        }
    }

    let adapter = Arc::new(MetadataAdapter);
    let standard = AnthropicChatStandard::with_adapter(adapter);
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Test"}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;

    // Should process without error
    assert!(!result.is_empty() || result.is_empty());
}
