//! Tests for Anthropic standard layer SSE adapter hooks.
//!
//! These tests verify that `AnthropicChatStandard` correctly applies
//! adapter transformations to SSE events during streaming.

use eventsource_stream::Event;
use std::sync::Arc;

use crate::error::LlmError;
use crate::standards::anthropic::chat::{AnthropicChatAdapter, AnthropicChatStandard};
use crate::streaming::ChatStreamEvent;

/// Mock adapter that transforms SSE events by adding a tag to text deltas.
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
        _req: &crate::types::ChatRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }

    fn transform_sse_event(&self, event: &mut serde_json::Value) -> Result<(), LlmError> {
        if event.get("type").and_then(|v| v.as_str()) == Some("content_block_delta")
            && let Some(delta) = event.get_mut("delta")
            && let Some(text) = delta.get_mut("text").and_then(|v| v.as_str())
        {
            delta["text"] = serde_json::Value::String(format!("{}{}", self.tag, text));
        }
        Ok(())
    }
}

#[tokio::test]
async fn anthropic_standard_adapter_transforms_sse_events() {
    let adapter = Arc::new(MockAnthropicAdapter::new("[ADAPTED] "));
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
    assert!(!result.is_empty(), "should have at least one event");

    let mut found_transformed = false;
    for event_result in result {
        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event_result
            && delta.starts_with("[ADAPTED] ")
        {
            found_transformed = true;
            assert_eq!(delta, "[ADAPTED] Hello");
        }
    }
    assert!(
        found_transformed,
        "should have found transformed content delta"
    );
}

#[tokio::test]
async fn anthropic_standard_without_adapter_no_transformation() {
    let standard = AnthropicChatStandard::new();

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
    assert!(!result.is_empty());

    let mut found_content = false;
    for event_result in result {
        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event_result {
            found_content = true;
            assert_eq!(delta, "Hello", "content should not be transformed");
        }
    }
    assert!(found_content, "should have found content delta");
}

#[tokio::test]
async fn anthropic_adapter_error_handling() {
    #[derive(Debug, Clone)]
    struct FailingAdapter;

    impl AnthropicChatAdapter for FailingAdapter {
        fn transform_request(
            &self,
            _req: &crate::types::ChatRequest,
            _body: &mut serde_json::Value,
        ) -> Result<(), LlmError> {
            Ok(())
        }

        fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
            Ok(())
        }

        fn transform_sse_event(&self, _event: &mut serde_json::Value) -> Result<(), LlmError> {
            Err(LlmError::InternalError(
                "Adapter transformation failed".to_string(),
            ))
        }
    }

    let standard = AnthropicChatStandard::with_adapter(Arc::new(FailingAdapter));
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
    assert_eq!(result.len(), 1);
    assert!(result[0].is_err(), "should return error from adapter");
}

#[tokio::test]
async fn anthropic_adapter_invalid_json_handling_is_non_panicking() {
    let standard =
        AnthropicChatStandard::with_adapter(Arc::new(MockAnthropicAdapter::new("[TAG] ")));
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    let event = Event {
        event: "".to_string(),
        data: r#"{"invalid_field_that_doesnt_exist_in_anthropic_format": true}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;
    assert!(!result.is_empty(), "should return at least one result");
}
