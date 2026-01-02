//! Tests for OpenAI standard layer SSE adapter hooks.
//!
//! These tests verify that `OpenAiChatStandard` correctly applies
//! adapter transformations to SSE events during streaming.

use eventsource_stream::Event;
use std::sync::Arc;

use crate::error::LlmError;
use crate::standards::openai::chat::{OpenAiChatAdapter, OpenAiChatStandard};
use crate::streaming::ChatStreamEvent;

/// Mock adapter that transforms SSE events by adding a prefix to content.
#[derive(Debug, Clone)]
struct MockOpenAiAdapter {
    prefix: String,
}

impl MockOpenAiAdapter {
    fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }
}

impl OpenAiChatAdapter for MockOpenAiAdapter {
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
        if let Some(choices) = event.get_mut("choices").and_then(|v| v.as_array_mut()) {
            for choice in choices {
                if let Some(delta) = choice.get_mut("delta")
                    && let Some(content) = delta.get_mut("content").and_then(|v| v.as_str())
                {
                    delta["content"] =
                        serde_json::Value::String(format!("{}{}", self.prefix, content));
                }
            }
        }
        Ok(())
    }
}

#[tokio::test]
async fn openai_standard_adapter_transforms_sse_events() {
    let adapter = Arc::new(MockOpenAiAdapter::new("[TRANSFORMED] "));
    let standard = OpenAiChatStandard::with_adapter(adapter);

    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    let event = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"delta":{"content":"Hello"}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;
    assert!(!result.is_empty(), "should have at least one event");

    let mut found_transformed = false;
    for event_result in result {
        if let Ok(ChatStreamEvent::ContentDelta { delta, .. }) = event_result
            && delta.starts_with("[TRANSFORMED] ")
        {
            found_transformed = true;
            assert_eq!(delta, "[TRANSFORMED] Hello");
        }
    }
    assert!(
        found_transformed,
        "should have found transformed content delta"
    );
}

#[tokio::test]
async fn openai_standard_without_adapter_no_transformation() {
    let standard = OpenAiChatStandard::new();

    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    let event = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"delta":{"content":"Hello"}}]}"#.to_string(),
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
async fn openai_adapter_error_handling() {
    #[derive(Debug, Clone)]
    struct FailingAdapter;

    impl OpenAiChatAdapter for FailingAdapter {
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

    let standard = OpenAiChatStandard::with_adapter(Arc::new(FailingAdapter));
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    let event = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"delta":{"content":"Hello"}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;
    assert_eq!(result.len(), 1);
    assert!(result[0].is_err(), "should return error from adapter");
}

#[tokio::test]
async fn openai_adapter_invalid_json_handling_is_non_panicking() {
    let standard = OpenAiChatStandard::with_adapter(Arc::new(MockOpenAiAdapter::new("[PREFIX] ")));
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    let event = Event {
        event: "".to_string(),
        data: r#"{"invalid_field_that_doesnt_exist_in_openai_format": true}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;
    assert!(!result.is_empty(), "should return at least one result");
}

#[tokio::test]
async fn openai_adapter_uses_first_choice_when_multiple() {
    let standard = OpenAiChatStandard::with_adapter(Arc::new(MockOpenAiAdapter::new("[MULTI] ")));
    let transformers = standard.create_transformers("test-provider");
    let stream_transformer = transformers.stream.expect("stream transformer");

    let event = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"delta":{"content":"First"}},{"delta":{"content":"Second"}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = stream_transformer.convert_event(event).await;
    let deltas = result
        .into_iter()
        .filter_map(|r| match r {
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) => Some(delta),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(deltas.iter().any(|d| d == "[MULTI] First"));
}
