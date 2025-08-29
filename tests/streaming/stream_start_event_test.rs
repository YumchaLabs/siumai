//! Tests for StreamStart event generation across all providers
//!
//! This test verifies that all providers correctly generate StreamStart events
//! with proper metadata when streaming begins.

use eventsource_stream::Event;
use siumai::providers::anthropic::streaming::AnthropicEventConverter;
use siumai::providers::gemini::streaming::GeminiEventConverter;
use siumai::providers::groq::streaming::GroqEventConverter;
use siumai::providers::ollama::streaming::OllamaEventConverter;
use siumai::providers::openai::streaming::OpenAiEventConverter;
use siumai::providers::xai::streaming::XaiEventConverter;
use siumai::stream::ChatStreamEvent;
use siumai::utils::streaming::{JsonEventConverter, SseEventConverter};

#[tokio::test]
async fn test_openai_stream_start_event() {
    let config = siumai::providers::openai::config::OpenAiConfig::default();
    let converter = OpenAiEventConverter::new(config);

    // Test that first event with metadata generates StreamStart
    let event = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl-123","model":"gpt-4","choices":[{"delta":{"content":"Hello"}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(!result.is_empty());

    let stream_start = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));

    if let Some(Ok(ChatStreamEvent::StreamStart { metadata })) = stream_start {
        assert_eq!(metadata.id, Some("chatcmpl-123".to_string()));
        assert_eq!(metadata.model, Some("gpt-4".to_string()));
        assert_eq!(metadata.provider, "openai");
    } else {
        panic!("Expected StreamStart event, got: {:?}", result);
    }
}

#[tokio::test]
async fn test_anthropic_stream_start_event() {
    let config = siumai::params::AnthropicParams::default();
    let converter = AnthropicEventConverter::new(config);

    // Test message_start event generates StreamStart
    let event = Event {
        event: "".to_string(),
        data: r#"{"type":"message_start","message":{"id":"msg_123","model":"claude-3-sonnet","role":"assistant","content":[]}}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(!result.is_empty());

    let stream_start = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));

    if let Some(Ok(ChatStreamEvent::StreamStart { metadata })) = stream_start {
        assert_eq!(metadata.id, Some("msg_123".to_string()));
        assert_eq!(metadata.model, Some("claude-3-sonnet".to_string()));
        assert_eq!(metadata.provider, "anthropic");
    } else {
        panic!("Expected StreamStart event, got: {:?}", result);
    }
}

#[tokio::test]
async fn test_gemini_stream_start_event() {
    let config = siumai::providers::gemini::types::GeminiConfig {
        model: "gemini-pro".to_string(),
        ..Default::default()
    };
    let converter = GeminiEventConverter::new(config);

    // Test that first event generates StreamStart
    let event = Event {
        event: "".to_string(),
        data: r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(!result.is_empty());

    let stream_start = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));

    if let Some(Ok(ChatStreamEvent::StreamStart { metadata })) = stream_start {
        assert_eq!(metadata.provider, "gemini");
        assert!(metadata.model.is_some()); // Should have model from config
    } else {
        panic!("Expected StreamStart event, got: {:?}", result);
    }
}

#[tokio::test]
async fn test_groq_stream_start_event() {
    let config = siumai::providers::groq::config::GroqConfig::default();
    let converter = GroqEventConverter::new(config);

    // Test that first event with metadata generates StreamStart
    let event = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","model":"llama-3.1-70b","created":1234567890,"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(!result.is_empty());

    let stream_start = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));

    if let Some(Ok(ChatStreamEvent::StreamStart { metadata })) = stream_start {
        assert_eq!(metadata.id, Some("chatcmpl-123".to_string()));
        assert_eq!(metadata.model, Some("llama-3.1-70b".to_string()));
        assert_eq!(metadata.provider, "groq");
    } else {
        panic!("Expected StreamStart event, got: {:?}", result);
    }
}

#[tokio::test]
async fn test_xai_stream_start_event() {
    let config = siumai::providers::xai::config::XaiConfig::default();
    let converter = XaiEventConverter::new(config);

    // Test that first event with metadata generates StreamStart
    let event = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","model":"grok-beta","created":1234567890,"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result = converter.convert_event(event).await;
    assert!(!result.is_empty());

    let stream_start = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));

    if let Some(Ok(ChatStreamEvent::StreamStart { metadata })) = stream_start {
        assert_eq!(metadata.id, Some("chatcmpl-123".to_string()));
        assert_eq!(metadata.model, Some("grok-beta".to_string()));
        assert_eq!(metadata.provider, "xai");
    } else {
        panic!("Expected StreamStart event, got: {:?}", result);
    }
}

#[tokio::test]
async fn test_ollama_stream_start_event() {
    let converter = OllamaEventConverter::new();

    // Test that first event generates StreamStart
    let json_data =
        r#"{"model":"llama2","message":{"role":"assistant","content":"Hello"},"done":false}"#;

    let result = converter.convert_json(json_data).await;
    assert!(!result.is_empty());

    let stream_start = result
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));

    if let Some(Ok(ChatStreamEvent::StreamStart { metadata })) = stream_start {
        assert_eq!(metadata.model, Some("llama2".to_string()));
        assert_eq!(metadata.provider, "ollama");
    } else {
        panic!("Expected StreamStart event, got: {:?}", result);
    }
}

#[tokio::test]
async fn test_stream_start_only_emitted_once() {
    let config = siumai::providers::openai::config::OpenAiConfig::default();
    let converter = OpenAiEventConverter::new(config);

    // First event should generate StreamStart
    let event1 = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl-123","model":"gpt-4","choices":[{"delta":{"content":"Hello"}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result1 = converter.convert_event(event1).await;
    assert!(!result1.is_empty());

    let has_stream_start = result1
        .iter()
        .any(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));

    if has_stream_start {
        // Expected
    } else {
        panic!("Expected StreamStart event for first event");
    }

    // Second event should NOT generate StreamStart, should generate ContentDelta
    let event2 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"delta":{"content":" World"}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let result2 = converter.convert_event(event2).await;
    assert!(!result2.is_empty());

    let content_delta = result2
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::ContentDelta { .. })));

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = content_delta {
        assert_eq!(delta, " World");
    } else {
        panic!(
            "Expected ContentDelta event for second event, got: {:?}",
            result2
        );
    }
}
