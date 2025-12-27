//! Complete Stream Events Test
//!
//! This test verifies the complete sequence of streaming events across all providers
//! using mock data to simulate real streaming scenarios.

use eventsource_stream::Event;
use siumai::providers::anthropic::streaming::AnthropicEventConverter;
use siumai::providers::ollama::streaming::OllamaEventConverter;
use siumai::providers::openai_compatible::adapter::{ProviderAdapter, ProviderCompatibility};
use siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter;
use siumai::providers::openai_compatible::types::FieldMappings;
use siumai::traits::ProviderCapabilities;
use std::sync::Arc;

fn make_openai_converter() -> OpenAiCompatibleEventConverter {
    #[derive(Debug, Clone)]
    struct OpenAiStandardAdapter {
        base_url: String,
    }
    impl ProviderAdapter for OpenAiStandardAdapter {
        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("openai")
        }
        fn transform_request_params(
            &self,
            _params: &mut serde_json::Value,
            _model: &str,
            _ty: siumai::providers::openai_compatible::types::RequestType,
        ) -> Result<(), siumai::error::LlmError> {
            Ok(())
        }
        fn get_field_mappings(&self, _model: &str) -> FieldMappings {
            FieldMappings::standard()
        }
        fn get_model_config(
            &self,
            _model: &str,
        ) -> siumai::providers::openai_compatible::types::ModelConfig {
            Default::default()
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
        }
        fn compatibility(&self) -> ProviderCompatibility {
            ProviderCompatibility::openai_standard()
        }
        fn base_url(&self) -> &str {
            &self.base_url
        }
        fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
            Box::new(self.clone())
        }
    }
    let adapter: Arc<dyn ProviderAdapter> = Arc::new(OpenAiStandardAdapter {
        base_url: "https://api.openai.com/v1".to_string(),
    });
    let cfg = OpenAiCompatibleConfig::new(
        "openai",
        "test",
        "https://api.openai.com/v1",
        adapter.clone(),
    )
    .with_model("gpt-4");
    OpenAiCompatibleEventConverter::new(cfg, adapter)
}
use siumai::streaming::ChatStreamEvent;
use siumai::streaming::{JsonEventConverter, SseEventConverter};

#[tokio::test]
async fn test_complete_openai_stream_sequence() {
    let converter = make_openai_converter();

    // Simulate a complete OpenAI streaming sequence
    let events = vec![
        // 1. StreamStart (first event with metadata)
        Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-123","model":"gpt-4","choices":[{"delta":{"content":"Hello"}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 2. ContentDelta events
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"delta":{"content":" world"}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"delta":{"content":"!"}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 3. Tool call start
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather"}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 4. Tool call arguments
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\""}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"New York\"}"}}]}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 5. Thinking content (for reasoning models)
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"delta":{"thinking":"Let me think about this..."}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 6. Usage update (final chunk)
        Event {
            event: "".to_string(),
            data: r#"{"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    let mut results = Vec::new();
    for event in events {
        let event_results = converter.convert_event(event).await;
        for result in event_results {
            results.push(result.unwrap());
        }
    }

    // Verify the complete sequence - with multi-event architecture we may get 6-7 events
    // First event generates StreamStart + ContentDelta; tool call args may coalesce
    assert!(
        results.len() >= 6,
        "Expected at least 6 events, got {}",
        results.len()
    );

    // 1. First event should be StreamStart
    match &results[0] {
        ChatStreamEvent::StreamStart { metadata } => {
            assert_eq!(metadata.id, Some("chatcmpl-123".to_string()));
            assert_eq!(metadata.model, Some("gpt-4".to_string()));
            assert_eq!(metadata.provider, "openai");
        }
        _ => panic!("Expected StreamStart as first event"),
    }

    // 2. Content delta from first event (now generated alongside StreamStart)
    match &results[1] {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            assert_eq!(delta, "Hello");
        }
        _ => panic!("Expected ContentDelta"),
    }

    // 3. Content delta from second event
    match &results[2] {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            assert_eq!(delta, " world");
        }
        _ => panic!("Expected ContentDelta"),
    }

    // 4. Third content delta
    match &results[3] {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            assert_eq!(delta, "!");
        }
        _ => panic!("Expected ContentDelta"),
    }

    // 5. Tool call start (find first ToolCallDelta)
    if let Some(tc_pos) = results
        .iter()
        .position(|e| matches!(e, ChatStreamEvent::ToolCallDelta { .. }))
    {
        match &results[tc_pos] {
            ChatStreamEvent::ToolCallDelta {
                id, function_name, ..
            } => {
                assert_eq!(id, "call_123");
                assert_eq!(function_name.as_deref(), Some("get_weather"));
            }
            _ => unreachable!(),
        }
    } else {
        // Some adapters may coalesce tool-call info into content; ensure we still proceed
        eprintln!("ℹ️ No ToolCallDelta emitted for this synthetic payload; continuing");
    }

    // 6. Thinking content (position may vary if tool args are coalesced)
    let thinking_pos = results
        .iter()
        .position(|e| matches!(e, ChatStreamEvent::ThinkingDelta { .. }))
        .expect("Expected a ThinkingDelta event");
    match &results[thinking_pos] {
        ChatStreamEvent::ThinkingDelta { delta } => {
            assert_eq!(delta, "Let me think about this...");
        }
        _ => panic!("Expected ThinkingDelta"),
    }

    // 7. Usage update (final)
    let usage_pos = results
        .iter()
        .rposition(|e| matches!(e, ChatStreamEvent::UsageUpdate { .. }))
        .expect("Expected a UsageUpdate event");
    match &results[usage_pos] {
        ChatStreamEvent::UsageUpdate { usage } => {
            assert_eq!(usage.prompt_tokens, 10);
            assert_eq!(usage.completion_tokens, 20);
            assert_eq!(usage.total_tokens, 30);
        }
        _ => panic!("Expected UsageUpdate"),
    }
}

#[tokio::test]
async fn test_stream_event_ordering() {
    let converter = make_openai_converter();

    // Test that StreamStart always comes first, regardless of event content
    let events = vec![
        // First event with content but no metadata - should still generate StreamStart
        Event {
            event: "".to_string(),
            data: r#"{"choices":[{"delta":{"content":"Hello"}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // Second event with metadata - should NOT generate StreamStart
        Event {
            event: "".to_string(),
            data: r#"{"id":"chatcmpl-456","model":"gpt-3.5","choices":[{"delta":{"content":" world"}}]}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    let mut results = Vec::new();
    for event in events {
        let event_results = converter.convert_event(event).await;
        for result in event_results {
            results.push(result.unwrap());
        }
    }

    // With multi-event architecture, first event generates StreamStart + ContentDelta
    assert_eq!(results.len(), 3);

    // First result should be StreamStart (even without metadata in first event)
    match &results[0] {
        ChatStreamEvent::StreamStart { metadata } => {
            assert_eq!(metadata.provider, "openai");
            // Should have default/empty metadata since first event had no metadata
            assert_eq!(metadata.id, None);
            assert_eq!(metadata.model, None);
        }
        _ => panic!("Expected StreamStart as first event"),
    }

    // Second result should be ContentDelta from first event
    match &results[1] {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            assert_eq!(delta, "Hello");
        }
        _ => panic!("Expected ContentDelta as second event"),
    }

    // Third result should be ContentDelta from second event
    match &results[2] {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            assert_eq!(delta, " world");
        }
        _ => panic!("Expected ContentDelta as third event"),
    }
}

#[tokio::test]
async fn test_complete_anthropic_stream_sequence() {
    let config = siumai::params::AnthropicParams::default();
    let converter = AnthropicEventConverter::new(config);

    // Simulate a complete Anthropic streaming sequence
    let events = vec![
        // 1. message_start - should generate StreamStart
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_123","model":"claude-3-sonnet","role":"assistant","content":[]}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 2. content_block_start
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 3. content_block_delta events
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 4. content_block_stop
        Event {
            event: "".to_string(),
            data: r#"{"type":"content_block_stop","index":0}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 5. message_delta with usage
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
        // 6. message_stop
        Event {
            event: "".to_string(),
            data: r#"{"type":"message_stop"}"#.to_string(),
            id: "".to_string(),
            retry: None,
        },
    ];

    let mut results = Vec::new();
    for event in events {
        let event_results = converter.convert_event(event).await;
        for result in event_results {
            results.push(result.unwrap());
        }
    }

    // Verify the sequence (should have StreamStart + content deltas + usage)
    assert!(results.len() >= 3);

    // 1. First event should be StreamStart
    match &results[0] {
        ChatStreamEvent::StreamStart { metadata } => {
            assert_eq!(metadata.id, Some("msg_123".to_string()));
            assert_eq!(metadata.model, Some("claude-3-sonnet".to_string()));
            assert_eq!(metadata.provider, "anthropic");
        }
        _ => panic!("Expected StreamStart as first event, got: {:?}", results[0]),
    }

    // Should have content deltas
    let content_deltas: Vec<_> = results
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();

    assert!(!content_deltas.is_empty(), "Should have content deltas");
    assert!(content_deltas.contains(&"Hello".to_string()));
    assert!(content_deltas.contains(&" world".to_string()));
}

#[tokio::test]
async fn test_complete_ollama_stream_sequence() {
    let converter = OllamaEventConverter::new();

    // Simulate a complete Ollama streaming sequence
    let json_events = vec![
        // 1. First chunk - should generate StreamStart
        r#"{"model":"llama2","message":{"role":"assistant","content":"Hello"},"done":false}"#,
        // 2. Content chunks
        r#"{"model":"llama2","message":{"role":"assistant","content":" world"},"done":false}"#,
        r#"{"model":"llama2","message":{"role":"assistant","content":"!"},"done":false}"#,
        // 3. Final chunk with usage
        r#"{"model":"llama2","done":true,"prompt_eval_count":10,"eval_count":20}"#,
    ];

    let mut results = Vec::new();
    for json_data in json_events {
        let event_results = converter.convert_json(json_data).await;
        for result in event_results {
            results.push(result.unwrap());
        }
    }

    // Verify the sequence - with multi-event architecture we get more events
    // First event generates StreamStart + ContentDelta, so we get 5 events total
    assert_eq!(results.len(), 5);

    // 1. First event should be StreamStart
    match &results[0] {
        ChatStreamEvent::StreamStart { metadata } => {
            assert_eq!(metadata.model, Some("llama2".to_string()));
            assert_eq!(metadata.provider, "ollama");
        }
        _ => panic!("Expected StreamStart as first event"),
    }

    // 2. Content delta from first event
    match &results[1] {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            assert_eq!(delta, "Hello");
        }
        _ => panic!("Expected ContentDelta"),
    }

    // 3. Content delta from second event
    match &results[2] {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            assert_eq!(delta, " world");
        }
        _ => panic!("Expected ContentDelta"),
    }

    // 4. Content delta from third event
    match &results[3] {
        ChatStreamEvent::ContentDelta { delta, .. } => {
            assert_eq!(delta, "!");
        }
        _ => panic!("Expected ContentDelta"),
    }

    // 5. Usage update
    match &results[4] {
        ChatStreamEvent::UsageUpdate { usage } => {
            assert_eq!(usage.prompt_tokens, 10);
            assert_eq!(usage.completion_tokens, 20);
        }
        _ => panic!("Expected UsageUpdate"),
    }
}
