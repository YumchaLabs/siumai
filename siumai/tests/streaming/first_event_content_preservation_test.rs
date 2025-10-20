use eventsource_stream::Event;
use siumai::providers::gemini::streaming::GeminiEventConverter;
use siumai::providers::gemini::types::GeminiConfig;
use siumai::providers::ollama::streaming::OllamaEventConverter;
use siumai::providers::openai_compatible::adapter::{ProviderAdapter, ProviderCompatibility};
use siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
/// Critical test to verify that first event content is preserved in streaming responses
/// This test validates the multi-event emission architecture that prevents content loss
///
/// Background: Before the refactor, the first SSE event content was lost because
/// converters would return StreamStart instead of preserving the actual content.
/// The new architecture allows multiple events to be emitted from a single SSE event.
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
        fn provider_id(&self) -> &'static str {
            "openai"
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
use siumai::stream::ChatStreamEvent;
use siumai::utils::streaming::{JsonEventConverter, SseEventConverter};

#[tokio::test]
async fn test_openai_first_event_with_content_preservation() {
    let converter = make_openai_converter();

    // Simulate first SSE event that contains actual content
    // This is the critical test case that was failing before the refactor
    let first_event_data = r#"{
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "Hello"
            },
            "finish_reason": null
        }]
    }"#;

    let event = Event {
        event: "".to_string(),
        data: first_event_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let results = converter.convert_event(event).await;

    // Critical assertions for multi-event architecture
    assert!(
        !results.is_empty(),
        "First event should generate at least one ChatStreamEvent"
    );
    assert!(
        results.len() >= 2,
        "First event with content should generate StreamStart + ContentDelta"
    );

    // Verify StreamStart is generated
    let stream_start = results
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));
    assert!(
        stream_start.is_some(),
        "StreamStart event must be generated for first event"
    );

    if let Some(Ok(ChatStreamEvent::StreamStart { metadata })) = stream_start {
        assert_eq!(metadata.id, Some("chatcmpl-123".to_string()));
        assert_eq!(metadata.model, Some("gpt-4".to_string()));
        assert_eq!(metadata.provider, "openai");
    }

    // Verify ContentDelta is generated and content is preserved
    let content_delta = results
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::ContentDelta { .. })));
    assert!(
        content_delta.is_some(),
        "ContentDelta event must be generated to preserve first event content"
    );

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, index })) = content_delta {
        assert_eq!(
            delta, "Hello",
            "First event content 'Hello' must be preserved"
        );
        assert_eq!(index, &Some(0));
    }

    println!("✅ OpenAI first event content preservation test passed");
    println!("Generated events: {:?}", results);
}

#[tokio::test]
async fn test_openai_subsequent_events_single_emission() {
    let converter = make_openai_converter();

    // First event to trigger StreamStart
    let first_event_data = r#"{
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "Hello"
            },
            "finish_reason": null
        }]
    }"#;

    let first_event = Event {
        event: "".to_string(),
        data: first_event_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    // Process first event
    let first_results = converter.convert_event(first_event).await;
    assert!(
        first_results.len() >= 2,
        "First event should generate multiple events"
    );

    // Second event should only generate ContentDelta (no more StreamStart)
    let second_event_data = r#"{
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "delta": {
                "content": " world"
            },
            "finish_reason": null
        }]
    }"#;

    let second_event = Event {
        event: "".to_string(),
        data: second_event_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let second_results = converter.convert_event(second_event).await;

    // Subsequent events should only generate single ContentDelta
    assert_eq!(
        second_results.len(),
        1,
        "Subsequent events should generate exactly one event"
    );

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = second_results.first() {
        assert_eq!(delta, " world");
    } else {
        panic!("Expected ContentDelta for subsequent event");
    }

    // Verify no more StreamStart events
    let has_stream_start = second_results
        .iter()
        .any(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));
    assert!(
        !has_stream_start,
        "Subsequent events should not generate StreamStart"
    );

    println!("✅ OpenAI subsequent events test passed");
}

#[tokio::test]
async fn test_gemini_first_event_with_content_preservation() {
    let config = GeminiConfig::default();
    let converter = GeminiEventConverter::new(config);

    // Simulate Gemini first event with content
    let first_event_data = r#"{
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "Hello from Gemini"
                }]
            }
        }]
    }"#;

    let event = Event {
        event: "".to_string(),
        data: first_event_data.to_string(),
        id: "".to_string(),
        retry: None,
    };

    let results = converter.convert_event(event).await;

    assert!(
        !results.is_empty(),
        "Gemini first event should generate events"
    );

    // Check for StreamStart
    let has_stream_start = results
        .iter()
        .any(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));
    assert!(
        has_stream_start,
        "Gemini should generate StreamStart for first event"
    );

    // Check for ContentDelta with preserved content
    let content_event = results
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::ContentDelta { .. })));
    assert!(
        content_event.is_some(),
        "Gemini should preserve first event content"
    );

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = content_event {
        assert_eq!(
            delta, "Hello from Gemini",
            "Gemini first event content must be preserved"
        );
    }

    println!("✅ Gemini first event content preservation test passed");
}

#[tokio::test]
async fn test_ollama_first_event_with_content_preservation() {
    let converter = OllamaEventConverter::new();

    // Simulate Ollama first event with content
    let first_event_data = r#"{
        "model": "llama2",
        "created_at": "2023-08-04T08:52:19.385406455-07:00",
        "message": {
            "role": "assistant",
            "content": "Hello from Ollama"
        },
        "done": false
    }"#;

    let results = converter.convert_json(first_event_data).await;

    assert!(
        !results.is_empty(),
        "Ollama first event should generate events"
    );

    // Check for StreamStart
    let has_stream_start = results
        .iter()
        .any(|event| matches!(event, Ok(ChatStreamEvent::StreamStart { .. })));
    assert!(
        has_stream_start,
        "Ollama should generate StreamStart for first event"
    );

    // Check for ContentDelta with preserved content
    let content_event = results
        .iter()
        .find(|event| matches!(event, Ok(ChatStreamEvent::ContentDelta { .. })));
    assert!(
        content_event.is_some(),
        "Ollama should preserve first event content"
    );

    if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = content_event {
        assert_eq!(
            delta, "Hello from Ollama",
            "Ollama first event content must be preserved"
        );
    }

    println!("✅ Ollama first event content preservation test passed");
}

#[tokio::test]
async fn test_complete_streaming_sequence_content_preservation() {
    let converter = make_openai_converter();

    // Simulate a complete streaming sequence where first event has content
    let events_data = [
        // First event with content - this is the critical test case
        r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}"#,
        // Second event
        r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" answer"},"finish_reason":null}]}"#,
        // Third event
        r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}"#,
        // Final event
        r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" 42."},"finish_reason":"stop"}]}"#,
    ];

    let mut all_content = String::new();
    let mut stream_start_count = 0;
    let mut content_delta_count = 0;

    for (i, event_data) in events_data.iter().enumerate() {
        let event = Event {
            event: "".to_string(),
            data: event_data.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let results = converter.convert_event(event).await;

        for result in results {
            match result.unwrap() {
                ChatStreamEvent::StreamStart { .. } => {
                    stream_start_count += 1;
                    println!("Event {}: StreamStart", i);
                }
                ChatStreamEvent::ContentDelta { delta, .. } => {
                    content_delta_count += 1;
                    all_content.push_str(&delta);
                    println!("Event {}: ContentDelta '{}'", i, delta);
                }
                _ => {}
            }
        }
    }

    // Verify complete content is preserved
    assert_eq!(
        all_content, "The answer is 42.",
        "Complete content must be preserved including first event"
    );
    assert_eq!(stream_start_count, 1, "Should have exactly one StreamStart");
    assert_eq!(
        content_delta_count, 4,
        "Should have four ContentDelta events"
    );

    println!("✅ Complete streaming sequence test passed");
    println!("Final content: '{}'", all_content);
}
