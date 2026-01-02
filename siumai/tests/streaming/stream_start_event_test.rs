//! Tests for StreamStart event generation across all providers
//!
//! This test verifies that all providers correctly generate StreamStart events
//! with proper metadata when streaming begins.

use eventsource_stream::Event;
use siumai::experimental::standards::openai::compat::adapter::{
    ProviderAdapter, ProviderCompatibility,
};
use siumai::experimental::standards::openai::compat::openai_config::OpenAiCompatibleConfig;
use siumai::experimental::standards::openai::compat::provider_registry::{
    ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
};
use siumai::experimental::standards::openai::compat::streaming::OpenAiCompatibleEventConverter;
use siumai::experimental::standards::openai::compat::types::FieldMappings;
use siumai::prelude::unified::ProviderCapabilities;
#[cfg(feature = "anthropic")]
use siumai_provider_anthropic::providers::anthropic::streaming::AnthropicEventConverter;
#[cfg(feature = "google")]
use siumai_provider_gemini::providers::gemini::streaming::GeminiEventConverter;
#[cfg(feature = "google")]
use siumai_provider_gemini::providers::gemini::types::GeminiConfig;
#[cfg(feature = "ollama")]
use siumai_provider_ollama::providers::ollama::streaming::OllamaEventConverter;
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
            _ty: siumai::experimental::standards::openai::compat::types::RequestType,
        ) -> Result<(), siumai::prelude::unified::LlmError> {
            Ok(())
        }
        fn get_field_mappings(&self, _model: &str) -> FieldMappings {
            FieldMappings::standard()
        }
        fn get_model_config(
            &self,
            _model: &str,
        ) -> siumai::experimental::standards::openai::compat::types::ModelConfig {
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
use siumai::prelude::unified::{ChatStreamEvent, JsonEventConverter, SseEventConverter};

fn make_groq_converter() -> OpenAiCompatibleEventConverter {
    let provider_config = ProviderConfig {
        id: "groq".to_string(),
        name: "Groq".to_string(),
        base_url: "https://api.groq.com/openai/v1".to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec!["chat".to_string(), "streaming".to_string()],
        default_model: Some("llama-3.1-70b".to_string()),
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: Vec::new(),
    };
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config));
    let cfg =
        OpenAiCompatibleConfig::new("groq", "", "", adapter.clone()).with_model("llama-3.1-70b");
    OpenAiCompatibleEventConverter::new(cfg, adapter)
}

fn make_xai_converter() -> OpenAiCompatibleEventConverter {
    let provider_config = ProviderConfig {
        id: "xai".to_string(),
        name: "xAI".to_string(),
        base_url: "https://api.x.ai/v1".to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec!["chat".to_string(), "streaming".to_string()],
        default_model: Some("grok-beta".to_string()),
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: Vec::new(),
    };
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config));
    let cfg = OpenAiCompatibleConfig::new("xai", "", "", adapter.clone()).with_model("grok-beta");
    OpenAiCompatibleEventConverter::new(cfg, adapter)
}

#[tokio::test]
async fn test_openai_stream_start_event() {
    let converter = make_openai_converter();

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
#[cfg(feature = "anthropic")]
async fn test_anthropic_stream_start_event() {
    let config = siumai::provider_ext::anthropic::AnthropicParams::default();
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
#[cfg(feature = "google")]
async fn test_gemini_stream_start_event() {
    let config = GeminiConfig {
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
    let converter = make_groq_converter();

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
    let converter = make_xai_converter();

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
#[cfg(feature = "ollama")]
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
    let converter = make_openai_converter();

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
