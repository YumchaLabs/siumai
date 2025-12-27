//! OpenAI-compatible streaming multi-event tests

use eventsource_stream::Event;
use siumai::error::LlmError;
use siumai::providers::openai_compatible::adapter::{ProviderAdapter, ProviderCompatibility};
use siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter;
use siumai::providers::openai_compatible::types::FieldMappings;
use siumai::streaming::ChatStreamEvent;
use siumai::streaming::SseEventConverter;
use siumai::traits::ProviderCapabilities;
use std::sync::Arc;

fn make_converter() -> OpenAiCompatibleEventConverter {
    #[derive(Debug, Clone)]
    struct Adapter;
    impl ProviderAdapter for Adapter {
        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("openai")
        }
        fn transform_request_params(
            &self,
            _params: &mut serde_json::Value,
            _model: &str,
            _ty: siumai::providers::openai_compatible::types::RequestType,
        ) -> Result<(), LlmError> {
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
            "https://api.openai.com/v1"
        }
        fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
            Box::new(self.clone())
        }
    }

    let adapter: Arc<dyn ProviderAdapter> = Arc::new(Adapter);
    let cfg = OpenAiCompatibleConfig::new(
        "openai",
        "sk-test",
        "https://api.openai.com/v1",
        adapter.clone(),
    )
    .with_model("gpt-4o-mini");
    OpenAiCompatibleEventConverter::new(cfg, adapter)
}

#[tokio::test]
async fn test_multi_event_sequence() {
    let converter = make_converter();

    // 1) First chunk with content + metadata -> StreamStart + ContentDelta
    let event1 = Event {
        event: "".to_string(),
        data: r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","created": 1731234567,
                  "choices":[{"index":0,"delta":{"content":"Hello"}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r1 = converter.convert_event(event1).await;
    assert!(
        r1.iter()
            .any(|e| matches!(e, Ok(ChatStreamEvent::StreamStart { .. })))
    );
    assert!(
        r1.iter().any(
            |e| matches!(e, Ok(ChatStreamEvent::ContentDelta{ delta, .. }) if delta == "Hello")
        )
    );

    // 2) Thinking delta
    let event2 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"thinking":"Reasoning..."}}]}"#.to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r2 = converter.convert_event(event2).await;
    assert!(r2.iter().any(
        |e| matches!(e, Ok(ChatStreamEvent::ThinkingDelta{ delta }) if delta == "Reasoning...")
    ));

    // 3) Tool call delta (function)
    let event3 = Event {
        event: "".to_string(),
        data: r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1",
                      "function":{"name":"lookup","arguments":"{\"q\":\"rust\"}"}}]}}]}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r3 = converter.convert_event(event3).await;
    assert!(r3.iter().any(|e| matches!(e,
        Ok(ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, .. })
        if id == "call_1" && function_name.as_deref() == Some("lookup") && arguments_delta.as_deref() == Some("{\"q\":\"rust\"}")
    )));

    // 4) Usage update
    let event4 = Event {
        event: "".to_string(),
        data: r#"{"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}"#
            .to_string(),
        id: "".to_string(),
        retry: None,
    };
    let r4 = converter.convert_event(event4).await;
    assert!(r4.iter().any(|e| matches!(e,
        Ok(ChatStreamEvent::UsageUpdate { usage }) if usage.prompt_tokens == 5 && usage.completion_tokens == 7 && usage.total_tokens == 12
    )));

    // 5) End of stream ([DONE]) -> StreamEnd
    let end = converter.handle_stream_end().expect("end event");
    match end {
        Ok(ChatStreamEvent::StreamEnd { response }) => {
            assert!(response.finish_reason.is_some());
        }
        other => panic!("Expected StreamEnd, got: {:?}", other),
    }
}
