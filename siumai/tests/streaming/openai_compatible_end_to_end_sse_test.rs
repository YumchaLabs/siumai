//! End-to-end SSE streaming test for OpenAI-compatible converter

use eventsource_stream::Event;
use futures_util::StreamExt;
use siumai::error::LlmError;
use siumai::providers::openai_compatible::adapter::{ProviderAdapter, ProviderCompatibility};
use siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter;
use siumai::providers::openai_compatible::types::FieldMappings;
use siumai::streaming::ChatStreamEvent;
use siumai::streaming::{SseEventConverter, SseStreamExt};
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
async fn end_to_end_sse_multi_event_flow() {
    let converter = make_converter();

    // Build SSE byte stream: multiple data: lines
    let sse_chunks = vec![
        format!(
            "data: {}\n\n",
            r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","created":1731234567,"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#
        ),
        format!(
            "data: {}\n\n",
            r#"{"choices":[{"index":0,"delta":{"thinking":"Reasoning..."}}]}"#
        ),
        format!(
            "data: {}\n\n",
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","function":{"name":"lookup","arguments":"{\"q\":\"rust\"}"}}]}}]}"#
        ),
        format!(
            "data: {}\n\n",
            r#"{"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}"#
        ),
        "data: [DONE]\n\n".to_string(),
    ];

    // Convert into a stream of bytes
    let bytes: Vec<Result<Vec<u8>, std::io::Error>> =
        sse_chunks.into_iter().map(|s| Ok(s.into_bytes())).collect();
    let byte_stream = futures_util::stream::iter(bytes);
    let mut sse_stream = byte_stream.into_sse_stream();

    // Collect ChatStreamEvents in order
    let mut events: Vec<ChatStreamEvent> = Vec::new();
    while let Some(item) = sse_stream.next().await {
        let event: Event = item.expect("valid event");
        if event.data.trim() == "[DONE]" {
            if let Some(end) = converter.handle_stream_end() {
                events.push(end.expect("stream end ok"));
            }
            break;
        }
        let converted = converter.convert_event(event).await;
        for e in converted {
            events.push(e.expect("ok"));
        }
    }

    // Validate sequence has key events
    assert!(
        matches!(events.first(), Some(ChatStreamEvent::StreamStart { .. })),
        "first should be StreamStart"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ChatStreamEvent::ContentDelta { delta, .. } if delta == "Hello")),
        "should contain content delta"
    );
    assert!(
        events.iter().any(
            |e| matches!(e, ChatStreamEvent::ThinkingDelta { delta } if delta == "Reasoning...")
        ),
        "should contain thinking delta"
    );
    assert!(events.iter().any(|e| matches!(e, ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, .. }
        if id == "call_1" && function_name.as_deref() == Some("lookup") && arguments_delta.as_deref() == Some("{\"q\":\"rust\"}")
    )), "should contain tool call delta");
    assert!(
        events.iter().any(
            |e| matches!(e, ChatStreamEvent::UsageUpdate { usage } if usage.total_tokens == 12)
        ),
        "should contain usage update"
    );
    assert!(
        matches!(events.last(), Some(ChatStreamEvent::StreamEnd { .. })),
        "last should be StreamEnd"
    );
}
