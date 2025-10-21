//! OpenAI-compatible SSE test: finish_reason without [DONE]

use eventsource_stream::Event;
use futures_util::StreamExt;
use siumai::providers::openai_compatible::adapter::{ProviderAdapter, ProviderCompatibility};
use siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter;
use siumai::providers::openai_compatible::types::FieldMappings;
use siumai::streaming::ChatStreamEvent;
use siumai::traits::ProviderCapabilities;
use siumai::utils::sse_stream::SseStreamExt;
use siumai::utils::streaming::SseEventConverter;
use std::sync::Arc;

fn make_converter() -> OpenAiCompatibleEventConverter {
    #[derive(Debug, Clone)]
    struct Adapter;
    impl ProviderAdapter for Adapter {
        fn provider_id(&self) -> &'static str { "openai" }
        fn transform_request_params(
            &self,
            _params: &mut serde_json::Value,
            _model: &str,
            _ty: siumai::providers::openai_compatible::types::RequestType,
        ) -> Result<(), siumai::error::LlmError> { Ok(()) }
        fn get_field_mappings(&self, _model: &str) -> FieldMappings { FieldMappings::standard() }
        fn get_model_config(
            &self,
            _model: &str,
        ) -> siumai::providers::openai_compatible::types::ModelConfig { Default::default() }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat().with_streaming().with_tools()
        }
        fn compatibility(&self) -> ProviderCompatibility { ProviderCompatibility::openai_standard() }
        fn base_url(&self) -> &str { "https://api.openai.com/v1" }
        fn clone_adapter(&self) -> Box<dyn ProviderAdapter> { Box::new(self.clone()) }
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
async fn finish_reason_without_done_emits_stream_end() {
    let converter = make_converter();
    // Simulate standard OpenAI chat.completions stream without [DONE]
    let sse_chunks = vec![
        // First delta: role only (common in OpenAI streams)
        format!(
            "data: {}\n\n",
            r#"{"id":"chatcmpl-1","model":"gpt-4o-mini","created":1731234567,"choices":[{"index":0,"delta":{"role":"assistant"}}]}"#
        ),
        // Content delta
        format!(
            "data: {}\n\n",
            r#"{"choices":[{"index":0,"delta":{"content":"1\n2\n"}}]}"#
        ),
        // Final chunk with finish_reason but no [DONE]
        format!(
            "data: {}\n\n",
            r#"{"choices":[{"index":0,"finish_reason":"stop"}]}"#
        ),
    ];

    let bytes: Vec<Result<Vec<u8>, std::io::Error>> = sse_chunks
        .into_iter()
        .map(|s| Ok(s.into_bytes()))
        .collect();
    let byte_stream = futures_util::stream::iter(bytes);
    let mut sse_stream = byte_stream.into_sse_stream();

    let mut saw_content = false;
    let mut saw_end = false;
    while let Some(item) = sse_stream.next().await {
        let event: Event = item.expect("valid event");
        let converted = converter.convert_event(event).await;
        for e in converted {
            match e.expect("ok") {
                ChatStreamEvent::ContentDelta { delta, .. } => {
                    assert_eq!(delta, "1\n2\n");
                    saw_content = true;
                }
                ChatStreamEvent::StreamEnd { .. } => saw_end = true,
                _ => {}
            }
        }
    }
    assert!(saw_content, "should see content delta");
    assert!(saw_end, "should emit StreamEnd on finish_reason");
}

