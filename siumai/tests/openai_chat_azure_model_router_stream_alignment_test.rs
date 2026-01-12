#![cfg(feature = "openai")]

//! Alignment test for Vercel `azure-model-router.1` Chat Completions streaming fixture.

use eventsource_stream::Event;
use siumai::experimental::standards::openai::compat::adapter::{
    ProviderAdapter, ProviderCompatibility,
};
use siumai::experimental::standards::openai::compat::openai_config::OpenAiCompatibleConfig;
use siumai::experimental::standards::openai::compat::streaming::OpenAiCompatibleEventConverter;
use siumai::experimental::standards::openai::compat::types::{
    FieldMappings, ModelConfig, RequestType,
};
use siumai::prelude::unified::{
    ChatStreamEvent, LlmError, ProviderCapabilities, SseEventConverter,
};
use std::path::Path;
use std::sync::Arc;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("chat-completions")
        .join("azure-model-router.1.chunks.txt")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn make_openai_converter(model: &str) -> OpenAiCompatibleEventConverter {
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
            _ty: RequestType,
        ) -> Result<(), LlmError> {
            Ok(())
        }
        fn get_field_mappings(&self, _model: &str) -> FieldMappings {
            FieldMappings::standard()
        }
        fn get_model_config(&self, _model: &str) -> ModelConfig {
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
    .with_model(model);
    OpenAiCompatibleEventConverter::new(cfg, adapter)
}

#[tokio::test]
async fn openai_chat_azure_model_router_stream_uses_request_model_when_chunk_model_empty() {
    let path = fixture_path();
    assert!(path.exists(), "fixture missing: {:?}", path);
    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let converter = make_openai_converter("test-azure-model-router");

    let mut events: Vec<ChatStreamEvent> = Vec::new();
    for (i, line) in lines.into_iter().enumerate() {
        let ev = Event {
            event: "".to_string(),
            data: line,
            id: i.to_string(),
            retry: None,
        };

        let out = converter.convert_event(ev).await;
        for item in out {
            events.push(item.expect("convert chunk"));
        }
    }

    let start = events
        .iter()
        .find_map(|e| match e {
            ChatStreamEvent::StreamStart { metadata } => Some(metadata),
            _ => None,
        })
        .expect("expected StreamStart");

    // Vercel parity: do not surface empty model ids from the initial router chunk.
    assert_eq!(start.model.as_deref(), Some("test-azure-model-router"));

    let text: String = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        text.contains("Capital of Denmark."),
        "unexpected text: {text}"
    );

    let ended = events
        .iter()
        .any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. }));
    assert!(ended, "expected StreamEnd");
}
