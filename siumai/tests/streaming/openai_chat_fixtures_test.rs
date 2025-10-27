//! OpenAI Chat Completions streaming fixtures tests

use siumai::providers::openai_compatible::adapter::ProviderAdapter;
use siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter;
use siumai::providers::openai_compatible::types::FieldMappings;
use siumai::streaming::ChatStreamEvent;
use siumai::traits::ProviderCapabilities;
use std::sync::Arc;

#[path = "../support/stream_fixture.rs"]
mod support;

fn make_openai_converter() -> OpenAiCompatibleEventConverter {
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
        fn compatibility(
            &self,
        ) -> siumai::providers::openai_compatible::adapter::ProviderCompatibility {
            siumai::providers::openai_compatible::adapter::ProviderCompatibility::openai_standard()
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
async fn chat_completions_role_content_finish_reason_fixture() {
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/chat-completions/role_content_finish_reason.sse",
    )
    .expect("load fixture");

    let converter = make_openai_converter();
    let events = support::collect_sse_events(bytes, converter).await;

    // Assert: at least one ContentDelta and a final StreamEnd with finish_reason
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ChatStreamEvent::ContentDelta { .. })),
        "expect content delta"
    );

    let end = events
        .into_iter()
        .find_map(|e| match e {
            ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expect stream end");
    assert_eq!(end.finish_reason, Some(siumai::types::FinishReason::Stop));
}

#[tokio::test]
async fn chat_completions_tool_calls_arguments_usage_fixture() {
    // This fixture simulates streaming tool call deltas with split arguments,
    // plus a usage update and stream termination.
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/chat-completions/tool_calls_arguments_usage.sse",
    )
    .expect("load fixture");

    let converter = make_openai_converter();
    let events = support::collect_sse_events(bytes, converter).await;

    // Expect: tool call with id and name present
    let saw_tool_call = events.iter().any(|e| {
        matches!(
            e,
            ChatStreamEvent::ToolCallDelta { id, function_name, .. }
            if id == "call_1" && function_name.as_deref() == Some("lookup")
        )
    });
    // Expect: arguments delta across chunks combine to the expected JSON
    let combined_args = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ToolCallDelta {
                id,
                arguments_delta: Some(a),
                ..
            } if id == "call_1" => Some(a.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    // Expect: usage update
    let saw_usage = events.iter().any(|e| {
        matches!(
            e,
            ChatStreamEvent::UsageUpdate { usage } if usage.total_tokens == 30
        )
    });
    // Expect: stream end with finish_reason
    let end = events
        .into_iter()
        .find_map(|e| match e {
            ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expect stream end");

    assert!(
        saw_tool_call,
        "expect initial tool call delta with id and name"
    );
    assert_eq!(combined_args, "{\"q\":\"rust\"}");
    assert!(saw_usage, "expect usage update event");
    assert_eq!(end.finish_reason, Some(siumai::types::FinishReason::Stop));
}

#[tokio::test]
async fn chat_completions_multiple_tool_calls_at_least_one_fixture() {
    // This fixture contains two parallel tool calls; our current converter extracts the first.
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/chat-completions/multiple_tool_calls.sse",
    )
    .expect("load fixture");

    let converter = make_openai_converter();
    let events = support::collect_sse_events(bytes, converter).await;
    // Assert at least one tool call is captured
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ChatStreamEvent::ToolCallDelta { .. })),
        "expect at least one tool call delta"
    );
}

#[tokio::test]
async fn chat_completions_multiple_tool_calls_full_aggregation_fixture() {
    // Pending: once we support aggregating multiple tool calls per chunk
    let bytes = support::load_sse_fixture_as_bytes(
        "tests/fixtures/openai/chat-completions/multiple_tool_calls.sse",
    )
    .expect("load fixture");
    let converter = make_openai_converter();
    let events = support::collect_sse_events(bytes, converter).await;
    let tool_calls = events
        .iter()
        .filter(|e| matches!(e, ChatStreamEvent::ToolCallDelta { .. }))
        .count();
    assert!(tool_calls >= 2, "expect both tool calls captured");
}
