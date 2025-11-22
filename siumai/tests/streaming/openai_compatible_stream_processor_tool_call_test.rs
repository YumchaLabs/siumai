//! OpenAI-compatible + StreamProcessor 工具调用累积测试
//!
//! 验证 OpenAiCompatibleEventConverter 产生的 ToolCallDelta 事件
//! 在 StreamProcessor 中能够正确累积为完整的 JSON arguments，
//! 行为与 Vercel AI SDK 的 tool-input-delta / tool-call 等价。

use eventsource_stream::Event;
use siumai::error::LlmError;
use siumai::providers::openai_compatible::adapter::{
    ProviderAdapter, ProviderCompatibility,
};
use siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use siumai::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter;
use siumai::providers::openai_compatible::types::FieldMappings;
use siumai::streaming::{ChatStreamEvent, StreamProcessor};
use siumai::traits::ProviderCapabilities;
use siumai::utils::streaming::SseEventConverter;
use std::sync::Arc;

fn make_converter() -> OpenAiCompatibleEventConverter {
    #[derive(Debug, Clone)]
    struct Adapter;
    impl ProviderAdapter for Adapter {
        fn provider_id(&self) -> &'static str {
            "openai"
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
            ProviderCapabilities::new().with_chat().with_streaming().with_tools()
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
async fn openai_compatible_tool_call_arguments_are_accumulated() {
    let converter = make_converter();
    let mut processor = StreamProcessor::new();

    // 1) 第一块：启动 tool call，提供 name 和 arguments 的首段
    let ev1 = Event {
        event: "".to_string(),
        data: r#"{
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"query\":\"rust\","
                        }
                    }]
                }
            }]
        }"#
        .to_string(),
        id: "".to_string(),
        retry: None,
    };
    for res in converter.convert_event(ev1).await {
        let event = res.expect("event ok");
        processor.process_event(event);
    }

    // 2) 第二块：追加 arguments 的剩余部分
    let ev2 = Event {
        event: "".to_string(),
        data: r#"{
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "arguments": "\"count\":42}"
                        }
                    }]
                }
            }]
        }"#
        .to_string(),
        id: "".to_string(),
        retry: None,
    };
    for res in converter.convert_event(ev2).await {
        let event = res.expect("event ok");
        processor.process_event(event);
    }

    // 结束流（不依赖 converter 的 StreamEnd，这里直接构造最终响应）
    let final_resp =
        processor.build_final_response_with_finish_reason(Some(siumai::types::FinishReason::Stop));

    assert!(
        final_resp.has_tool_calls(),
        "final response should contain tool calls"
    );

    let calls = final_resp.tool_calls();
    assert_eq!(calls.len(), 1, "expected exactly one tool call");

    let call = calls[0]
        .as_tool_call()
        .expect("content part should be a tool call");
    assert_eq!(call.tool_call_id, "call_1");
    assert_eq!(call.tool_name, "lookup");

    // arguments 应该是完整 JSON，等价于 {"query":"rust","count":42}
    let args = call.arguments;
    let obj = args
        .as_object()
        .expect("tool call arguments should be a JSON object");
    assert_eq!(obj.get("query").and_then(|v| v.as_str()), Some("rust"));
    assert_eq!(obj.get("count").and_then(|v| v.as_i64()), Some(42));
}

