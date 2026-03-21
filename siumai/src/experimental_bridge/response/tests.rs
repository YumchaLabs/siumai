use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use siumai_core::bridge::{
    BridgeLossAction, BridgeLossPolicy, BridgeMode, BridgeOptions, BridgePrimitiveContext,
    BridgePrimitiveRemapper, BridgeTarget, RequestBridgeContext, ResponseBridgeContext,
    ResponseBridgeHook, StreamBridgeContext,
};
use siumai_core::encoding::JsonEncodeOptions;
use siumai_core::types::{ChatResponse, ContentPart, FinishReason, MessageContent, Usage};

#[cfg(feature = "anthropic")]
use super::bridge_chat_response_to_anthropic_messages_json_value;
#[cfg(feature = "openai")]
use super::{
    bridge_chat_response_to_openai_chat_completions_json_value,
    bridge_chat_response_to_openai_chat_completions_json_value_with_options,
    bridge_chat_response_to_openai_responses_json_value,
};

struct PrefixRemapper;

impl BridgePrimitiveRemapper for PrefixRemapper {
    fn remap_tool_name(&self, _ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        Some(format!("gw_{name}"))
    }
}

struct ContinueLossyPolicy;

impl BridgeLossPolicy for ContinueLossyPolicy {
    fn request_action(
        &self,
        _ctx: &RequestBridgeContext,
        _report: &siumai_core::bridge::BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }

    fn response_action(
        &self,
        _ctx: &ResponseBridgeContext,
        _report: &siumai_core::bridge::BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }

    fn stream_action(
        &self,
        _ctx: &StreamBridgeContext,
        _report: &siumai_core::bridge::BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }
}

struct RedactResponseHook;

impl ResponseBridgeHook for RedactResponseHook {
    fn transform_response(
        &self,
        ctx: &ResponseBridgeContext,
        response: &mut ChatResponse,
        report: &mut siumai_core::bridge::BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        assert_eq!(ctx.route_label.as_deref(), Some("tests.response.hook"));
        response.content = MessageContent::Text("[hooked]".to_string());
        report.record_lossy_field(
            "response.content",
            "response hook replaced content before target serialization",
        );
        Ok(())
    }
}

#[cfg(feature = "openai")]
#[test]
fn strict_openai_responses_bridge_preserves_reasoning_blocks() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::Reasoning {
            text: "hidden chain of thought".to_string(),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                json!({
                    "itemId": "rs_1",
                    "reasoningEncryptedContent": "enc_1"
                }),
            )])),
        },
        ContentPart::text("final answer"),
    ]));
    response.id = Some("resp_1".to_string());
    response.model = Some("gpt-4.1-mini".to_string());
    response.provider_metadata = Some(HashMap::from([(
        "openai".to_string(),
        HashMap::from([("itemId".to_string(), json!("msg_1"))]),
    )]));

    let bridged = bridge_chat_response_to_openai_responses_json_value(
        &response,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::Strict,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_exact());

    let value = bridged.value.expect("json body");
    assert_eq!(value["output"][0]["id"], json!("msg_1"));
    assert_eq!(value["output"][1]["type"], json!("reasoning"));
    assert_eq!(value["output"][1]["id"], json!("rs_1"));
    assert_eq!(value["output"][1]["encrypted_content"], json!("enc_1"));
}

#[cfg(feature = "openai")]
#[test]
fn openai_chat_response_bridge_preserves_native_top_level_fields() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("visible answer"),
        ContentPart::tool_call("call_1", "web_search", json!({ "q": "rust" }), None),
    ]));
    response.id = Some("chatcmpl_1".to_string());
    response.model = Some("gpt-4.1-mini".to_string());
    response.system_fingerprint = Some("fp_123".to_string());
    response.service_tier = Some("priority".to_string());
    response.finish_reason = Some(FinishReason::ToolCalls);

    let bridged = bridge_chat_response_to_openai_chat_completions_json_value(
        &response,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_exact());

    let value = bridged.value.expect("json body");
    assert_eq!(value["id"], json!("chatcmpl_1"));
    assert_eq!(value["system_fingerprint"], json!("fp_123"));
    assert_eq!(value["service_tier"], json!("priority"));
    assert_eq!(value["choices"][0]["finish_reason"], json!("tool_calls"));
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        json!("web_search")
    );
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_response_bridge_reports_usage_and_metadata_loss() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("searching"),
        ContentPart::tool_call("call_1", "web_search", json!({ "q": "rust" }), Some(true)),
    ]));
    response.id = Some("msg_1".to_string());
    response.model = Some("claude-sonnet-4-5".to_string());
    response.finish_reason = Some(FinishReason::ToolCalls);
    response.usage = Some(
        Usage::builder()
            .prompt_tokens(10)
            .completion_tokens(5)
            .total_tokens(15)
            .with_cached_tokens(3)
            .with_reasoning_tokens(2)
            .build(),
    );
    response.provider_metadata = Some(HashMap::from([(
        "openai".to_string(),
        HashMap::from([("responseId".to_string(), json!("resp_1"))]),
    )]));

    let bridged = bridge_chat_response_to_anthropic_messages_json_value(
        &response,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());
    assert!(
        bridged
            .report
            .lossy_fields
            .iter()
            .any(|field| field == "usage.prompt_tokens_details"),
        "expected prompt token detail loss"
    );
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "provider_metadata.openai"),
        "expected provider metadata drop"
    );

    let value = bridged.value.expect("json body");
    assert_eq!(value["type"], json!("message"));
    assert_eq!(value["stop_reason"], json!("tool_use"));
    assert_eq!(value["usage"]["input_tokens"], json!(10));
    assert_eq!(value["content"][1]["type"], json!("tool_use"));
    assert_eq!(value["content"][1]["id"], json!("call_1"));
}

#[cfg(feature = "anthropic")]
#[test]
fn strict_anthropic_response_bridge_rejects_usage_detail_loss() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("searching"),
        ContentPart::tool_call("call_1", "web_search", json!({ "q": "rust" }), Some(true)),
    ]));
    response.id = Some("msg_1".to_string());
    response.model = Some("claude-sonnet-4-5".to_string());
    response.finish_reason = Some(FinishReason::ToolCalls);
    response.usage = Some(
        Usage::builder()
            .prompt_tokens(10)
            .completion_tokens(5)
            .total_tokens(15)
            .with_cached_tokens(3)
            .with_reasoning_tokens(2)
            .build(),
    );
    response.provider_metadata = Some(HashMap::from([(
        "openai".to_string(),
        HashMap::from([("responseId".to_string(), json!("resp_1"))]),
    )]));

    let bridged = bridge_chat_response_to_anthropic_messages_json_value(
        &response,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::Strict,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(bridged.is_rejected());
    assert!(bridged.report.is_rejected());
    assert!(
        bridged
            .report
            .lossy_fields
            .iter()
            .any(|field| field == "usage.prompt_tokens_details"),
        "expected prompt token detail loss"
    );
}

#[cfg(feature = "anthropic")]
#[test]
fn custom_response_loss_policy_can_allow_lossy_bridge_in_strict_mode() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("searching"),
        ContentPart::tool_call("call_1", "web_search", json!({ "q": "rust" }), Some(true)),
    ]));
    response.id = Some("msg_1".to_string());
    response.model = Some("claude-sonnet-4-5".to_string());
    response.finish_reason = Some(FinishReason::ToolCalls);
    response.usage = Some(
        Usage::builder()
            .prompt_tokens(10)
            .completion_tokens(5)
            .total_tokens(15)
            .with_cached_tokens(3)
            .with_reasoning_tokens(2)
            .build(),
    );
    response.provider_metadata = Some(HashMap::from([(
        "openai".to_string(),
        HashMap::from([("responseId".to_string(), json!("resp_1"))]),
    )]));

    let bridged = super::bridge_chat_response_to_anthropic_messages_json_value_with_options(
        &response,
        Some(BridgeTarget::OpenAiResponses),
        BridgeOptions::new(BridgeMode::Strict).with_loss_policy(Arc::new(ContinueLossyPolicy)),
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());
    assert!(bridged.value.is_some());
}

#[cfg(feature = "anthropic")]
#[test]
fn strict_anthropic_response_bridge_preserves_thinking_replay_fields() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("visible answer"),
        ContentPart::reasoning("internal thinking"),
        ContentPart::ToolCall {
            tool_call_id: "toolu_1".to_string(),
            tool_name: "weather".to_string(),
            arguments: json!({ "city": "Tokyo" }),
            provider_executed: None,
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                json!({
                    "caller": {
                        "type": "code_execution_20250825",
                        "tool_id": "srvtoolu_1"
                    }
                }),
            )])),
        },
    ]));
    response.id = Some("msg_1".to_string());
    response.model = Some("claude-sonnet-4-5".to_string());
    response.finish_reason = Some(FinishReason::StopSequence);
    response.service_tier = Some("priority".to_string());
    response.provider_metadata = Some(HashMap::from([(
        "anthropic".to_string(),
        HashMap::from([
            ("thinking_signature".to_string(), json!("sig_1")),
            ("redacted_thinking_data".to_string(), json!("redacted_123")),
            ("stopSequence".to_string(), json!("</tool>")),
        ]),
    )]));

    let bridged = bridge_chat_response_to_anthropic_messages_json_value(
        &response,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::Strict,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_exact());

    let value = bridged.value.expect("json body");
    assert_eq!(value["stop_sequence"], json!("</tool>"));
    assert_eq!(value["usage"]["service_tier"], json!("priority"));
    assert_eq!(value["content"][1]["type"], json!("thinking"));
    assert_eq!(value["content"][1]["signature"], json!("sig_1"));
    assert_eq!(
        value["content"][2]["caller"]["tool_id"],
        json!("srvtoolu_1")
    );
    assert_eq!(value["content"][3]["type"], json!("redacted_thinking"));
    assert_eq!(value["content"][3]["data"], json!("redacted_123"));
}

#[cfg(feature = "openai")]
#[test]
fn strict_openai_responses_bridge_preserves_provider_executed_custom_tool_items() {
    let response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::ToolCall {
            tool_call_id: "browser_1".to_string(),
            tool_name: "browser_agent".to_string(),
            arguments: json!({
                "url": "https://example.com"
            }),
            provider_executed: Some(true),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                json!({
                    "itemId": "ct_1"
                }),
            )])),
        },
        ContentPart::ToolResult {
            tool_call_id: "browser_1".to_string(),
            tool_name: "browser_agent".to_string(),
            output: siumai_core::types::ToolResultOutput::json(json!({
                "status": "completed",
                "message": "ok"
            })),
            provider_executed: Some(true),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                json!({
                    "itemId": "ct_1"
                }),
            )])),
        },
    ]));

    let bridged = bridge_chat_response_to_openai_responses_json_value(
        &response,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::Strict,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_exact());

    let value = bridged.value.expect("json body");
    assert_eq!(value["output"][0]["type"], json!("custom_tool_call"));
    assert_eq!(value["output"][0]["id"], json!("ct_1"));
    assert_eq!(value["output"][0]["name"], json!("browser_agent"));
    assert_eq!(
        value["output"][0]["input"],
        json!(r#"{"url":"https://example.com"}"#)
    );
    assert_eq!(value["output"][0]["output"]["message"], json!("ok"));
}

#[cfg(feature = "openai")]
#[test]
fn response_bridge_options_can_remap_tool_call_names() {
    let response = ChatResponse::new(MessageContent::MultiModal(vec![ContentPart::tool_call(
        "call_1",
        "weather",
        json!({ "city": "Tokyo" }),
        None,
    )]));

    let bridged = bridge_chat_response_to_openai_chat_completions_json_value_with_options(
        &response,
        Some(BridgeTarget::AnthropicMessages),
        BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.response.remap")
            .with_primitive_remapper(Arc::new(PrefixRemapper)),
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        json!("gw_weather")
    );
}

#[cfg(feature = "openai")]
#[test]
fn response_bridge_hook_can_rewrite_content_before_serialization() {
    let response = ChatResponse::new(MessageContent::Text("visible answer".to_string()));

    let bridged = bridge_chat_response_to_openai_chat_completions_json_value_with_options(
        &response,
        Some(BridgeTarget::AnthropicMessages),
        BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.response.hook")
            .with_response_hook(Arc::new(RedactResponseHook)),
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());
    assert!(
        bridged
            .report
            .lossy_fields
            .iter()
            .any(|field| field == "response.content")
    );

    let value = bridged.value.expect("json body");
    assert_eq!(value["choices"][0]["message"]["content"], json!("[hooked]"));
}
