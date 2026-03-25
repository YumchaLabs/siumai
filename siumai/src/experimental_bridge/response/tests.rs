#![allow(dead_code, unused_imports)]

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use siumai_core::bridge::{
    BridgeCustomization, BridgeMode, BridgeOptions, BridgePrimitiveContext,
    BridgePrimitiveRemapper, BridgeTarget, ResponseBridgeContext, ResponseBridgeHook,
};
#[cfg(feature = "anthropic")]
use siumai_core::bridge::{
    BridgeLossAction, BridgeLossPolicy, RequestBridgeContext, StreamBridgeContext,
};
use siumai_core::encoding::JsonEncodeOptions;
use siumai_core::types::{ChatResponse, ContentPart, FinishReason, MessageContent};

#[cfg(any(feature = "anthropic", feature = "google"))]
use siumai_core::types::Usage;

#[cfg(feature = "anthropic")]
use super::bridge_chat_response_to_anthropic_messages_json_value;
#[cfg(feature = "google")]
use super::bridge_chat_response_to_gemini_generate_content_json_value;
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

#[cfg(feature = "anthropic")]
struct ContinueLossyPolicy;

#[cfg(feature = "anthropic")]
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

struct CompositeResponseCustomization;

impl BridgeCustomization for CompositeResponseCustomization {
    fn transform_response(
        &self,
        ctx: &ResponseBridgeContext,
        response: &mut ChatResponse,
        report: &mut siumai_core::bridge::BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        assert_eq!(ctx.source, Some(BridgeTarget::AnthropicMessages));
        assert_eq!(ctx.target, BridgeTarget::OpenAiChatCompletions);
        assert_eq!(
            ctx.route_label.as_deref(),
            Some("tests.response.customization")
        );
        assert_eq!(ctx.path_label.as_deref(), Some("normalized-response"));

        let MessageContent::MultiModal(parts) = &response.content else {
            panic!("expected multimodal response content");
        };
        let ContentPart::ToolCall {
            tool_call_id,
            tool_name,
            ..
        } = &parts[1]
        else {
            panic!("expected tool call in response content");
        };
        assert_eq!(tool_call_id, "bundle_call_1");
        assert_eq!(tool_name, "bundle_weather");

        response.content =
            MessageContent::MultiModal(vec![ContentPart::text("[bundle]"), parts[1].clone()]);
        report.record_lossy_field(
            "response.content",
            "bundled customization rewrote response content before serialization",
        );
        Ok(())
    }

    fn remap_tool_name(&self, ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        assert_eq!(ctx.source, Some(BridgeTarget::AnthropicMessages));
        assert_eq!(ctx.target, BridgeTarget::OpenAiChatCompletions);
        Some(format!("bundle_{name}"))
    }

    fn remap_tool_call_id(&self, ctx: &BridgePrimitiveContext, id: &str) -> Option<String> {
        assert_eq!(ctx.source, Some(BridgeTarget::AnthropicMessages));
        assert_eq!(ctx.target, BridgeTarget::OpenAiChatCompletions);
        Some(format!("bundle_{id}"))
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
fn strict_openai_responses_bridge_preserves_tool_approval_requests() {
    let response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::tool_call(
            "id-0",
            "mcp.create_short_url",
            json!({
                "alias": "",
                "description": "",
                "max_clicks": 100,
                "password": "",
                "url": "https://ai-sdk.dev/"
            }),
            Some(true),
        ),
        ContentPart::tool_approval_request("mcpr_1", "id-0"),
    ]));

    let bridged = bridge_chat_response_to_openai_responses_json_value(
        &response,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::Strict,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_exact());

    let value = bridged.value.expect("json body");
    assert_eq!(value["output"].as_array().map(Vec::len), Some(1));
    assert_eq!(value["output"][0]["type"], json!("mcp_approval_request"));
    assert_eq!(value["output"][0]["id"], json!("mcpr_1"));
    assert_eq!(value["output"][0]["name"], json!("create_short_url"));
    assert_eq!(
        value["output"][0]["arguments"],
        json!(
            r#"{"alias":"","description":"","max_clicks":100,"password":"","url":"https://ai-sdk.dev/"}"#
        )
    );
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

#[cfg(feature = "openai")]
#[test]
fn strict_openai_chat_completions_bridge_rejects_reasoning_tool_result_and_metadata_loss() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::reasoning("internal thinking"),
        ContentPart::text("visible answer"),
        ContentPart::ToolResult {
            tool_call_id: "call_1".to_string(),
            tool_name: "weather".to_string(),
            output: siumai_core::types::ToolResultOutput::json(json!({
                "temperature": 72
            })),
            provider_executed: Some(true),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                json!({
                    "itemId": "fc_1"
                }),
            )])),
        },
        ContentPart::tool_approval_request("mcpr_1", "call_1"),
    ]));
    response.finish_reason = Some(FinishReason::StopSequence);
    response.provider_metadata = Some(HashMap::from([(
        "openai".to_string(),
        HashMap::from([("responseId".to_string(), json!("resp_1"))]),
    )]));

    let bridged = bridge_chat_response_to_openai_chat_completions_json_value(
        &response,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::Strict,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(bridged.is_rejected());
    assert!(bridged.report.is_rejected());
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "content[0]"),
        "expected reasoning block drop"
    );
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "content[2]"),
        "expected tool result drop"
    );
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "content[3]"),
        "expected tool approval request drop"
    );
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "provider_metadata.openai"),
        "expected top-level provider metadata drop"
    );
    assert!(
        bridged
            .report
            .lossy_fields
            .iter()
            .any(|field| field == "finish_reason"),
        "expected stop-sequence collapse to be reported as lossy"
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
    assert_eq!(value["content"][1]["type"], json!("server_tool_use"));
    assert_eq!(value["content"][1]["id"], json!("call_1"));
    assert_eq!(value["content"][1]["name"], json!("web_search"));
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

#[cfg(feature = "anthropic")]
#[test]
fn strict_anthropic_response_bridge_preserves_text_part_citations_exactly() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("Intro"),
        ContentPart::Text {
            text: "Grounded fact".to_string(),
            provider_metadata: Some(HashMap::from([(
                "anthropic".to_string(),
                json!({
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://example.com/fact",
                            "title": "Fact"
                        }
                    ]
                }),
            )])),
        },
        ContentPart::text("Outro"),
    ]));
    response.id = Some("msg_1".to_string());
    response.model = Some("claude-sonnet-4-5".to_string());
    response.provider_metadata = Some(HashMap::from([(
        "anthropic".to_string(),
        HashMap::from([(
            "citations".to_string(),
            json!([
                {
                    "content_block_index": 1,
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://example.com/fact",
                            "title": "Fact"
                        }
                    ]
                }
            ]),
        )]),
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
    assert!(value["content"][0].get("citations").is_none());
    assert_eq!(
        value["content"][1]["citations"][0]["url"],
        json!("https://example.com/fact")
    );
    assert!(value["content"][2].get("citations").is_none());
}

#[cfg(feature = "anthropic")]
#[test]
fn strict_anthropic_response_bridge_preserves_mcp_server_name_metadata() {
    let response = ChatResponse::new(MessageContent::MultiModal(vec![ContentPart::ToolCall {
        tool_call_id: "mcptoolu_1".to_string(),
        tool_name: "echo".to_string(),
        arguments: json!({ "message": "hello" }),
        provider_executed: Some(true),
        provider_metadata: Some(HashMap::from([(
            "anthropic".to_string(),
            json!({
                "serverName": "echo-prod"
            }),
        )])),
    }]));

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
    assert_eq!(value["content"][0]["type"], json!("mcp_tool_use"));
    assert_eq!(value["content"][0]["name"], json!("echo"));
    assert_eq!(value["content"][0]["server_name"], json!("echo-prod"));
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

#[cfg(feature = "openai")]
#[test]
fn response_bridge_customization_bundle_can_rewrite_content_and_remap_tools() {
    let response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("visible answer"),
        ContentPart::tool_call("call_1", "weather", json!({ "city": "Tokyo" }), None),
    ]));

    let bridged = bridge_chat_response_to_openai_chat_completions_json_value_with_options(
        &response,
        Some(BridgeTarget::AnthropicMessages),
        BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.response.customization")
            .with_customization(Arc::new(CompositeResponseCustomization)),
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
    assert_eq!(value["choices"][0]["message"]["content"], json!("[bundle]"));
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"][0]["id"],
        json!("bundle_call_1")
    );
    assert_eq!(
        value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        json!("bundle_weather")
    );
}

#[cfg(feature = "google")]
#[test]
fn gemini_response_bridge_reports_reasoning_finish_reason_and_usage_breakdown_loss() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::reasoning("internal thinking"),
        ContentPart::text("visible answer"),
        ContentPart::tool_call("call_1", "search", json!({ "q": "rust" }), None),
    ]));
    response.id = Some("resp_1".to_string());
    response.model = Some("gemini-2.5-pro".to_string());
    response.finish_reason = Some(FinishReason::ToolCalls);
    response.system_fingerprint = Some("fp_123".to_string());
    response.service_tier = Some("priority".to_string());
    response.usage = Some(
        Usage::builder()
            .prompt_tokens(11)
            .completion_tokens(7)
            .total_tokens(18)
            .with_prompt_audio_tokens(2)
            .with_cached_tokens(3)
            .with_completion_audio_tokens(1)
            .with_reasoning_tokens(4)
            .with_accepted_prediction_tokens(5)
            .with_rejected_prediction_tokens(6)
            .build(),
    );

    let bridged = bridge_chat_response_to_gemini_generate_content_json_value(
        &response,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());
    assert!(
        !bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "content[0]"),
        "expected Gemini reasoning block to survive bridge"
    );
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "system_fingerprint"),
        "expected system fingerprint drop"
    );
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "service_tier"),
        "expected service tier drop"
    );
    for field in [
        "finish_reason",
        "usage.prompt_tokens_details.audio_tokens",
        "usage.completion_tokens_details.audio_tokens",
        "usage.completion_tokens_details.accepted_prediction_tokens",
        "usage.completion_tokens_details.rejected_prediction_tokens",
    ] {
        assert!(
            bridged
                .report
                .lossy_fields
                .iter()
                .any(|candidate| candidate == field),
            "expected lossy field {field}"
        );
    }

    let value = bridged.value.expect("json body");
    assert_eq!(
        value["candidates"][0]["content"]["parts"][0]["text"],
        json!("internal thinking")
    );
    assert_eq!(
        value["candidates"][0]["content"]["parts"][0]["thought"],
        json!(true)
    );
    assert_eq!(
        value["candidates"][0]["content"]["parts"][1]["text"],
        json!("visible answer")
    );
    assert_eq!(
        value["candidates"][0]["content"]["parts"][2]["functionCall"]["name"],
        json!("search")
    );
    assert_eq!(value["candidates"][0]["finishReason"], json!("STOP"));
    assert_eq!(value["responseId"], json!("resp_1"));
    assert_eq!(value["modelVersion"], json!("gemini-2.5-pro"));
    assert_eq!(value["usageMetadata"]["promptTokenCount"], json!(11));
    assert_eq!(value["usageMetadata"]["candidatesTokenCount"], json!(7));
    assert_eq!(value["usageMetadata"]["totalTokenCount"], json!(18));
    assert_eq!(value["usageMetadata"]["cachedContentTokenCount"], json!(3));
    assert_eq!(value["usageMetadata"]["thoughtsTokenCount"], json!(4));
}

#[cfg(feature = "google")]
#[test]
fn gemini_response_bridge_preserves_native_metadata_and_source_grounding() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::text("visible answer"),
        ContentPart::tool_call("call_1", "search", json!({ "q": "rust" }), None),
        ContentPart::source("src_1", "url", "https://example.com/fact", "Fact"),
    ]));
    response.id = Some("resp_1".to_string());
    response.model = Some("gemini-2.5-pro".to_string());
    response.finish_reason = Some(FinishReason::Stop);
    response.usage = Some(
        Usage::builder()
            .prompt_tokens(11)
            .completion_tokens(7)
            .total_tokens(18)
            .with_cached_tokens(3)
            .with_reasoning_tokens(4)
            .build(),
    );
    response.provider_metadata = Some(HashMap::from([(
        "google".to_string(),
        HashMap::from([
            (
                "groundingMetadata".to_string(),
                json!({
                    "searchEntryPoint": { "renderedContent": "<div/>" },
                    "groundingChunks": [
                        { "web": { "uri": "https://example.com/original", "title": "Original" } }
                    ]
                }),
            ),
            (
                "urlContextMetadata".to_string(),
                json!({
                    "urlMetadata": [
                        {
                            "retrievedUrl": "https://example.com/fact",
                            "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"
                        }
                    ]
                }),
            ),
            (
                "promptFeedback".to_string(),
                json!({
                    "blockReason": "BLOCK_REASON_UNSPECIFIED"
                }),
            ),
            (
                "safetyRatings".to_string(),
                json!([
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE"
                    }
                ]),
            ),
            ("avgLogprobs".to_string(), json!(-0.25)),
            (
                "logprobsResult".to_string(),
                json!({
                    "topCandidates": [],
                    "chosenCandidates": [],
                    "logProbabilitySum": -1.0
                }),
            ),
            (
                "sources".to_string(),
                json!([
                    {
                        "id": "src_0",
                        "source_type": "url",
                        "url": "https://example.com/original",
                        "title": "Original"
                    }
                ]),
            ),
        ]),
    )]));

    let bridged = bridge_chat_response_to_gemini_generate_content_json_value(
        &response,
        Some(BridgeTarget::GeminiGenerateContent),
        BridgeMode::Strict,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_exact());

    let value = bridged.value.expect("json body");
    assert_eq!(value["responseId"], json!("resp_1"));
    assert_eq!(value["modelVersion"], json!("gemini-2.5-pro"));
    assert_eq!(
        value["promptFeedback"]["blockReason"],
        json!("BLOCK_REASON_UNSPECIFIED")
    );
    assert_eq!(value["usageMetadata"]["cachedContentTokenCount"], json!(3));
    assert_eq!(
        value["candidates"][0]["groundingMetadata"]["searchEntryPoint"]["renderedContent"],
        json!("<div/>")
    );
    assert_eq!(
        value["candidates"][0]["groundingMetadata"]["groundingChunks"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(
        value["candidates"][0]["urlContextMetadata"]["urlMetadata"][0]["retrievedUrl"],
        json!("https://example.com/fact")
    );
    assert_eq!(
        value["candidates"][0]["safetyRatings"][0]["category"],
        json!("HARM_CATEGORY_HATE_SPEECH")
    );
    assert_eq!(value["candidates"][0]["avgLogprobs"], json!(-0.25));
    assert_eq!(
        value["candidates"][0]["logprobsResult"]["logProbabilitySum"],
        json!(-1.0)
    );
}
