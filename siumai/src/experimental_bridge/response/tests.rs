use std::collections::HashMap;

use serde_json::json;
use siumai_core::bridge::{BridgeMode, BridgeTarget};
use siumai_core::encoding::JsonEncodeOptions;
use siumai_core::types::{ChatResponse, ContentPart, FinishReason, MessageContent, Usage};

#[cfg(feature = "anthropic")]
use super::bridge_chat_response_to_anthropic_messages_json_value;
#[cfg(feature = "openai")]
use super::{
    bridge_chat_response_to_openai_chat_completions_json_value,
    bridge_chat_response_to_openai_responses_json_value,
};

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
