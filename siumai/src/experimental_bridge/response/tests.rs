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
fn strict_openai_responses_bridge_rejects_reasoning_loss() {
    let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
        ContentPart::reasoning("hidden chain of thought"),
        ContentPart::text("final answer"),
    ]));
    response.id = Some("resp_1".to_string());
    response.model = Some("gpt-4.1-mini".to_string());

    let bridged = bridge_chat_response_to_openai_responses_json_value(
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
        "expected dropped reasoning field in report"
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
