use std::collections::HashMap;

use serde_json::json;
use siumai_core::bridge::{BridgeMode, BridgeTarget};
use siumai_core::types::{ChatMessage, ChatRequest, ContentPart};

#[cfg(feature = "anthropic")]
use super::bridge_chat_request_to_anthropic_messages_json;
#[cfg(feature = "openai")]
use super::{
    bridge_chat_request_to_openai_chat_completions_json,
    bridge_chat_request_to_openai_responses_json,
};

#[cfg(feature = "openai")]
#[test]
fn strict_openai_chat_bridge_rejects_reasoning_loss() {
    let mut request = ChatRequest::new(vec![
        ChatMessage::assistant_with_content(vec![
            ContentPart::reasoning("step by step"),
            ContentPart::text("final answer"),
        ])
        .build(),
    ]);
    request.common_params.model = "gpt-4o-mini".to_string();

    let bridged = bridge_chat_request_to_openai_chat_completions_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::Strict,
    )
    .expect("bridge");

    assert!(bridged.is_rejected());
    assert!(bridged.report.is_rejected());
    assert_eq!(bridged.report.lossy_fields.len(), 1);
}

#[cfg(feature = "openai")]
#[test]
fn best_effort_openai_responses_bridge_returns_json_and_report() {
    let mut request = ChatRequest::new(vec![
        ChatMessage::assistant_with_content(vec![
            ContentPart::Reasoning {
                text: "hidden chain of thought".to_string(),
                provider_metadata: Some(HashMap::from([(
                    "openai".to_string(),
                    json!({ "itemId": "rs_1" }),
                )])),
            },
            ContentPart::text("visible answer"),
        ])
        .build(),
    ]);
    request.common_params.model = "gpt-4o-mini".to_string();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::GeminiGenerateContent),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());

    let value = bridged.value.expect("json body");
    let input = value
        .get("input")
        .and_then(|value| value.as_array())
        .expect("responses input array");
    assert!(
        input.iter().any(|item| {
            item.get("type")
                .and_then(|value| value.as_str())
                .is_some_and(|kind| kind == "item_reference")
                && item
                    .get("id")
                    .and_then(|value| value.as_str())
                    .is_some_and(|id| id == "rs_1")
        }),
        "expected reasoning item reference in responses input"
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn anthropic_direct_pair_bridge_flattens_system_and_forces_responses_store_policy() {
    let request = ChatRequest::builder()
        .message(ChatMessage::system("sys-1").build())
        .message(ChatMessage::developer("sys-2").build())
        .message(ChatMessage::user("hi").build())
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    assert_eq!(value["store"], serde_json::json!(false));

    let include = value
        .get("include")
        .and_then(|value| value.as_array())
        .expect("responses include array");
    assert!(
        include
            .iter()
            .any(|value| value.as_str() == Some("reasoning.encrypted_content")),
        "expected reasoning.encrypted_content include"
    );

    let input = value
        .get("input")
        .and_then(|value| value.as_array())
        .expect("responses input array");
    assert_eq!(input[0]["role"], serde_json::json!("system"));
    assert_eq!(input[0]["content"], serde_json::json!("sys-1\n\nsys-2"));
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn anthropic_direct_pair_bridge_translates_web_search_tool_and_required_choice() {
    let request = ChatRequest::builder()
        .message(ChatMessage::user("search rust").build())
        .tools(vec![
            crate::tools::anthropic::web_search_20250305().with_args(json!({
                "allowedDomains": ["example.com"],
                "maxUses": 2
            })),
        ])
        .tool_choice(siumai_core::types::ToolChoice::Required)
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    let tools = value
        .get("tools")
        .and_then(|value| value.as_array())
        .expect("responses tools array");

    assert_eq!(tools[0]["type"], serde_json::json!("web_search"));
    assert_eq!(
        tools[0]["filters"]["allowed_domains"],
        serde_json::json!(["example.com"])
    );
    assert_eq!(value["tool_choice"], serde_json::json!("required"));
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "tools[0].args.maxUses"),
        "expected dropped maxUses warning in bridge report"
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn anthropic_direct_pair_bridge_replays_assistant_reasoning_without_openai_item_id() {
    let mut assistant = ChatMessage::assistant_with_content(vec![
        ContentPart::reasoning("hidden chain of thought"),
        ContentPart::text("final answer"),
    ])
    .build();
    assistant.metadata.custom.insert(
        "anthropic_redacted_thinking_data".to_string(),
        serde_json::json!("enc_payload"),
    );

    let request = ChatRequest::builder()
        .message(assistant)
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    let input = value
        .get("input")
        .and_then(|value| value.as_array())
        .expect("responses input array");

    let reasoning_item = input
        .iter()
        .find(|item| item.get("type").and_then(|value| value.as_str()) == Some("reasoning"))
        .expect("expected reasoning item");

    assert_eq!(
        reasoning_item["id"],
        serde_json::json!("anthropic_reasoning_0_0")
    );
    assert_eq!(
        reasoning_item["encrypted_content"],
        serde_json::json!("enc_payload")
    );
    assert_eq!(
        reasoning_item["summary"][0]["text"],
        serde_json::json!("hidden chain of thought")
    );
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[test]
fn anthropic_direct_pair_bridge_maps_tool_result_to_function_call_output() {
    let request = ChatRequest::builder()
        .message(ChatMessage::tool_result_text("call_1", "web_search", "done").build())
        .tools(vec![crate::tools::anthropic::web_search_20250305()])
        .model("gpt-4.1-mini")
        .build();

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    let value = bridged.value.expect("json body");
    let input = value
        .get("input")
        .and_then(|value| value.as_array())
        .expect("responses input array");

    let tool_output = input
        .iter()
        .find(|item| {
            item.get("type")
                .and_then(|value| value.as_str())
                .is_some_and(|kind| kind == "function_call_output")
        })
        .expect("expected function_call_output item");

    assert_eq!(tool_output["call_id"], serde_json::json!("call_1"));
    assert_eq!(tool_output["output"], serde_json::json!("done"));
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_bridge_reports_cache_breakpoints_beyond_limit() {
    let mut request = ChatRequest::new(vec![
        ChatMessage::system("s1")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
        ChatMessage::system("s2")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
        ChatMessage::user("u1")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
        ChatMessage::assistant("a1")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
        ChatMessage::user("u2")
            .cache_control(siumai_core::types::CacheControl::Ephemeral)
            .build(),
    ]);
    request.common_params.model = "claude-3-5-sonnet-latest".to_string();

    let bridged = bridge_chat_request_to_anthropic_messages_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(bridged.report.is_lossy());
    assert_eq!(bridged.report.dropped_fields.len(), 1);
    assert_eq!(
        bridged.report.dropped_fields[0],
        "messages[4].metadata.cache_control"
    );
    assert!(bridged.value.is_some());
}
