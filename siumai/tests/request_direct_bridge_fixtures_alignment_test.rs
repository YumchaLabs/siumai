#![cfg(all(feature = "anthropic", feature = "openai"))]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_request_to_anthropic_messages_json,
    bridge_chat_request_to_openai_responses_json,
};
use siumai::prelude::unified::ChatRequest;
use std::path::{Path, PathBuf};

fn anthropic_fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("anthropic")
        .join("messages")
}

fn openai_fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses")
        .join("input")
}

fn read_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture text")
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let text = read_text(path);
    serde_json::from_str(&text).expect("parse fixture json")
}

fn anthropic_request(case: &str) -> ChatRequest {
    read_json(anthropic_fixtures_dir().join(case).join("request.json"))
}

fn openai_request(case: &str) -> ChatRequest {
    read_json(openai_fixtures_dir().join(case).join("request.json"))
}

fn bridge_anthropic_fixture_to_openai(
    case: &str,
) -> siumai::experimental::bridge::BridgeResult<Value> {
    let request = anthropic_request(case);
    bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .unwrap_or_else(|err| panic!("failed to bridge fixture case {case}: {err:?}"))
}

fn bridge_openai_fixture_to_anthropic(
    case: &str,
) -> siumai::experimental::bridge::BridgeResult<Value> {
    let request = openai_request(case);
    bridge_chat_request_to_anthropic_messages_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .unwrap_or_else(|err| panic!("failed to bridge fixture case {case}: {err:?}"))
}

fn array_field<'a>(value: &'a Value, key: &str, case: &str) -> &'a [Value] {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture case {case} missing array field `{key}`"))
}

#[test]
fn anthropic_request_bridge_fixture_web_search_case_matches_projection() {
    let bridged = bridge_anthropic_fixture_to_openai("anthropic-web-search-tool.1");
    assert!(
        !bridged.is_rejected(),
        "bridge rejected: {:?}",
        bridged.report
    );

    let body = bridged.value.expect("json body");
    let include = array_field(&body, "include", "anthropic-web-search-tool.1");
    let tools = array_field(&body, "tools", "anthropic-web-search-tool.1");

    assert_eq!(body["store"], serde_json::json!(false));
    assert!(
        include
            .iter()
            .any(|value| value.as_str() == Some("reasoning.encrypted_content")),
        "expected reasoning.encrypted_content include"
    );
    assert!(
        include
            .iter()
            .any(|value| value.as_str() == Some("web_search_call.action.sources")),
        "expected web_search_call.action.sources include"
    );
    assert_eq!(tools[0]["type"], serde_json::json!("web_search"));
    assert_eq!(
        tools[0]["user_location"]["country"],
        serde_json::json!("US")
    );
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "tools[0].args.maxUses"),
        "expected dropped maxUses report entry"
    );
}

#[test]
fn anthropic_request_bridge_fixture_code_execution_case_matches_projection() {
    let bridged = bridge_anthropic_fixture_to_openai("anthropic-code-execution-20250825.1");
    assert!(
        !bridged.is_rejected(),
        "bridge rejected: {:?}",
        bridged.report
    );

    let body = bridged.value.expect("json body");
    let include = array_field(&body, "include", "anthropic-code-execution-20250825.1");
    let tools = array_field(&body, "tools", "anthropic-code-execution-20250825.1");

    assert_eq!(body["store"], serde_json::json!(false));
    assert!(
        include
            .iter()
            .any(|value| value.as_str() == Some("code_interpreter_call.outputs")),
        "expected code_interpreter_call.outputs include"
    );
    assert_eq!(tools[0]["type"], serde_json::json!("code_interpreter"));
}

#[test]
fn anthropic_request_bridge_fixture_effort_case_matches_projection() {
    let bridged = bridge_anthropic_fixture_to_openai("anthropic-effort.1");
    assert!(
        !bridged.is_rejected(),
        "bridge rejected: {:?}",
        bridged.report
    );

    let body = bridged.value.expect("json body");
    assert_eq!(body["reasoning"]["effort"], serde_json::json!("high"));
    assert_eq!(body["reasoning"]["summary"], serde_json::json!("auto"));
}

#[test]
fn anthropic_request_bridge_fixture_structured_output_case_matches_projection() {
    let bridged = bridge_anthropic_fixture_to_openai("anthropic-json-output-format.1");
    assert!(
        !bridged.is_rejected(),
        "bridge rejected: {:?}",
        bridged.report
    );

    let body = bridged.value.expect("json body");
    assert_eq!(
        body["text"]["format"]["type"],
        serde_json::json!("json_schema")
    );
    assert_eq!(
        body["text"]["format"]["schema"]["type"],
        serde_json::json!("object")
    );
    assert_eq!(
        body["text"]["format"]["schema"]["required"],
        serde_json::json!(["name"])
    );
}

#[test]
fn openai_request_bridge_fixture_reasoning_case_carries_redacted_thinking() {
    let bridged = bridge_openai_fixture_to_anthropic("reasoning-store-false-encrypted");
    assert!(
        !bridged.is_rejected(),
        "bridge rejected: {:?}",
        bridged.report
    );

    let body = bridged.value.expect("json body");
    let messages = array_field(&body, "messages", "reasoning-store-false-encrypted");
    let content = messages[0]["content"]
        .as_array()
        .expect("assistant content array");

    assert!(
        content.iter().any(|part| {
            part["type"] == serde_json::json!("redacted_thinking")
                && part["data"] == serde_json::json!("encrypted_content_001")
        }),
        "expected redacted_thinking content block"
    );
    assert!(
        bridged
            .report
            .carried_provider_metadata
            .iter()
            .any(|entry| entry == "openai.reasoning_encrypted_content"),
        "expected carried provider metadata entry"
    );
}

#[test]
fn openai_request_bridge_fixture_system_case_preserves_system_block() {
    let bridged = bridge_openai_fixture_to_anthropic("system-mode-system");
    assert!(
        !bridged.is_rejected(),
        "bridge rejected: {:?}",
        bridged.report
    );

    let body = bridged.value.expect("json body");
    assert_eq!(body["system"], serde_json::json!("Hello"));
}

#[test]
fn openai_request_bridge_fixture_unsupported_local_shell_is_reported() {
    let bridged = bridge_openai_fixture_to_anthropic("local-shell-store-false");
    assert!(
        !bridged.is_rejected(),
        "bridge rejected: {:?}",
        bridged.report
    );

    let body = bridged.value.expect("json body");
    assert!(
        bridged
            .report
            .dropped_fields
            .iter()
            .any(|field| field == "tools[0]"),
        "expected dropped unsupported tool report entry"
    );
    assert!(
        body.get("tools").is_none()
            || body
                .get("tools")
                .and_then(Value::as_array)
                .is_some_and(|tools| tools.is_empty()),
        "expected unsupported OpenAI local_shell tool to be omitted from Anthropic payload"
    );
}
