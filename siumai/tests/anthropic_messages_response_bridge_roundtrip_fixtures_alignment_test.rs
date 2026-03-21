#![cfg(feature = "anthropic")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_response_to_anthropic_messages_json_value,
};
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::experimental::encoding::JsonEncodeOptions;
use siumai::prelude::unified::{ChatRequest, ChatResponse};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("anthropic")
        .join("messages")
}

fn stream_fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("anthropic")
        .join("messages-stream")
}

fn read_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture text")
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let text = read_text(path);
    serde_json::from_str(&text).expect("parse fixture json")
}

fn normalize_json(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for inner in map.values_mut() {
                normalize_json(inner);
            }

            let keys: Vec<String> = map
                .iter()
                .filter_map(|(key, inner)| {
                    if inner.is_null() {
                        return Some(key.clone());
                    }
                    if let Value::Object(obj) = inner
                        && obj.is_empty()
                    {
                        return Some(key.clone());
                    }
                    None
                })
                .collect();
            for key in keys {
                map.remove(&key);
            }
        }
        Value::Array(items) => {
            for inner in items.iter_mut() {
                normalize_json(inner);
            }
        }
        _ => {}
    }
}

fn project_expected_anthropic_roundtrip(mut value: Value) -> Value {
    if let Some(obj) = value.as_object_mut() {
        if let Some(usage) = obj.get_mut("usage").and_then(Value::as_object_mut) {
            usage.remove("cached_tokens");
            usage.remove("prompt_tokens_details");
        }

        if let Some(provider_metadata) = obj
            .get_mut("provider_metadata")
            .and_then(Value::as_object_mut)
        {
            provider_metadata.remove("anthropic");
        }
    }
    normalize_json(&mut value);
    value
}

fn normalize_response_json(value: Value) -> Value {
    let mut value = value;
    normalize_json(&mut value);
    value
}

fn transform_anthropic_provider_response(request: &ChatRequest, raw: &Value) -> ChatResponse {
    let spec = siumai_provider_anthropic::providers::anthropic::spec::AnthropicSpec::new();
    let ctx = ProviderContext::new(
        "anthropic",
        "https://api.anthropic.com/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );
    spec.choose_chat_transformers(request, &ctx)
        .response
        .transform_chat_response(raw)
        .expect("transform provider response")
}

fn transform_provider_response_json(case: &str, response_path: &Path) -> Value {
    let root = fixtures_dir().join(case);
    let request: ChatRequest = read_json(root.join("request.json"));
    let raw: Value = read_json(response_path);
    let roundtripped = transform_anthropic_provider_response(&request, &raw);

    project_expected_anthropic_roundtrip(
        serde_json::to_value(roundtripped).expect("serialize provider response"),
    )
}

fn transform_provider_response_full_json(case: &str, response_path: &Path) -> Value {
    let root = fixtures_dir().join(case);
    let request: ChatRequest = read_json(root.join("request.json"));
    let raw: Value = read_json(response_path);
    let response = transform_anthropic_provider_response(&request, &raw);
    normalize_response_json(serde_json::to_value(response).expect("serialize provider response"))
}

fn roundtrip_chat_response_json(case: &str, response: &ChatResponse) -> Value {
    let root = fixtures_dir().join(case);

    let bridged = bridge_chat_response_to_anthropic_messages_json_value(
        response,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
        JsonEncodeOptions::default(),
    )
    .unwrap_or_else(|err| panic!("failed to bridge fixture case {}: {err:?}", root.display()));

    assert!(
        !bridged.is_rejected(),
        "bridge rejected fixture case {} with report {:?}",
        root.display(),
        bridged.report
    );

    let request: ChatRequest = read_json(root.join("request.json"));
    let bridged_json = bridged.value.expect("bridged json body");
    let roundtripped = transform_anthropic_provider_response(&request, &bridged_json);

    project_expected_anthropic_roundtrip(
        serde_json::to_value(roundtripped).expect("serialize roundtripped response"),
    )
}

fn roundtrip_response_json(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let response: ChatResponse = read_json(root.join("expected_response.json"));
    roundtrip_chat_response_json(case, &response)
}

fn roundtrip_provider_response_json(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    roundtrip_provider_response_json_from_path(case, &root.join("response.json"))
}

fn roundtrip_provider_response_json_from_path(case: &str, response_path: &Path) -> Value {
    let root = fixtures_dir().join(case);
    let request: ChatRequest = read_json(root.join("request.json"));
    let raw: Value = read_json(response_path);
    let response = transform_anthropic_provider_response(&request, &raw);
    roundtrip_chat_response_json(case, &response)
}

fn roundtrip_provider_response_full_json_from_path(case: &str, response_path: &Path) -> Value {
    let root = fixtures_dir().join(case);
    let request: ChatRequest = read_json(root.join("request.json"));
    let raw: Value = read_json(response_path);
    let response = transform_anthropic_provider_response(&request, &raw);
    normalize_response_json(roundtrip_chat_response_json_value(
        case, &request, &response,
    ))
}

fn roundtrip_chat_response_json_value(
    case: &str,
    request: &ChatRequest,
    response: &ChatResponse,
) -> Value {
    let root = fixtures_dir().join(case);
    let bridged = bridge_chat_response_to_anthropic_messages_json_value(
        response,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
        JsonEncodeOptions::default(),
    )
    .unwrap_or_else(|err| panic!("failed to bridge fixture case {}: {err:?}", root.display()));

    assert!(
        !bridged.is_rejected(),
        "bridge rejected fixture case {} with report {:?}",
        root.display(),
        bridged.report
    );

    let bridged_json = bridged.value.expect("bridged json body");
    let roundtripped = transform_anthropic_provider_response(request, &bridged_json);
    serde_json::to_value(roundtripped).expect("serialize roundtripped response")
}

fn expected_response_json(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let response: ChatResponse = read_json(root.join("expected_response.json"));
    project_expected_anthropic_roundtrip(
        serde_json::to_value(response).expect("serialize fixture response"),
    )
}

fn multimodal_parts<'a>(value: &'a Value, case: &str) -> &'a [Value] {
    value["content"]["MultiModal"]
        .as_array()
        .unwrap_or_else(|| panic!("fixture case {} missing content.MultiModal", case))
}

fn collect_text_parts(value: &Value, case: &str) -> Vec<String> {
    multimodal_parts(value, case)
        .iter()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|part| {
            part.get("text")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .collect()
}

fn collect_tool_call_names(value: &Value, case: &str) -> Vec<String> {
    multimodal_parts(value, case)
        .iter()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("tool-call"))
        .filter_map(|part| {
            part.get("toolName")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .collect()
}

fn collect_tool_result_names(value: &Value, case: &str) -> Vec<String> {
    multimodal_parts(value, case)
        .iter()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("tool-result"))
        .filter_map(|part| {
            part.get("toolName")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .collect()
}

fn count_part_type(value: &Value, case: &str, part_type: &str) -> usize {
    multimodal_parts(value, case)
        .iter()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some(part_type))
        .count()
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_fixture_projected_cases_match() {
    let projected_cases = [
        "anthropic-json-output-format.1",
        "anthropic-json-tool.1",
        "anthropic-json-other-tool.1",
    ];

    for case in projected_cases {
        assert_eq!(
            roundtrip_response_json(case),
            expected_response_json(case),
            "fixture case: {}",
            fixtures_dir().join(case).display()
        );
    }
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_tool_search_cases_match_projection() {
    let exact_cases = [
        "anthropic-tool-search-bm25.1",
        "anthropic-tool-search-regex.1",
    ];

    for case in exact_cases {
        assert_eq!(
            roundtrip_response_json(case),
            expected_response_json(case),
            "fixture case: {}",
            fixtures_dir().join(case).display()
        );
    }
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_tool_search_provider_cases_match_projection() {
    let exact_cases = [
        "anthropic-tool-search-bm25.1",
        "anthropic-tool-search-regex.1",
    ];

    for case in exact_cases {
        let response_path = fixtures_dir().join(case).join("response.json");
        assert_eq!(
            roundtrip_provider_response_json_from_path(case, &response_path),
            transform_provider_response_json(case, &response_path),
            "fixture case: {}",
            response_path.display()
        );
    }
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_programmatic_tool_case_matches_provider_projection()
{
    let case = "anthropic-programmatic-tool-calling.1";
    assert_eq!(
        roundtrip_provider_response_json(case),
        transform_provider_response_json(case, &fixtures_dir().join(case).join("response.json")),
        "fixture case: {}",
        fixtures_dir().join(case).display()
    );
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_code_execution_case_preserves_visible_parts() {
    let case = "anthropic-code-execution-20250825.1";
    let roundtripped = roundtrip_provider_response_json(case);
    let expected =
        transform_provider_response_json(case, &fixtures_dir().join(case).join("response.json"));

    assert_eq!(
        roundtripped.get("model"),
        expected.get("model"),
        "fixture case {} should preserve model",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        roundtripped.get("usage"),
        expected.get("usage"),
        "fixture case {} should preserve usage totals",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        roundtripped.get("finish_reason"),
        expected.get("finish_reason"),
        "fixture case {} should preserve finish_reason",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        collect_text_parts(&roundtripped, case),
        collect_text_parts(&expected, case),
        "fixture case {} should preserve assistant text",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        collect_tool_call_names(&roundtripped, case),
        collect_tool_call_names(&expected, case),
        "fixture case {} should preserve visible tool-call sequence",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        collect_tool_result_names(&roundtripped, case),
        collect_tool_result_names(&expected, case),
        "fixture case {} should preserve visible tool-result sequence",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        count_part_type(&roundtripped, case, "tool-call"),
        count_part_type(&expected, case, "tool-call"),
        "fixture case {} should preserve tool-call count",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        count_part_type(&roundtripped, case, "tool-result"),
        count_part_type(&expected, case, "tool-result"),
        "fixture case {} should preserve tool-result count",
        fixtures_dir().join(case).display()
    );
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_code_execution_case_preserves_container_metadata() {
    let case = "anthropic-code-execution-20250825.1";
    let response_path = fixtures_dir().join(case).join("response.json");
    let roundtripped = roundtrip_provider_response_full_json_from_path(case, &response_path);
    let expected = transform_provider_response_full_json(case, &response_path);

    assert_eq!(
        roundtripped.pointer("/provider_metadata/anthropic/container"),
        expected.pointer("/provider_metadata/anthropic/container"),
        "fixture case {} should preserve anthropic container metadata",
        response_path.display()
    );
    assert_eq!(
        roundtripped.pointer("/provider_metadata/anthropic/usage"),
        expected.pointer("/provider_metadata/anthropic/usage"),
        "fixture case {} should preserve raw anthropic usage metadata",
        response_path.display()
    );
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_memory_case_preserves_context_management() {
    let case = "anthropic-memory-20250818.1";
    let response_path = fixtures_dir().join(case).join("response.json");
    let roundtripped = roundtrip_provider_response_full_json_from_path(case, &response_path);
    let expected = transform_provider_response_full_json(case, &response_path);

    assert_eq!(
        roundtripped.pointer("/provider_metadata/anthropic/contextManagement"),
        expected.pointer("/provider_metadata/anthropic/contextManagement"),
        "fixture case {} should preserve anthropic context management metadata",
        response_path.display()
    );
    assert_eq!(
        roundtripped.pointer("/provider_metadata/anthropic/usage"),
        expected.pointer("/provider_metadata/anthropic/usage"),
        "fixture case {} should preserve raw anthropic usage metadata",
        response_path.display()
    );
    assert_eq!(
        collect_tool_call_names(&roundtripped, case),
        collect_tool_call_names(&expected, case),
        "fixture case {} should preserve visible tool-call sequence",
        response_path.display()
    );
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_provider_hosted_cases_match_projection() {
    let projected_cases = [
        (
            "anthropic-mcp.1",
            stream_fixtures_dir().join("anthropic-mcp.1.json"),
        ),
        (
            "anthropic-web-fetch-tool.1",
            stream_fixtures_dir().join("anthropic-web-fetch-tool.1.json"),
        ),
    ];

    for (case, response_path) in projected_cases {
        assert_eq!(
            roundtrip_provider_response_json_from_path(case, &response_path),
            transform_provider_response_json(case, &response_path),
            "fixture case: {}",
            response_path.display()
        );
    }
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_web_fetch_case_matches_full_projection() {
    let case = "anthropic-web-fetch-tool.1";
    let response_path = stream_fixtures_dir().join("anthropic-web-fetch-tool.1.json");

    assert_eq!(
        roundtrip_provider_response_full_json_from_path(case, &response_path),
        transform_provider_response_full_json(case, &response_path),
        "fixture case: {}",
        response_path.display()
    );
}

#[test]
fn anthropic_messages_response_bridge_roundtrip_web_search_case_preserves_usage_sources_and_citations()
 {
    let case = "anthropic-web-search-tool.1";
    let response_path = stream_fixtures_dir().join("anthropic-web-search-tool.1.json");
    let roundtripped = roundtrip_provider_response_full_json_from_path(case, &response_path);
    let expected = transform_provider_response_full_json(case, &response_path);

    assert_eq!(
        roundtripped.get("model"),
        expected.get("model"),
        "fixture case {} should preserve model",
        response_path.display()
    );
    assert_eq!(
        roundtripped.get("usage"),
        expected.get("usage"),
        "fixture case {} should preserve aggregate usage",
        response_path.display()
    );
    assert_eq!(
        roundtripped.pointer("/provider_metadata/anthropic/usage"),
        expected.pointer("/provider_metadata/anthropic/usage"),
        "fixture case {} should preserve raw anthropic usage metadata",
        response_path.display()
    );
    assert_eq!(
        roundtripped.pointer("/provider_metadata/anthropic/sources"),
        expected.pointer("/provider_metadata/anthropic/sources"),
        "fixture case {} should preserve extracted web-search sources",
        response_path.display()
    );
    assert_eq!(
        roundtripped.pointer("/provider_metadata/anthropic/citations"),
        expected.pointer("/provider_metadata/anthropic/citations"),
        "fixture case {} should preserve citation payloads and block grouping",
        response_path.display()
    );
    assert_eq!(
        collect_text_parts(&roundtripped, case),
        collect_text_parts(&expected, case),
        "fixture case {} should preserve visible assistant text",
        response_path.display()
    );
    assert_eq!(
        collect_tool_call_names(&roundtripped, case),
        collect_tool_call_names(&expected, case),
        "fixture case {} should preserve provider tool-call sequence",
        response_path.display()
    );
    assert_eq!(
        collect_tool_result_names(&roundtripped, case),
        collect_tool_result_names(&expected, case),
        "fixture case {} should preserve provider tool-result sequence",
        response_path.display()
    );
}
