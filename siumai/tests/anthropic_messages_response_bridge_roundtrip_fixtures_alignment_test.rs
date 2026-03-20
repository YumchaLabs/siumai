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

fn roundtrip_response_json(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let request: ChatRequest = read_json(root.join("request.json"));
    let response: ChatResponse = read_json(root.join("expected_response.json"));

    let bridged = bridge_chat_response_to_anthropic_messages_json_value(
        &response,
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

    let spec = siumai_provider_anthropic::providers::anthropic::spec::AnthropicSpec::new();
    let ctx = ProviderContext::new(
        "anthropic",
        "https://api.anthropic.com/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );
    let transformers = spec.choose_chat_transformers(&request, &ctx);
    let roundtripped = transformers
        .response
        .transform_chat_response(&bridged.value.expect("bridged json body"))
        .expect("transform bridged response");

    project_expected_anthropic_roundtrip(
        serde_json::to_value(roundtripped).expect("serialize roundtripped response"),
    )
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
fn anthropic_messages_response_bridge_roundtrip_tool_search_cases_preserve_text_and_calls() {
    let lossy_cases = [
        "anthropic-tool-search-bm25.1",
        "anthropic-tool-search-regex.1",
    ];

    for case in lossy_cases {
        let roundtripped = roundtrip_response_json(case);
        let expected = expected_response_json(case);

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
    }
}
