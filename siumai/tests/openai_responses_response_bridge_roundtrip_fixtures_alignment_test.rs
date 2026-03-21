#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_response_to_openai_responses_json_value,
};
use siumai::experimental::encoding::JsonEncodeOptions;
use siumai::experimental::execution::transformers::response::ResponseTransformer;
use siumai::prelude::unified::ChatResponse;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses")
        .join("response")
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

fn roundtrip_response_json(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let response: ChatResponse = read_json(root.join("expected_response.json"));

    let bridged = bridge_chat_response_to_openai_responses_json_value(
        &response,
        Some(BridgeTarget::OpenAiResponses),
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

    let tx = siumai::experimental::standards::openai::transformers::response::OpenAiResponsesResponseTransformer::new();
    let roundtripped = tx
        .transform_chat_response(&bridged.value.expect("bridged json body"))
        .expect("transform bridged response");

    let mut value = serde_json::to_value(roundtripped).expect("serialize roundtripped response");
    normalize_json(&mut value);
    value
}

fn expected_response_json(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let response: ChatResponse = read_json(root.join("expected_response.json"));
    let mut value = serde_json::to_value(response).expect("serialize fixture response");
    normalize_json(&mut value);
    value
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

fn collect_tool_action_types(value: &Value, case: &str) -> Vec<String> {
    multimodal_parts(value, case)
        .iter()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("tool-result"))
        .filter_map(|part| {
            part.pointer("/output/value/action/type")
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

fn message_level_sources(value: &Value) -> Vec<Value> {
    value["provider_metadata"]["openai"]["sources"]
        .as_array()
        .map(|sources| {
            sources
                .iter()
                .filter(|source| source.get("tool_call_id").is_none())
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn openai_responses_response_bridge_roundtrip_fixture_exact_cases_match() {
    let exact_cases = [
        "basic-text",
        "file-search-tool.1",
        "logprobs.1",
        "mcp-tool-approval.1",
        "tool-calls",
        "reasoning-empty-summary",
        "reasoning-encrypted-content.1",
        "reasoning-summary",
        "apply-patch-tool.1",
        "code-interpreter-tool.1",
        "computer-use-tool.1",
        "image-generation-tool.1",
        "local-shell-tool.1",
        "mcp-tool.1",
        "shell-tool.1",
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
fn openai_responses_response_bridge_roundtrip_source_citation_case_preserves_message_sources() {
    let case = "web-search-tool.1";
    let roundtripped = roundtrip_response_json(case);
    let expected = expected_response_json(case);

    assert_eq!(
        roundtripped.pointer("/provider_metadata/openai/sources"),
        expected.pointer("/provider_metadata/openai/sources"),
        "fixture case {} should preserve message-level source citations",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        collect_text_parts(&roundtripped, case),
        collect_text_parts(&expected, case),
        "fixture case {} should preserve assistant text output",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        collect_tool_action_types(&roundtripped, case),
        collect_tool_action_types(&expected, case),
        "fixture case {} should preserve web-search action ordering",
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
    assert!(
        roundtripped
            .pointer("/content/MultiModal/2/output/value/sources")
            .is_none(),
        "fixture case {} documents that tool-result embedded sources remain lossy",
        fixtures_dir().join(case).display()
    );
}

#[test]
fn openai_responses_response_bridge_roundtrip_file_search_projection_preserves_results() {
    let case = "file-search-tool.2";
    let roundtripped = roundtrip_response_json(case);
    let expected = expected_response_json(case);

    assert_eq!(
        collect_text_parts(&roundtripped, case),
        collect_text_parts(&expected, case),
        "fixture case {} should preserve assistant text output",
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
    assert_eq!(
        roundtripped.pointer("/content/MultiModal/2/output/value/results"),
        expected.pointer("/content/MultiModal/2/output/value/results"),
        "fixture case {} should preserve file-search result payloads",
        fixtures_dir().join(case).display()
    );
    assert_eq!(
        message_level_sources(&roundtripped),
        message_level_sources(&expected),
        "fixture case {} should preserve message-level citations",
        fixtures_dir().join(case).display()
    );
    assert!(
        roundtripped["provider_metadata"]["openai"]["sources"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|source| source.get("tool_call_id").is_some()),
        "fixture case {} documents that tool-scoped file-search sources are still projected into provider metadata",
        fixtures_dir().join(case).display()
    );
}
