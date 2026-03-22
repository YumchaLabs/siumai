#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_request_to_openai_responses_json,
};
use siumai::prelude::unified::ChatRequest;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
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

fn roundtrip_request_json(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let request: ChatRequest = read_json(root.join("request.json"));

    let bridged = bridge_chat_request_to_openai_responses_json(
        &request,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .unwrap_or_else(|err| panic!("failed to bridge fixture case {}: {err:?}", root.display()));

    assert!(
        !bridged.is_rejected(),
        "bridge rejected fixture case {} with report {:?}",
        root.display(),
        bridged.report
    );

    let mut value = bridged.value.expect("bridged request body");
    normalize_json(&mut value);
    value
}

fn expected_body_json(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let mut value: Value = read_json(root.join("expected_body.json"));
    normalize_json(&mut value);
    value
}

#[test]
fn openai_responses_request_bridge_roundtrip_fixture_exact_cases_match() {
    let exact_cases = [
        "assistant-text",
        "assistant-tool-call",
        "assistant-tool-call-multiple",
        "structured-output-json-schema",
        "system-mode-system",
        "user-text",
    ];

    for case in exact_cases {
        assert_eq!(
            roundtrip_request_json(case),
            expected_body_json(case),
            "fixture case: {}",
            fixtures_dir().join(case).display()
        );
    }
}
