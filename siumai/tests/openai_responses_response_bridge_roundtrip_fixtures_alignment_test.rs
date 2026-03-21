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
        "web-search-tool.1",
        "file-search-tool.2",
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
