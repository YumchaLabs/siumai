#![cfg(feature = "openai")]

use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_response_to_openai_chat_completions_json_value,
};
use siumai::experimental::encoding::JsonEncodeOptions;
use siumai::experimental::execution::transformers::response::ResponseTransformer;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("chat-completions-response")
}

fn read_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture text")
}

fn read_json(path: impl AsRef<Path>) -> Value {
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

fn normalize_chat_response_json(response: &siumai::prelude::unified::ChatResponse) -> Value {
    let mut value = serde_json::to_value(response).expect("serialize chat response");
    normalize_json(&mut value);
    value
}

fn roundtrip_response(case: &str) -> (Value, Value) {
    let root = fixtures_dir().join(case);
    let raw = read_json(root.join("response.json"));
    let tx =
        siumai::experimental::standards::openai::transformers::response::OpenAiResponseTransformer;

    let response = tx
        .transform_chat_response(&raw)
        .unwrap_or_else(|err| panic!("transform raw response {}: {err:?}", root.display()));

    let bridged = bridge_chat_response_to_openai_chat_completions_json_value(
        &response,
        Some(BridgeTarget::OpenAiChatCompletions),
        BridgeMode::Strict,
        JsonEncodeOptions::default(),
    )
    .unwrap_or_else(|err| panic!("bridge response {}: {err:?}", root.display()));

    assert!(
        !bridged.is_rejected(),
        "bridge rejected fixture case {} with report {:?}",
        root.display(),
        bridged.report
    );
    assert!(
        bridged.report.is_exact(),
        "expected exact roundtrip for {}, got {:?}",
        root.display(),
        bridged.report
    );

    let bridged_json = bridged.value.expect("bridged response json");
    let roundtripped = tx
        .transform_chat_response(&bridged_json)
        .unwrap_or_else(|err| {
            panic!(
                "transform bridged response {} failed: {err:?}",
                root.display()
            )
        });

    (bridged_json, normalize_chat_response_json(&roundtripped))
}

#[test]
fn openai_chat_completions_response_bridge_roundtrip_fixture_exact_cases_match() {
    let cases = ["legacy-function-call.1", "top-level-tool-call.1"];

    for case in cases {
        let root = fixtures_dir().join(case);
        let raw = read_json(root.join("response.json"));
        let tx =
            siumai::experimental::standards::openai::transformers::response::OpenAiResponseTransformer;
        let response = tx
            .transform_chat_response(&raw)
            .unwrap_or_else(|err| panic!("transform raw response {}: {err:?}", root.display()));
        let expected = normalize_chat_response_json(&response);

        let (bridged_json, roundtripped) = roundtrip_response(case);
        assert_eq!(roundtripped, expected, "fixture case: {}", root.display());

        if case == "top-level-tool-call.1" {
            assert_eq!(
                bridged_json["system_fingerprint"],
                serde_json::json!("fp_123"),
                "fixture case: {}",
                root.display()
            );
            assert_eq!(
                bridged_json["service_tier"],
                serde_json::json!("priority"),
                "fixture case: {}",
                root.display()
            );
            assert_eq!(
                bridged_json["usage"]["completion_tokens_details"]["accepted_prediction_tokens"],
                serde_json::json!(5),
                "fixture case: {}",
                root.display()
            );
        }
    }
}
