#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_request_to_openai_chat_completions_json,
};
use siumai::prelude::unified::ChatRequest;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("chat-messages")
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
    let bridged = bridge_request(case);

    assert!(
        !bridged.is_rejected(),
        "bridge rejected fixture case {} with report {:?}",
        fixtures_dir().join(case).display(),
        bridged.report
    );

    let mut value = bridged.value.expect("bridged request body");
    normalize_json(&mut value);
    value
}

fn bridge_request(case: &str) -> siumai::experimental::bridge::BridgeResult<Value> {
    let root = fixtures_dir().join(case);
    let request: ChatRequest = read_json(root.join("request.json"));

    bridge_chat_request_to_openai_chat_completions_json(
        &request,
        Some(BridgeTarget::OpenAiChatCompletions),
        BridgeMode::BestEffort,
    )
    .unwrap_or_else(|err| panic!("failed to bridge fixture case {}: {err:?}", root.display()))
}

fn expected_body_json(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let mut value: Value = read_json(root.join("expected_body.json"));
    normalize_json(&mut value);
    value
}

#[test]
fn openai_chat_request_bridge_roundtrip_fixture_exact_cases_match() {
    let exact_cases = [
        "openai-chat-system-message-default.1",
        "openai-chat-assistant-tool-call-with-text.1",
        "openai-chat-assistant-tool-call-and-tool-result-json.1",
        "openai-chat-response-format-json-schema.1",
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

#[test]
fn openai_chat_request_bridge_roundtrip_documents_unsupported_pdf_file_id_case() {
    let root = fixtures_dir().join("openai-chat-user-pdf-file-id.1");
    let request: ChatRequest = read_json(root.join("request.json"));

    let err = bridge_chat_request_to_openai_chat_completions_json(
        &request,
        Some(BridgeTarget::OpenAiChatCompletions),
        BridgeMode::BestEffort,
    )
    .expect_err("pdf file-id case should be unsupported");

    let msg = err.user_message();
    assert!(
        msg.contains("application/pdf"),
        "expected unsupported pdf message, got: {msg}"
    );
}
