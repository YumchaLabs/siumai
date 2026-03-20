#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use siumai::experimental::bridge::bridge_openai_chat_completions_json_to_chat_request;
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

fn request_json_from_expected_body(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let body: Value = read_json(root.join("expected_body.json"));
    let request =
        bridge_openai_chat_completions_json_to_chat_request(&body).unwrap_or_else(|err| {
            panic!(
                "failed to normalize fixture case {}: {err:?}",
                root.display()
            )
        });
    let mut value = serde_json::to_value(request).expect("serialize normalized request");
    normalize_json(&mut value);
    value
}

fn request_json_from_fixture(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let request: ChatRequest = read_json(root.join("request.json"));
    let mut value = serde_json::to_value(request).expect("serialize fixture request");
    normalize_json(&mut value);
    value
}

#[test]
fn openai_chat_request_normalization_fixture_exact_cases_match() {
    let exact_cases = [
        "openai-chat-system-message-default.1",
        "openai-chat-assistant-tool-call-with-text.1",
        "openai-chat-assistant-tool-call-and-tool-result-json.1",
        "openai-chat-user-pdf-file-id.1",
    ];

    for case in exact_cases {
        let got = request_json_from_expected_body(case);
        let expected = request_json_from_fixture(case);
        assert_eq!(
            got,
            expected,
            "fixture case: {}",
            fixtures_dir().join(case).display()
        );
    }
}

#[test]
fn openai_chat_request_normalization_best_effort_cases_are_documented() {
    let mut expected = json!({
        "messages": [
            {
                "role": "developer",
                "content": { "Text": "You are a helpful assistant." }
            }
        ],
        "common_params": { "model": "gpt-4o-mini" },
        "stream": false
    });
    normalize_json(&mut expected);
    assert_eq!(
        request_json_from_expected_body("openai-chat-system-message-mode-developer.1"),
        expected
    );

    let mut expected = json!({
        "messages": [],
        "common_params": { "model": "gpt-4o-mini" },
        "stream": false
    });
    normalize_json(&mut expected);
    assert_eq!(
        request_json_from_expected_body("openai-chat-system-message-mode-remove.1"),
        expected
    );

    let mut expected = json!({
        "messages": [
            {
                "role": "user",
                "content": { "Text": "Return JSON." }
            }
        ],
        "common_params": { "model": "gpt-4o-mini" },
        "responseFormat": {
            "type": "json",
            "schema": {
                "type": "object",
                "properties": { "ok": { "type": "boolean" } },
                "required": ["ok"]
            },
            "name": "response",
            "strict": false
        },
        "stream": false
    });
    normalize_json(&mut expected);
    assert_eq!(
        request_json_from_expected_body("openai-chat-response-format-json-schema.1"),
        expected
    );
}
