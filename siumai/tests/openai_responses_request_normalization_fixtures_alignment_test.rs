#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use siumai::experimental::bridge::bridge_openai_responses_json_to_chat_request;
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

fn request_json_from_expected_body(case: &str) -> Value {
    let root = fixtures_dir().join(case);
    let body: Value = read_json(root.join("expected_body.json"));
    let request = bridge_openai_responses_json_to_chat_request(&body).unwrap_or_else(|err| {
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
fn openai_responses_request_normalization_fixture_exact_cases_match() {
    let exact_cases = [
        "assistant-text",
        "assistant-tool-call",
        "assistant-tool-call-multiple",
        "local-shell-store-false",
        "tool-output-local-shell",
        "tool-output-shell",
        "tool-output-apply-patch",
        "tool-output-shell-and-apply-patch",
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
fn openai_responses_request_normalization_item_reference_cases_are_best_effort() {
    let got = request_json_from_expected_body("assistant-tool-call-ids");
    let mut expected = json!({
        "messages": [
            {
                "role": "assistant",
                "content": { "Text": "" },
                "metadata": { "id": "id_123" }
            },
            {
                "role": "assistant",
                "content": { "Text": "" },
                "metadata": { "id": "id_456" }
            }
        ],
        "common_params": { "model": "gpt-5-nano" },
        "stream": false
    });
    normalize_json(&mut expected);
    assert_eq!(got, expected);

    let got = request_json_from_expected_body("provider-tool-call-store-true-item-reference");
    let mut expected = json!({
        "messages": [
            {
                "role": "assistant",
                "content": { "Text": "" },
                "metadata": {
                    "id": "ci_68c2e2cf522c81908f3e2c1bccd1493b0b24aae9c6c01e4f"
                }
            }
        ],
        "common_params": { "model": "gpt-5-nano" },
        "stream": false
    });
    normalize_json(&mut expected);
    assert_eq!(got, expected);

    let got = request_json_from_expected_body("local-shell-store-true");
    let mut expected = json!({
        "messages": [
            {
                "role": "assistant",
                "content": { "Text": "" },
                "metadata": {
                    "id": "lsh_68c2e2cf522c81908f3e2c1bccd1493b0b24aae9c6c01e4f"
                }
            },
            {
                "role": "tool",
                "content": {
                    "MultiModal": [
                        {
                            "type": "tool-result",
                            "toolCallId": "call_XWgeTylovOiS8xLNz2TONOgO",
                            "toolName": "shell",
                            "output": {
                                "type": "json",
                                "value": { "output": "example output" }
                            }
                        }
                    ]
                },
                "metadata": {}
            }
        ],
        "tools": [
            { "type": "provider-defined", "id": "openai.local_shell", "name": "shell" }
        ],
        "common_params": { "model": "gpt-5-nano" },
        "stream": false
    });
    normalize_json(&mut expected);
    assert_eq!(got, expected);
}

#[test]
fn openai_responses_request_normalization_function_output_only_fixtures_drop_unknown_tool_names() {
    let got = request_json_from_expected_body("tool-message-text");
    let mut expected = json!({
        "messages": [
            {
                "role": "tool",
                "content": {
                    "MultiModal": [
                        {
                            "type": "tool-result",
                            "toolCallId": "call_123",
                            "toolName": "",
                            "output": {
                                "type": "text",
                                "value": "The weather in San Francisco is 72°F"
                            }
                        }
                    ]
                }
            }
        ],
        "common_params": { "model": "gpt-5-nano" },
        "stream": false
    });
    normalize_json(&mut expected);
    assert_eq!(got, expected);
}
