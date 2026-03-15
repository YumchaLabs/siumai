#![cfg(feature = "xai")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::execution::transformers::response::ResponseTransformer;
use siumai::prelude::ChatResponse;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("xai")
        .join("responses")
        .join("response")
}

fn case_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .expect("read fixture root dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.path())
        .collect();

    dirs.sort_by(|a, b| {
        a.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .cmp(&b.file_name().unwrap_or_default().to_string_lossy())
    });

    dirs
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
            for v in map.values_mut() {
                normalize_json(v);
            }

            let keys: Vec<String> = map
                .iter()
                .filter_map(|(k, v)| {
                    if v.is_null() {
                        return Some(k.clone());
                    }
                    if let Value::Object(obj) = v
                        && obj.is_empty()
                    {
                        return Some(k.clone());
                    }
                    None
                })
                .collect();
            for k in keys {
                map.remove(&k);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                normalize_json(v);
            }
        }
        _ => {}
    }
}

fn run_case(root: &Path) {
    let response: Value = read_json(root.join("response.json"));
    let expected: Value = read_json(root.join("expected_response.json"));

    let tx = siumai_provider_xai::standards::xai::responses_response::XaiResponsesResponseTransformer::new();
    let resp = tx.transform_chat_response(&response).expect("transform");

    let mut got_value = serde_json::to_value(resp).expect("serialize");
    let mut expected_value = expected;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(
        got_value,
        expected_value,
        "fixture case: {}",
        root.display()
    );
}

fn run_case_response(root_name: &str) -> ChatResponse {
    let root = fixtures_dir().join(root_name);
    let response: Value = read_json(root.join("response.json"));
    let tx =
        siumai_provider_xai::standards::xai::responses_response::XaiResponsesResponseTransformer::new();
    tx.transform_chat_response(&response).expect("transform")
}

#[test]
fn xai_responses_response_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no fixture cases found");
    for root in roots {
        run_case(&root);
    }
}

#[test]
fn xai_web_search_response_uses_generic_tool_call_and_typed_metadata_boundary() {
    let resp = run_case_response("web-search-tool.1");
    let value = serde_json::to_value(resp).expect("serialize response");
    let parts = value["content"]["MultiModal"]
        .as_array()
        .expect("expected multimodal content");

    let tool_call = parts
        .iter()
        .find(|part| part.get("type").and_then(|value| value.as_str()) == Some("tool-call"))
        .expect("expected tool call");

    assert_eq!(tool_call["toolName"], serde_json::json!("web_search"));
    assert_eq!(tool_call["providerExecuted"], serde_json::json!(true));
    assert_eq!(
        tool_call["input"],
        serde_json::json!("{\"query\":\"what is xAI\",\"num_results\":5}")
    );

    let source_count = parts
        .iter()
        .filter(|part| part.get("type").and_then(|value| value.as_str()) == Some("source"))
        .count();
    assert_eq!(source_count, 5);
}

#[test]
fn xai_x_search_response_keeps_provider_specific_tool_name_on_generic_tool_call() {
    let resp = run_case_response("x-search-tool.1");
    let value = serde_json::to_value(resp).expect("serialize response");
    let parts = value["content"]["MultiModal"]
        .as_array()
        .expect("expected multimodal content");

    let tool_call = parts
        .iter()
        .find(|part| part.get("type").and_then(|value| value.as_str()) == Some("tool-call"))
        .expect("expected tool call");

    assert_eq!(tool_call["toolName"], serde_json::json!("x_search"));
    assert_eq!(tool_call["providerExecuted"], serde_json::json!(true));
    assert!(
        tool_call["input"]
            .as_str()
            .is_some_and(|value| value.contains("\"query\":\"AI artificial intelligence\"")),
        "expected x_search input payload"
    );
}
