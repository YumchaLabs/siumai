#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::LlmError;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai-compatible")
        .join("chat-messages")
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

fn build_body(req: &siumai::prelude::unified::ChatRequest) -> Result<Value, LlmError> {
    let adapter =
        siumai_provider_openai::providers::openai_compatible::adapter::OpenAiStandardAdapter {
            base_url: "https://my.api.com/v1".to_string(),
        };
    let spec =
        siumai_provider_openai::standards::openai::compat::spec::OpenAiCompatibleSpecWithAdapter::new(
            Arc::new(adapter),
        );

    let ctx = ProviderContext::new(
        "test-provider",
        "https://my.api.com/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );

    let transformers = spec.choose_chat_transformers(req, &ctx);
    let mut body = transformers.request.transform_chat(req)?;
    if let Some(cb) = spec.chat_before_send(req, &ctx) {
        body = cb(&body)?;
    }
    Ok(body)
}

fn run_case(root: &Path) {
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let expected_error_path = root.join("expected_error.txt");
    if expected_error_path.exists() {
        let expected = read_text(expected_error_path);
        let err = build_body(&req).expect_err("expected transform error");
        let msg = err.to_string();
        assert!(
            msg.contains(expected.trim()),
            "error mismatch: expected substring '{expected}', got '{msg}'"
        );
        return;
    }

    let expected_body: Value = read_json(root.join("expected_body.json"));

    let got_body = build_body(&req).expect("transform");

    let mut got_value = got_body;
    let mut expected_value = expected_body;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(got_value, expected_value);
}

#[test]
fn openai_compatible_chat_messages_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no fixture cases found");
    for root in roots {
        run_case(&root);
    }
}
