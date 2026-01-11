#![cfg(feature = "groq")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::{ChatRequest, LlmError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("groq")
        .join("chat-requests")
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

fn build_body(req: &ChatRequest) -> Result<Value, LlmError> {
    let spec = siumai_provider_groq::providers::groq::spec::GroqSpec;
    let ctx = ProviderContext::new(
        "groq",
        "https://api.groq.com/openai/v1",
        Some("gsk-test".to_string()),
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
    let req: ChatRequest = read_json(root.join("request.json"));
    let expected_body: Value = read_json(root.join("expected_body.json"));
    let expected_url = read_text(root.join("expected_url.txt")).trim().to_string();

    let got_body = build_body(&req).expect("transform");
    let mut got_value = got_body;
    let mut expected_value = expected_body;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(
        got_value,
        expected_value,
        "fixture case: {}",
        root.display()
    );

    let spec = siumai_provider_groq::providers::groq::spec::GroqSpec;
    let ctx = ProviderContext::new(
        "groq",
        "https://api.groq.com/openai/v1",
        Some("gsk-test".to_string()),
        HashMap::new(),
    );
    let url = spec.chat_url(req.stream, &req, &ctx);
    assert_eq!(url, expected_url, "fixture case: {}", root.display());
}

#[test]
fn groq_chat_request_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no fixture cases found");
    for root in roots {
        run_case(&root);
    }
}
