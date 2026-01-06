#![cfg(feature = "openai")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::experimental::execution::transformers::request::RequestTransformer;
use siumai_provider_openai::provider_options::openai::ResponsesApiConfig;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses")
        .join("apply-patch")
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

fn build_body(req: &siumai::prelude::unified::ChatRequest) -> Value {
    let tx =
        siumai::experimental::standards::openai::transformers::request::OpenAiResponsesRequestTransformer;
    let mut body = tx.transform_chat(req).expect("transform");

    let spec = siumai_provider_openai::providers::openai::spec::OpenAiSpec::new()
        .with_forced_responses_api(ResponsesApiConfig::new());
    let ctx = ProviderContext::new("openai", "https://api.openai.com/v1", None, HashMap::new());

    if let Some(cb) = spec.chat_before_send(req, &ctx) {
        body = cb(&body).expect("before_send");
    }

    body
}

fn run_case(root: &Path) {
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let expected_body: Value = read_json(root.join("expected_body.json"));

    let mut got_value = build_body(&req);
    let mut expected_value = expected_body;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(got_value, expected_value);
}

#[test]
fn openai_responses_apply_patch_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no fixture cases found");
    for root in roots {
        run_case(&root);
    }
}
