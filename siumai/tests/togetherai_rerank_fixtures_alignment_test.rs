#![cfg(feature = "togetherai")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::prelude::unified::RerankRequest;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("togetherai")
        .join("rerank")
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
    let request: RerankRequest = read_json(root.join("request.json"));
    let expected_body: Value = read_json(root.join("expected_body.json"));
    let response: Value = read_json(root.join("response.json"));
    let expected_response: Value = read_json(root.join("expected_response.json"));
    let expected_url = read_text(root.join("expected_url.txt")).trim().to_string();

    let standard =
        siumai::experimental::standards::togetherai::rerank::TogetherAiRerankStandard::new();
    let transformers = standard.create_transformers("togetherai");
    let body = transformers
        .request
        .transform(&request)
        .expect("transform body");

    let mut got_body = body;
    let mut expected_body = expected_body;
    normalize_json(&mut got_body);
    normalize_json(&mut expected_body);
    assert_eq!(got_body, expected_body, "fixture case: {}", root.display());

    let got_response = transformers
        .response
        .transform(response)
        .expect("transform response");
    let mut got_value = serde_json::to_value(got_response).expect("serialize");
    let mut expected_value = expected_response;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(
        got_value,
        expected_value,
        "fixture case: {}",
        root.display()
    );

    let spec = standard.create_spec("togetherai");
    let ctx = ProviderContext::new(
        "togetherai",
        "https://api.together.xyz/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );
    let url = spec.rerank_url(&request, &ctx);
    assert_eq!(url, expected_url, "fixture case: {}", root.display());

    let headers = spec.build_headers(&ctx).expect("headers");
    assert_eq!(
        headers
            .get(reqwest::header::AUTHORIZATION)
            .expect("authorization")
            .to_str()
            .expect("header string"),
        "Bearer test-api-key"
    );
    assert_eq!(
        headers
            .get(reqwest::header::CONTENT_TYPE)
            .expect("content-type")
            .to_str()
            .expect("header string"),
        "application/json"
    );
}

#[test]
fn togetherai_rerank_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no fixture cases found");
    for root in roots {
        run_case(&root);
    }
}
