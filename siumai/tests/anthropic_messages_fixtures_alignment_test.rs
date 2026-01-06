#![cfg(feature = "anthropic")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("anthropic")
        .join("messages")
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
    let spec = siumai_provider_anthropic::providers::anthropic::spec::AnthropicSpec::new();
    let ctx = ProviderContext::new(
        "anthropic",
        "https://api.anthropic.com/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );

    let transformers = spec.choose_chat_transformers(req, &ctx);
    let mut body = transformers.request.transform_chat(req).expect("transform");
    if let Some(cb) = spec.chat_before_send(req, &ctx) {
        body = cb(&body).expect("before_send");
    }
    body
}

fn run_case(root: &Path) {
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let expected_body: Value = read_json(root.join("expected_body.json"));

    let got_body = build_body(&req);

    let mut got_value = got_body;
    let mut expected_value = expected_body;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(got_value, expected_value);

    let resp_path = root.join("response.json");
    let expected_resp_path = root.join("expected_response.json");
    if resp_path.exists() && expected_resp_path.exists() {
        let raw: Value = read_json(resp_path);
        let expected_resp: Value = read_json(expected_resp_path);

        let transformers =
            siumai_provider_anthropic::providers::anthropic::spec::AnthropicSpec::new()
                .choose_chat_transformers(
                    &req,
                    &ProviderContext::new(
                        "anthropic",
                        "https://api.anthropic.com/v1",
                        Some("test-api-key".to_string()),
                        HashMap::new(),
                    ),
                );
        let got = transformers
            .response
            .transform_chat_response(&raw)
            .expect("transform response");

        let mut got_resp_value = serde_json::to_value(got).unwrap();
        let mut expected_resp_value = expected_resp;
        normalize_json(&mut got_resp_value);
        normalize_json(&mut expected_resp_value);
        assert_eq!(got_resp_value, expected_resp_value);
    }
}

#[test]
fn anthropic_messages_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no fixture cases found");
    for root in roots {
        run_case(&root);
    }
}

#[test]
fn anthropic_mcp_servers_enable_beta_header() {
    let req: siumai::prelude::unified::ChatRequest =
        read_json(fixtures_dir().join("anthropic-mcp.1").join("request.json"));

    let spec = siumai_provider_anthropic::providers::anthropic::spec::AnthropicSpec::new();
    let ctx = ProviderContext::new(
        "anthropic",
        "https://api.anthropic.com/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );

    let base = spec.build_headers(&ctx).expect("base headers");
    let extra = spec.chat_request_headers(false, &req, &ctx);
    let merged = spec.merge_request_headers(base, &extra);
    let beta = merged
        .get("anthropic-beta")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    assert!(
        beta.split(',').any(|t| t.trim() == "mcp-client-2025-04-04"),
        "missing mcp-client beta token: {beta}"
    );
}

#[test]
fn anthropic_tool_search_enables_beta_header() {
    let req: siumai::prelude::unified::ChatRequest = read_json(
        fixtures_dir()
            .join("anthropic-tool-search-regex.1")
            .join("request.json"),
    );

    let spec = siumai_provider_anthropic::providers::anthropic::spec::AnthropicSpec::new();
    let ctx = ProviderContext::new(
        "anthropic",
        "https://api.anthropic.com/v1",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );

    let base = spec.build_headers(&ctx).expect("base headers");
    let extra = spec.chat_request_headers(false, &req, &ctx);
    let merged = spec.merge_request_headers(base, &extra);
    let beta = merged
        .get("anthropic-beta")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    assert!(
        beta.split(',')
            .any(|t| t.trim() == "advanced-tool-use-2025-11-20"),
        "missing advanced-tool-use beta token: {beta}"
    );
}
