#![cfg(feature = "google-vertex")]

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vertex")
        .join("embedding")
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

fn normalize_floats(value: &mut Value) {
    match value {
        Value::Number(n) => {
            if n.as_i64().is_some() || n.as_u64().is_some() {
                return;
            }
            let Some(f) = n.as_f64() else {
                return;
            };
            let rounded = (f * 1_000_000.0).round() / 1_000_000.0;
            if let Some(nn) = serde_json::Number::from_f64(rounded) {
                *n = nn;
            }
        }
        Value::Object(map) => {
            for v in map.values_mut() {
                normalize_floats(v);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                normalize_floats(v);
            }
        }
        _ => {}
    }
}

fn vertex_ctx_enterprise() -> ProviderContext {
    let base_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google";
    let mut extra: HashMap<String, String> = HashMap::new();
    extra.insert("Authorization".to_string(), "Bearer token".to_string());
    ProviderContext::new("vertex", base_url.to_string(), None, extra)
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingRequestFixture {
    input: Vec<String>,
    model: String,
    #[serde(default)]
    dimensions: Option<u32>,
    #[serde(default)]
    task_type: Option<siumai_core::types::EmbeddingTaskType>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    provider_options_map: siumai_core::types::ProviderOptionsMap,
}

fn to_request(fx: EmbeddingRequestFixture) -> siumai::prelude::unified::EmbeddingRequest {
    let mut req = siumai::prelude::unified::EmbeddingRequest::new(fx.input)
        .with_model(fx.model)
        .with_provider_options_map(fx.provider_options_map);
    if let Some(d) = fx.dimensions {
        req = req.with_dimensions(d);
    }
    if let Some(tt) = fx.task_type {
        req = req.with_task_type(tt);
    }
    if let Some(t) = fx.title {
        req = req.with_title(t);
    }
    req
}

fn run_case(root: &Path) {
    let fx: EmbeddingRequestFixture = read_json(root.join("request.json"));
    let req = to_request(fx);

    let expected_body: Value = read_json(root.join("expected_body.json"));
    let expected_url = read_text(root.join("expected_url.txt")).trim().to_string();
    let raw_response: Value = read_json(root.join("response.json"));
    let expected_response: Value = read_json(root.join("expected_response.json"));

    let ctx = vertex_ctx_enterprise();
    let spec = siumai::experimental::providers::google_vertex::standards::vertex_embedding::VertexEmbeddingStandard::new().create_spec("vertex");

    let url = spec.embedding_url(&req, &ctx);
    assert_eq!(url, expected_url, "fixture case: {}", root.display());

    let transformers = spec.choose_embedding_transformers(&req, &ctx);
    let got_body = transformers.request.transform_embedding(&req).unwrap();
    assert_eq!(got_body, expected_body, "fixture case: {}", root.display());

    let got = transformers
        .response
        .transform_embedding_response(&raw_response)
        .unwrap();

    let mut got_value = serde_json::to_value(got).unwrap();
    let mut expected_value = expected_response;
    normalize_floats(&mut got_value);
    normalize_floats(&mut expected_value);
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(
        got_value,
        expected_value,
        "fixture case: {}",
        root.display()
    );
}

#[test]
fn vertex_embedding_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no embedding fixture cases found");
    for root in roots {
        run_case(&root);
    }
}
