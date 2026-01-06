#![cfg(feature = "google-vertex")]

use serde::de::DeserializeOwned;
use serde_json::Value;
use siumai::experimental::core::ProviderSpec;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vertex")
        .join("imagen")
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

fn vertex_ctx() -> siumai::experimental::core::ProviderContext {
    let base_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google";
    let mut extra = std::collections::HashMap::new();
    extra.insert("Authorization".to_string(), "Bearer token".to_string());
    siumai::experimental::core::ProviderContext::new("vertex", base_url.to_string(), None, extra)
}

fn run_generate_case(root: &Path) {
    let req: siumai::prelude::unified::ImageGenerationRequest =
        read_json(root.join("request.json"));
    let expected_body: Value = read_json(root.join("expected_body.json"));
    let expected_url = read_text(root.join("expected_url.txt")).trim().to_string();
    let raw_response: Value = read_json(root.join("response.json"));
    let expected_response: Value = read_json(root.join("expected_response.json"));

    let ctx = vertex_ctx();
    let spec = siumai::experimental::providers::google_vertex::standards::vertex_imagen::VertexImagenStandard::new().create_spec("vertex");

    let url = spec.image_url(&req, &ctx);
    assert_eq!(url, expected_url);

    let transformers = spec.choose_image_transformers(&req, &ctx);
    let got_body = transformers.request.transform_image(&req).unwrap();
    assert_eq!(got_body, expected_body);

    let got = transformers
        .response
        .transform_image_response(&raw_response)
        .unwrap();

    let mut got_value = serde_json::to_value(got).unwrap();
    let mut expected_value = expected_response;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(got_value, expected_value);
}

#[test]
fn vertex_imagen_generate_fixtures_match() {
    let roots = case_dirs(&fixtures_dir().join("generate"));
    assert!(!roots.is_empty(), "no generate fixture cases found");
    for root in roots {
        run_generate_case(&root);
    }
}

fn run_edit_case(root: &Path) {
    use siumai::experimental::execution::transformers::request::ImageHttpBody;

    let req: siumai::prelude::extensions::types::ImageEditRequest =
        read_json(root.join("request.json"));
    let expected_body: Value = read_json(root.join("expected_body.json"));
    let expected_url = read_text(root.join("expected_url.txt")).trim().to_string();
    let raw_response: Value = read_json(root.join("response.json"));
    let expected_response: Value = read_json(root.join("expected_response.json"));

    let ctx = vertex_ctx();
    let spec = siumai::experimental::providers::google_vertex::standards::vertex_imagen::VertexImagenStandard::new().create_spec("vertex");

    let url = spec.image_edit_url(&req, &ctx);
    assert_eq!(url, expected_url);

    let selector = siumai::prelude::unified::ImageGenerationRequest {
        model: req.model.clone(),
        ..Default::default()
    };
    let transformers = spec.choose_image_transformers(&selector, &ctx);
    let body = transformers.request.transform_image_edit(&req).unwrap();
    let ImageHttpBody::Json(got_body) = body else {
        panic!("expected json body for Vertex Imagen");
    };
    assert_eq!(got_body, expected_body);

    let got = transformers
        .response
        .transform_image_response(&raw_response)
        .unwrap();

    let mut got_value = serde_json::to_value(got).unwrap();
    let mut expected_value = expected_response;
    normalize_json(&mut got_value);
    normalize_json(&mut expected_value);
    assert_eq!(got_value, expected_value);
}

#[test]
fn vertex_imagen_edit_fixtures_match() {
    let roots = case_dirs(&fixtures_dir().join("edit"));
    assert!(!roots.is_empty(), "no edit fixture cases found");
    for root in roots {
        run_edit_case(&root);
    }
}
